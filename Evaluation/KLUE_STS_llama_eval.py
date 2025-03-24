import logging
import json
import torch
import os
from tqdm import tqdm
import ollama  # Ollama 클라이언트 사용
import re
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datasets import load_dataset
import evaluate


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정 클래스
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_ollama=False, ollama_host=None):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_ollama = is_ollama
        self.ollama_host = ollama_host
        self.ollama_client = None
        self.ollama_model_name = model_path  # Ollama 모델 이름 설정

# 모델 설정들 (Ollama 사용 시)
MODEL_CONFIGS = [
    ModelConfig(
        name="llama3.2", 
        model_path="llama3.2", 
        output_dir="llama3.2-klue-sts",
        is_ollama=True,
        ollama_host="http://sg046:11435"  # Ollama 서버 주소
    )
]

DATA_CACHE_DIR = "./klue_sts_LLAMA_cache"
JSON_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/fine-tuned-models/klue_sts.json"

# Ollama 서버 연결 함수
def connect_ollama_client(ollama_host):
    """Ollama 클라이언트를 생성하고 연결합니다."""
    try:
        client = ollama.Client(host=ollama_host)  # Ollama 클라이언트 생성
        logger.info(f"Connected to Ollama server at {ollama_host}")
        return client
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        return None

# 데이터셋 준비 함수 - JSON 파일 생성
def prepare_dataset_json():
    """KLUE STS 데이터셋을 불러와서 JSON 파일로 변환합니다."""
    if os.path.exists(JSON_DATASET_PATH):
        logger.info(f"Dataset already exists: {JSON_DATASET_PATH}")
        return
    
    logger.info("KLUE STS dataset loading...")
    klue_sts = load_dataset("klue", "sts", cache_dir=DATA_CACHE_DIR)
    
    # 학습 및 검증 데이터를 위한 리스트
    train_samples = []
    val_samples = []
    
    # 함수: 프롬프트와 완성 텍스트 생성
    def create_prompt(sentence1, sentence2):
        return f"Analyze the following sentence pairs and provide a similarity score between 0 and 5, where 0 means completely different and 5 means identical in meaning. Sentence 1: {sentence1} Sentence 2: {sentence2}"

    def create_completion(score):
        # 정수로 반올림된 점수 반환
        return f" The similarity score is {score}"
    
    # 학습 데이터 준비
    logger.info("Creating Klue_dataset.json ...")
    for item in tqdm(klue_sts["train"]):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        score = item["labels"]["label"]  # 0-5 척도
        
        # 스코어 정규화
        normalized_score = max(0, min(5, score))
        
        sample = {
            "input": create_prompt(sentence1, sentence2),
            "output": create_completion(normalized_score)
        }
        train_samples.append(sample)
    
    # 검증 데이터 준비 (메모리 절약을 위해 일부만 사용)
    logger.info("Translating Valid data...")
    val_subset = klue_sts["validation"]
    for item in tqdm(val_subset):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        score = item["labels"]["label"]
        
        normalized_score = max(0, min(5, score))
        
        sample = {
            "input": create_prompt(sentence1, sentence2),
            "output": create_completion(normalized_score)
        }
        val_samples.append(sample)
    
    # JSON 파일로 저장
    dataset = {
        "train": train_samples,
        "validation": val_samples
    }
    
    logger.info(f"JSON dataset saving... (train: {len(train_samples)}, valid: {len(val_samples)})")
    with open(JSON_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created klue_sts dataset: {JSON_DATASET_PATH}")

# 모델 평가 함수 (Ollama API 지원)
def evaluate_model(model_config):
    logger.info(f"Evaluating the model: {model_config.name}")
    
    # Ollama 클라이언트 연결
    client = connect_ollama_client(model_config.ollama_host)
    if not client:
        logger.error("Ollama client connection failed.")
        return
    
    # KLUE STS 데이터셋 로드
    klue_sts = load_dataset("klue", "sts", cache_dir=DATA_CACHE_DIR)
    val_subset = klue_sts["validation"]
    
    true_scores = []
    pred_scores = []
    
    log_file_path = os.path.join(model_config.output_dir, "log.json")
    logs = []

    for item in tqdm(val_subset):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        true_score = item["labels"]["label"]

        # 몇 가지 예시를 제공하는 방식으로 프롬프트를 수정
        prompt = f"""
        Example 1:
        Sentence 1: "I love programming."
        Sentence 2: "I enjoy coding."
        Similarity score: 4.5

        Now, analyze the following sentence pair and MUST provide a similarity score between 0 and 5, where 0 means completely different and 5 means identical in meaning.
        Give me only predicted float number between 0 and 5.

        Sentence 1: {sentence1}
        Sentence 2: {sentence2}
        """
                
        try:
            response = client.chat(
                model=model_config.ollama_model_name,  # 사용하려는 모델 이름
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.1,  # 낮은 온도로 일관된 결과
                    'top_p': 0.95,
                }
            )
            generated_text = response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            continue

        # 점수 추출 (실수 처리)
        match = re.search(r'\b([0-4]\.\d+|5\.0|0\.0)\b', generated_text)
        
        if match:
            predicted_score = float(match.group(1))  # 실수로 변환
            if 0 <= predicted_score <= 5:
                true_scores.append(true_score)
                pred_scores.append(predicted_score)
            else:
                logger.warning(f"Out of scope (0-5): {predicted_score}")
        else:
            logger.warning(f"Can't extract the label: {generated_text}")

        # 로그 저장을 위한 데이터
        log_data = {
            "input": sentence1 + " | " + sentence2,  # 입력 문장 쌍
            "generated_text": generated_text,  # 생성된 텍스트
            "true_score": true_score,  # 실제 점수
            "predicted_score": predicted_score  # 예측 점수
        }
        
        logs.append(log_data)

    # 로그를 파일에 저장
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 메트릭 계산
    if true_scores and pred_scores:
        logger.info(f"Eval result:")
        logger.info(f"Eval Model: {model_config.model_path}")
        logger.info(f"Evaluated samples: {len(true_scores)}")

        # Pearson Correlation 계산
        pearson_corr = np.corrcoef(true_scores, pred_scores)[0, 1]
        logger.info(f"Pearson Correction: {pearson_corr:.4f}")
        
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
        logger.info(f"RMSE: {rmse:.4f}")
        
        # MAE 계산
        mae = mean_absolute_error(true_scores, pred_scores)
        logger.info(f"MAE: {mae:.4f}")
        
        # MSE 계산
        mse = mean_squared_error(true_scores, pred_scores)
        logger.info(f"MSE: {mse:.4f}")

        # 평가 지표 로드
        # bleu_metric = evaluate.load("bleu")  # ✅ 변경된 부분
        # bleu_score = bleu_metric.compute(predictions=[pred_scores], references=[true_scores])
        # rouge_metric = evaluate.load("rouge")  # ✅ 변경된 부분
        # rouge_score = rouge_metric.compute(predictions=[pred_scores], references=[true_scores])
        
        # # 실제로는 여러 개의 예측문장이 있을 수 있으므로, 적절한 형식으로 데이터를 맞춰야 함
        # bleu_score = bleu_metric.compute(predictions=[pred_scores], references=[true_scores])
        # logger.info(f"BLEU: {bleu_score['bleu']:.4f}")
        # rouge_score = rouge_metric.compute(predictions=[pred_scores], references=[true_scores])
        # logger.info(f"ROUGE: {rouge_score['rouge']:.4f}")

        # 평가 결과를 파일에 저장
        eval_results = {
            "model": model_config.name,
            "pearson_correlation": float(pearson_corr),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "num_samples": len(true_scores)
        }

        eval_file_path = os.path.join(model_config.output_dir, "eval_results.json")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
    else:
        logger.warning("No valid prediction.")

# 메인 실행 함수
if __name__ == "__main__":
    # 각 모델별로 평가 실행
    for model_config in MODEL_CONFIGS:
        os.makedirs(model_config.output_dir, exist_ok=True)
        
        logger.info(f"Starting evaluation for {model_config.name}")
        
        try:
            # 모델 평가 (Ollama 클라이언트만 사용)
            evaluate_model(model_config)
            
            logger.info(f"Completed evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")
