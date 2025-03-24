import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import re
from tqdm import tqdm

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

# 기본 설정
DATA_CACHE_DIR = "./klue_sts_origin_cache"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

# 평가할 모델 경로
PRETRAINED_MODEL_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/olmo1B-org-klue-sts/final"
OUTPUT_DIR = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/olmo1B-org-klue-sts"
MODEL = "olmo1B-org-klue-sts"

# 모델 및 토크나이저 로드 함수
def load_model_and_tokenizer(model_path):
    """주어진 경로에서 모델과 토크나이저를 로드합니다."""
    logger.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # bfloat16 정밀도로 모델 로드 (메모리 효율성 증가)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 자동으로 GPU에 모델 분산
        trust_remote_code=True  # OLMo 모델에 필요
    )
    
    return model, tokenizer

# 모델 평가 함수
def evaluate_model(model, tokenizer, model_name="OLMo-7B-finetuned"):
    logger.info(f"Evaluating the model: {model_name}")
    
    # KLUE STS 데이터셋 로드
    klue_sts = load_dataset("klue", "sts", cache_dir=DATA_CACHE_DIR)
    
    # 평가용 하위 집합
    val_subset = klue_sts["validation"]

    model.eval()
    
    true_scores = []
    pred_scores = []

    log_file_path = os.path.join(OUTPUT_DIR, "evaluation_log.json")
    logs = []

    for item in tqdm(val_subset):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        true_score = item["labels"]["label"]
        
        # 프롬프트 생성
        prompt = f"Analyze the following sentence pairs and provide a similarity score between 0 and 5, where 0 means completely different and 5 means identical in meaning. Sentence 1: {sentence1} Sentence 2: {sentence2}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 추론
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )

        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 점수 추출 (실수 처리)
        match = re.search(r'\b([0-4]\.\d+|5\.0|0\.0)\b', generated_text)
        
        predicted_score = None
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
            "true_score": float(true_score),  # 실제 점수
            "predicted_score": predicted_score  # 예측 점수
        }
        
        logs.append(log_data)

    # 로그를 파일에 저장
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # 메트릭 계산
    if true_scores and pred_scores:
        logger.info(f"Eval result:")
        logger.info(f"Eval Model: {model_name}")
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

        # 평가 결과를 파일에 저장
        eval_results = {
            "model": model_name,
            "pearson_correlation": float(pearson_corr),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
            "num_samples": len(true_scores)
        }

        # 결과를 JSON 파일로 저장
        eval_file_path = os.path.join(OUTPUT_DIR, "eval_results.json")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
    else:
        logger.warning("No valid prediction.")

# 메인 실행 함수
if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    logger.info(f"Starting evaluation for model at: {PRETRAINED_MODEL_PATH}")
    
    try:
        # 모델 로드
        model, tokenizer = load_model_and_tokenizer(PRETRAINED_MODEL_PATH)
        
        # 모델 평가
        evaluate_model(model, tokenizer, MODEL)
        
        logger.info(f"Completed evaluation successfully")
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.exception("Exception details:")