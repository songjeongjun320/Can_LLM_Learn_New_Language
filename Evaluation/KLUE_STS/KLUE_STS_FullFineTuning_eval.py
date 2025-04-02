import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import re
from tqdm import tqdm
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        name="full-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_sts_results/full-olmo1B-org-klue-sts",
        is_local=False
    ),
    ModelConfig(
        name="full-OLMo-1b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
        output_dir="klue_sts_results/full-olmo1B-v12-klue-sts",
        is_local=True
    ),
    ModelConfig(
        name="full-OLMo-7b-org", 
        model_path="allenai/OLMo-7B", 
        output_dir="klue_sts_results/full-olmo7B-org-klue-sts",
        is_local=False
    ),
    ModelConfig(
        name="full-OLMo-7b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
        output_dir="klue_sts_results/full-olmo7B-v13-klue-sts",
        is_local=True
    ),
        ModelConfig(
        name="full-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_sts_results/full-llama3.2-3b-klue-sts",
        is_local=True
    )
]

# 기본 설정
DATA_CACHE_DIR = "./klue_sts_origin_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_validation.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

def prepare_dataset_json():
    """이미 저장된 JSON 파일에서 KLUE STS 데이터셋을 불러와서 전처리한 후, 통합 JSON 파일로 저장합니다."""
    if os.path.exists(JSON_DATASET_PATH):
        logger.info(f"Dataset already exists: {JSON_DATASET_PATH}")
        return

    # JSON 파일 존재 여부 확인 및 불러오기
    if not os.path.exists(JSON_TRAIN_DATASET_PATH):
        logger.error(f"Train dataset file does not exist: {JSON_TRAIN_DATASET_PATH}")
        return
    if not os.path.exists(JSON_VAL_DATASET_PATH):
        logger.error(f"Validation dataset file does not exist: {JSON_VAL_DATASET_PATH}")
        return

    logger.info("Loading train dataset from JSON file...")
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    logger.info("Loading validation dataset from JSON file...")
    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_samples = []
    val_samples = []

    def create_prompt(sentence1, sentence2):
        return (
            f"Analyze the following sentence pairs and provide a similarity score between 0 and 5, "
            f"where 0 means completely different and 5 means identical in meaning. "
            f"Sentence 1: {sentence1} Sentence 2: {sentence2}"
        )

    def create_completion(score):
        return f" The similarity score is {score}"

    logger.info("Processing train data...")
    for item in tqdm(train_data):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        score = item["labels"]["label"]  # 0-5 척도
        normalized_score = max(0, min(5, score))
        sample = {
            "input": create_prompt(sentence1, sentence2),
            "output": create_completion(normalized_score)
        }
        train_samples.append(sample)

    logger.info("Processing validation data...")
    for item in tqdm(val_data):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        score = item["labels"]["label"]
        normalized_score = max(0, min(5, score))
        sample = {
            "input": create_prompt(sentence1, sentence2),
            "output": create_completion(normalized_score)
        }
        val_samples.append(sample)

    dataset = {
        "train": train_samples,
        "validation": val_samples
    }

    logger.info(f"Saving combined JSON dataset... (train: {len(train_samples)}, valid: {len(val_samples)})")
    with open(JSON_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Created klue_sts dataset: {JSON_DATASET_PATH}")


# 데이터 전처리 함수
def preprocess_function(examples, tokenizer, max_length=MAX_LENGTH):
    # 프롬프트 형식
    inputs = tokenizer([ex for ex in examples["input"]], 
                      truncation=True, 
                      max_length=max_length,
                      padding="max_length",
                      return_tensors="pt")
    
    # 출력 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([ex for ex in examples["output"]], 
                         truncation=True, 
                         max_length=128,  # 출력은 짧기 때문에 더 작은 길이 사용
                         padding="max_length",
                         return_tensors="pt")
    
    # 라벨 처리: -100은 손실 계산에서 무시됨
    for i in range(len(labels["input_ids"])):
        labels["input_ids"][i][labels["input_ids"][i] == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }

# Model and tokenizer loading function
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model: {model_config.model_path}")

    is_local=False
    if (model_config.is_local):
        is_local = True

    # 일반적인 HuggingFace 모델 로드 (OLMo 1B, OLMo 7B)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, 
        local_files_only=is_local,
        trust_remote_code=True
        )
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # bfloat16 정밀도로 모델 로드 (메모리 효율성 증가)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 자동으로 GPU에 모델 분산
        local_files_only=is_local,
        trust_remote_code=True  # OLMo 모델에 필요
    )
    
    return model, tokenizer

# 메인 학습 함수
def train_model(model_config):
    # Compute metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_f1_support(labels, predictions, average="macro", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    """Train the model for RE using sequence classification approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    full_train_data = REDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.2,  # Val 20%
        random_state=42,  # 재현성 보장
        shuffle=True
    )
    logger.info(f"Loaded data - train: {len(train_data)} examples, validation: {len(val_data)} examples")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 학습용 데이터셋 생성
    from torch.utils.data import Dataset

    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # 배치 크기 증가
        per_device_eval_batch_size=8,  # 배치 크기 증가
        gradient_accumulation_steps=2,  # 축적 단계 감소
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False,  # FP16으로 전환
        bf16=True,  # BF16 비활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Warmup 비율 감소
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,  # 체크포인팅 비활성화
        optim="adamw_torch",  # 필요 시 "adamw_8bit"로 변경
    )
    
    # 얼리 스토핑 콜백
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5
    )
    
    # 트레이너 초기화 및 학습
    logger.info("Reset Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )
    
    # 학습 실행
    trainer.train()
    
    # 최종 모델 및 토크나이저 저장
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Final Model: {final_model_path}")
    
    # Ollama 모델이 아닌 경우에만 저장 (로컬 모델)
    if not model_config.is_ollama:
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
    else:
        # Ollama 모델의 경우 토크나이저만 저장
        tokenizer.save_pretrained(final_model_path)
        # Ollama 모델 정보 저장
        with open(os.path.join(final_model_path, "ollama_config.json"), "w") as f:
            json.dump({
                "model": model_config.model_path,
                "host": model_config.ollama_host
            }, f)
    
    logger.info("Tuned!")
    return model, tokenizer

# 모델 평가 함수 (Ollama API 지원 추가)
def evaluate_model(model, tokenizer, model_config="OLMo7b"):
    logger.info(f"Evaluating the model: {model_config.name}")
    
    # KLUE STS 데이터셋 로드
    klue_sts = load_dataset("klue", "sts", cache_dir=DATA_CACHE_DIR)
    
    # 평가용 하위 집합
    val_subset = klue_sts["validation"]

    model.eval()
    
    true_scores = []
    pred_scores = []

    log_file_path = os.path.join(model_config.output_dir, "log.json")
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

        # 간단한 정규식으로 0-5 사이의 소수점 숫자 추출 (단어 경계 없이)
        matches = re.findall(r'([0-5](?:\.\d+)?)', generated_text)
        predicted_score = None
        
        # 먼저 "The similarity score is" 다음에 오는 숫자를 찾아보기
        if "The similarity score is" in generated_text:
            after_score_text = generated_text.split("The similarity score is")[1].strip()
            score_matches = re.findall(r'([0-5](?:\.\d+)?)', after_score_text)
            if score_matches:
                try:
                    predicted_score = float(score_matches[0])
                    if 0 <= predicted_score <= 5:
                        true_scores.append(true_score)
                        pred_scores.append(predicted_score)
                    else:
                        logger.warning(f"Out of scope (0-5): {predicted_score}")
                except ValueError:
                    logger.warning(f"Failed to convert to float: {score_matches[0]}")

        # 로그 저장을 위한 데이터
        log_data = {
            "input": sentence1 + " | " + sentence2,  # 입력 문장 쌍
            "generated_text": generated_text,  # 생성된 텍스트
            "true_score": true_score,  # 실제 점수
            "predicted_score": predicted_score # 예측 점수
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
        
        # 실제로는 여러 개의 예측문장이 있을 수 있으므로, 적절한 형식으로 데이터를 맞춰야 함
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
            # "bleu": float(bleu_score['bleu']),
            # "rouge": float(rouge_score['rouge']),
            "num_samples": len(true_scores)
        }

        # 결과를 JSON 파일로 저장
        eval_file_path = os.path.join(model_config.output_dir, "eval_results.json")
        with open(eval_file_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
    else:
        logger.warning("No valid prediction.")

# 메인 실행 함수
if __name__ == "__main__":
    # 각 모델별로 학습 및 평가 실행
    for model_config in MODEL_CONFIGS:
        # 출력 디렉토리 생성
        os.makedirs(model_config.output_dir, exist_ok=True)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        
        logger.info(f"Starting training for {model_config.name}")
        
        try:
            # 모델 학습
            model, tokenizer = train_model(model_config)
            
            # 모델 평가
            evaluate_model(model, tokenizer, model_config)
            
            logger.info(f"Completed training and evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")
