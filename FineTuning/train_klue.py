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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import re
from tqdm import tqdm
import evaluate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 및 하이퍼파라미터 설정
MODEL_PATH = "olmo7B-v12-80000"
OUTPUT_DIR = "olmo7B-v12-klue-sts"
DATA_CACHE_DIR = "./klue_sts_cache"
JSON_DATASET_PATH = "./klue_sts.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

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

# 메인 학습 함수
def train_model():
    # 데이터셋 준비
    prepare_dataset_json()
    
    # 데이터셋 로드
    logger.info("JSON loading...")
    with open(JSON_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    train_data = data["train"]
    val_data = data["validation"]
    
    logger.info(f"train data: {len(train_data)}, valid data: {len(val_data)}")
    
    # 모델 및 토크나이저 로드
    logger.info(f"Load model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # bfloat16 정밀도로 모델 로드 (메모리 효율성 증가)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # 자동으로 GPU에 모델 분산
    )
    
    # 학습용 데이터셋 생성
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            prompt = item["input"]
            completion = item["output"]
            
            # 프롬프트와 완성 결합
            full_text = prompt + completion
            
            # 토큰화
            encoded = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 프롬프트 부분 토큰화 (라벨 마스킹용)
            prompt_encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 라벨 생성: 프롬프트 부분은 -100으로 마스킹
            labels = encoded["input_ids"].clone().squeeze(0)
            prompt_length = prompt_encoded["input_ids"].shape[1]
            labels[:prompt_length] = -100
            
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": labels
            }
    
    # 데이터셋 생성
    train_dataset = SimpleDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = SimpleDataset(val_data, tokenizer, MAX_LENGTH)
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 학습 하이퍼파라미터 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,  # bfloat16 정밀도 사용
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,  # 메모리 절약을 위한 그래디언트 체크포인팅
        optim="adamw_torch",  # PyTorch 구현 사용
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )
    
    # 학습 실행
    trainer.train()
    
    # 최종 모델 및 토크나이저 저장
    final_model_path = os.path.join(OUTPUT_DIR, "final")
    logger.info(f"Final Model: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Tuned!")
    return model, tokenizer

# 모델 평가 함수
def evaluate_model(model, tokenizer):
    logger.info("Evaluating the model...")
    
    # KLUE STS 데이터셋 로드
    klue_sts = load_dataset("klue", "sts", cache_dir=DATA_CACHE_DIR)
    
    # 평가용 하위 집합
    val_subset = klue_sts["validation"]
    
    model.eval()
    
    true_scores = []
    pred_scores = []

    logs = []
    log_file_path = os.path.join(OUTPUT_DIR, "log.json")

    for item in tqdm(val_subset):
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        true_score = item["labels"]["label"]
        
        # 프롬프트 생성
        prompt = f"Analyze the following sentence pairs and provide a similarity score between 0 and 5, where 0 means completely different and 5 means identical in meaning. Sentence 1: {sentence1} Sentence 2: {sentence2}"
        
        # 토큰화
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
        match = re.search(r'similarity score is (\d+(\.\d+)?)', generated_text)
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
            # "bleu": float(bleu_score['bleu']),
            # "rouge": float(rouge_score['rouge']),
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
    
    # 모델 학습
    model, tokenizer = train_model()

    # 모델 평가
    evaluate_model(model, tokenizer)