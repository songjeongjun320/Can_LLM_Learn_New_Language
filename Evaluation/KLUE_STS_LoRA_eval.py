# Standard library imports
import json
import logging
import os
import re
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from peft.utils.other import fsdp_auto_wrap_policy

from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정 클래스
class ModelConfig:
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir

# 모델 설정들
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-olmo1B-org-klue-sts", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_sts_results/olmo1B-lora-org-klue-sts"
    ),
    # ModelConfig(
    #     name="lora-olmo1B-v12-klue-sts", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo1B-v12", 
    #     output_dir="klue_sts_results/lora-olmo1B-v12-klue-sts"
    # ),
    # ModelConfig(
    #     name="lora-olmo7B-org-klue-sts", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_sts_results/lora-olmo7B-org-klue-sts"
    # ),
    # ModelConfig(
    #     name="lora-olmo7B-v13-klue-sts", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo7B-v13", 
    #     output_dir="klue_sts_results/lora-olmo7B-v13-klue-sts"
    # ),
    # ModelConfig(
    #     name="lora-llama3.2:3b-klue-sts", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_sts_results/lora-llama3.2:3b-klue-sts"
    # )
]

# 기본 설정
DATA_CACHE_DIR = "./klue_sts_origin_cache"
JSON_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_sts.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

# 데이터셋 준비 함수 (이전 스크립트와 동일)
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
    
    # 검증 데이터 준비
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

# 모델 및 토크나이저 로드 함수
def load_model_and_tokenizer(model_config):
    """LoRA를 위한 모델과 토크나이저 로드"""
    logger.info(f"Load model: {model_config.model_path}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path, trust_remote_code=True)
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 자동으로 GPU에 모델 분산
        trust_remote_code=True  # OLMo 모델에 필요
    )
    
    # 나머지 코드는 동일하게 유지
    model = prepare_model_for_kbit_training(model)
    print("Check the model architecture")
    # print(model)
    
    model = prepare_model_for_kbit_training(model)
    # model.print_trainable_parameters() # Only for OLMo
    
    return model, tokenizer

# 학습 데이터셋 클래스
class SimpleDataset(torch.utils.data.Dataset):
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

# 메인 학습 함수
def train_model(model_config):
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
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    train_dataset = Dataset.from_dict({
        "text": [f"{item['input']}{item['output']}" for item in train_data]
    })
    val_dataset = Dataset.from_dict({
        "text": [f"{item['input']}{item['output']}" for item in val_data]
    })
    
    # LoRA 설정 추가
    peft_params = LoraConfig(
        lora_alpha=16,  # LoRA 스케일링 팩터
        lora_dropout=0.1,  # LoRA 드롭아웃 비율
        r=64,  # LoRA 랭크
        bias="none",  
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"],  # OLMo model attention layer targetting
        # target_modules=["q_proj", "k_proj"]  # Llama model
    )

    # 모델 및 토크나이저 로드 시 LoRA 설정 적용
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
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
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # SFTTrainer 초기화 시 tokenizer와 packing 제거
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_params,
    )

    # 학습 실행
    logger.info("Starting training...")
    trainer.train()
    
    # 최종 모델 저장 (PEFT 모델로)
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    
    # PEFT 모델과 토크나이저 저장
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Fine-tuning completed!")
    return model, tokenizer

# 모델 평가 함수 (이전 스크립트와 거의 동일)
def evaluate_model(model, tokenizer, model_config):
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

        # 간단한 정규식으로 0-5 사이의 소수점 숫자 추출
        matches = re.findall(r'([0-5](?:\.\d+)?)', generated_text)
        predicted_score = None
        
        # "The similarity score is" 다음에 오는 숫자를 찾아보기
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
            "input": sentence1 + " | " + sentence2,
            "generated_text": generated_text,
            "true_score": true_score,
            "predicted_score": predicted_score
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

        # 평가 결과를 파일에 저장
        eval_results = {
            "model": model_config.name,
            "pearson_correlation": float(pearson_corr),
            "rmse": float(rmse),
            "mae": float(mae),
            "mse": float(mse),
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