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
)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from functools import partial # partial import 추가

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
        name="OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_sts_results/olmo1B-org-klue-sts",
        is_local=False
    ),
    ModelConfig(
        name="OLMo-1b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
        output_dir="klue_sts_results/olmo1B-v12-klue-sts",
        is_local=True
    ),
    ModelConfig(
        name="OLMo-7b-org", 
        model_path="allenai/OLMo-7B", 
        output_dir="klue_sts_results/olmo7B-org-klue-sts",
        is_local=False
    ),
    ModelConfig(
        name="OLMo-7b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
        output_dir="klue_sts_results/olmo7B-v13-klue-sts",
        is_local=True
    ),
    # ModelConfig(
    #     name="BERT-base-uncased",
    #     model_path="bert-base-uncased",
    #     is_local=False,
    #     output_dir="klue_sts_results/BERT-base-uncased-klue-sts",
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_sts_results/BERT-base-uncased-Tuned-klue-sts",
    # ),
    ModelConfig(
        name="Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_sts_results/llama3.2-3b-klue-sts",
        is_local=True
    ),
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        is_local=True,
        output_dir="klue_sts_results/llama3.2-3b-it-klue-sts",
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        is_local=True,
        output_dir="klue_sts_results/llama3.1-8b-it-klue-sts",
    ),
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

# 메인 학습 함수 수정
def train_model(model_config):
    """
    모델을 학습시킵니다.
    JSON_TRAIN_DATASET_PATH를 90% 학습 / 10% 검증으로 분할하여 사용합니다.
    """
    # --- 데이터 전처리 함수 정의 (train_model 내부로 이동 또는 확인) ---
    def preprocess_function_for_training(examples, tokenizer, max_length=MAX_LENGTH):
        """학습용 데이터 전처리: input과 output을 결합하여 Causal LM 형식으로 만듭니다."""
        # 입력이나 출력이 None인 경우 처리 (데이터 클리닝 강화)
        texts = []
        for inp, outp in zip(examples["input"], examples["output"]):
            if inp is None or outp is None:
                # logger.warning(f"Skipping sample with None input/output: inp='{inp}', outp='{outp}'")
                # None 대신 빈 문자열로 처리하거나, 샘플을 건너뛰도록 로직 추가 가능
                # 여기서는 예시로 빈 문자열 처리
                inp = inp or ""
                outp = outp or ""

            # 토크나이저의 eos_token 확인
            eos_tok = tokenizer.eos_token
            if eos_tok is None:
                # 만약 eos_token이 없으면 다른 토큰 사용 (예: sep_token) 또는 에러 발생
                eos_tok = tokenizer.sep_token # 예: sep_token 사용
                if eos_tok is None:
                     raise ValueError("Tokenizer has neither eos_token nor sep_token.")

            texts.append(inp + eos_tok + outp + eos_tok) # 여기서 eos_tok 사용

        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    # --- 전처리 함수 정의 끝 ---

    logger.info(f"Starting training for {model_config.name}")
    logger.info(f"Splitting {JSON_TRAIN_DATASET_PATH} into 90% train / 10% validation for training.")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)

    # --- 데이터 로딩 및 분할 ---
    logger.info(f"Loading dataset to be split from: {JSON_TRAIN_DATASET_PATH}")
    try:
        full_train_val_dataset = load_dataset("json", data_files={"train": JSON_TRAIN_DATASET_PATH}, cache_dir=DATA_CACHE_DIR, split="train")
        logger.info(f"Loaded dataset with {len(full_train_val_dataset)} samples.")
        split_dataset = full_train_val_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
        raw_train_dataset = split_dataset["train"]
        raw_val_dataset_for_training = split_dataset["test"]
        logger.info(f"Split dataset: train={len(raw_train_dataset)}, validation_during_training={len(raw_val_dataset_for_training)}")
    except Exception as e:
        logger.error(f"Failed to load or split dataset from {JSON_TRAIN_DATASET_PATH}: {e}")
        raise

    # --- 전처리 ---
    logger.info("Preprocessing training dataset...")
    # partial 사용 시점에 preprocess_function_for_training이 정의되어 있어야 함
    preprocess_with_tokenizer = partial(preprocess_function_for_training, tokenizer=tokenizer, max_length=MAX_LENGTH)
    tokenized_train_dataset = raw_train_dataset.map(preprocess_with_tokenizer, batched=True, remove_columns=raw_train_dataset.column_names)
    logger.info(f"Tokenized training dataset prepared: {len(tokenized_train_dataset)} examples")

    logger.info("Preprocessing validation dataset (used during training)...")
    tokenized_val_dataset_for_training = raw_val_dataset_for_training.map(preprocess_with_tokenizer, batched=True, remove_columns=raw_val_dataset_for_training.column_names)
    logger.info(f"Tokenized validation dataset prepared: {len(tokenized_val_dataset_for_training)} examples")

    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments (검증 설정 유지)
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=400,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5, # 필요시 epoch 조정
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # 트레이너 초기화
    logger.info("Initializing Trainer with split train/validation datasets...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset_for_training,
        data_collator=data_collator,
    )

    # 학습 실행
    logger.info("Starting model training with periodic validation...")
    trainer.train()
    logger.info("Training finished.")

    # 최종 모델 및 토크나이저 저장 (최고 성능 모델 저장)
    final_model_path = os.path.join(model_config.output_dir, "final_model") # 경로 일관성 위해 이름 변경
    logger.info(f"Saving final best model (based on validation subset) to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info("Training and saving best model completed!")


# 메인 실행 함수
if __name__ == "__main__":
    # 각 모델별로 학습 실행
    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        # 출력 디렉토리 생성
        os.makedirs(model_config.output_dir, exist_ok=True)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        # 변수 초기화 (필요시)
        model = None
        tokenizer = None
        trainer = None # try 블록 내에서만 사용되므로 사실상 불필요

        try:
            # 모델 학습만 실행
            train_model(model_config) # 반환값 받지 않음

            logger.info(f"--- Completed training for {model_config.name} ---") # 로그 메시지 수정

        except Exception as e:
            logger.error(f"Error processing model {model_config.name}: {e}")
            logger.exception("Exception details:")
        finally:
            # --- 메모리 정리 (Trainer는 train_model 내에서 소멸) ---
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del model
            del tokenizer
            # del trainer # train_model 함수 종료 시 자동으로 참조 해제됨
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache.")
            logger.info(f"Resource cleanup finished for {model_config.name}.")
            print("-" * 50)