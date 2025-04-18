import os
import re
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)

from sklearn.model_selection import train_test_split  # 여기가 핵심!
from sklearn.metrics import accuracy_score, f1_score
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from datasets import load_dataset
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_mrc_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# 모델 설정들 (기본 OLMo 1B, OLMo 7B)
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="full-OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_mrc_results/full-olmo1B-org-klue-mrc",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-1b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_mrc_results/full-olmo1B-v12-klue-mrc",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_mrc_results/full-olmo7B-org-klue-mrc",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_mrc_results/full-olmo7B-v13-klue-mrc",
    #     is_local=True
    # ),
    #     ModelConfig(
    #     name="full-Llama-3.2:3B", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_mrc_results/full-llama3.2-3b-klue-mrc",
    #     is_local=True
    # ),
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        output_dir="klue_mrc_results/lora-llama3.2-3b-it-klue-mrc",
        is_local=True,
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        output_dir="klue_mrc_results/lora-llama3.1-8b-it-klue-mrc",
        is_local=True
    ),
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_mrc_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_train.json"
MAX_LENGTH = 1024  # Increased length for MRC which often has longer contexts

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

def train_model(model_config):
    """Train the model for machine reading comprehension using language modeling approach."""
    logger.info(f"Starting training for {model_config.name} (MRC Task)")
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_config)

    # --- 데이터 로딩 (datasets 라이브러리 사용) ---
    logger.info(f"Loading dataset from {JSON_TRAIN_DATASET_PATH}...")
    try:
        # 파일 전체가 리스트 형태라고 가정하고 로드 후 Dataset으로 변환
        with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        from datasets import Dataset # datasets.Dataset import
        full_dataset = Dataset.from_list(data_list)
        logger.info(f"Loaded dataset with {len(full_dataset)} samples.")

        # --- 데이터셋 분할 (datasets 라이브러리 기능 사용) ---
        logger.info("Splitting dataset into train/validation (90/10)...")
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
        raw_train_dataset = split_dataset["train"]
        raw_val_dataset = split_dataset["test"]
        logger.info(f"Split complete - Train: {len(raw_train_dataset)} examples, Validation: {len(raw_val_dataset)} examples")

    except Exception as e:
        logger.error(f"Failed to load or split dataset: {e}")
        raise e

    # --- 전처리 함수 정의 (기존 Dataset 클래스의 __getitem__ 로직 활용) ---
    def preprocess_mrc_data(examples, tokenizer, max_length=MAX_LENGTH):
        processed_examples = {'input_ids': [], 'attention_mask': [], 'labels': []}
        skipped_count = 0

        # examples는 딕셔너리 형태의 배치
        for i in range(len(examples['context'])): # 'context' 키가 있다고 가정 (데이터 구조 확인 필요)
            title = examples['title'][i] if 'title' in examples else ""
            context = examples['context'][i] if 'context' in examples else ""
            question = examples['question'][i] if 'question' in examples else ""
            # answers 구조 확인 필요: 'answers': {'text': ['ans']}
            answers = examples['answers'][i] if 'answers' in examples else {'text': [""]}
            answer_text = answers.get("text", [""])[0] if isinstance(answers.get("text"), list) and answers.get("text") else ""

            if not context or not question:
                skipped_count += 1
                continue

            prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
            completion = f" {answer_text}"
            full_text = prompt + completion

            encoded = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding="max_length", # 콜레이터에서 처리하므로 False 가능
            )
            prompt_encoded = tokenizer(
                prompt, truncation=True, max_length=max_length,
                # padding=False # 프롬프트 길이는 가변적이므로 패딩 불필요
            )

            labels = list(encoded["input_ids"]) # 리스트로 변환
            prompt_length = len(prompt_encoded["input_ids"]) # 패딩 안된 길이 사용
            for k in range(min(prompt_length, len(labels))): # 길이 초과 방지
                 labels[k] = -100

            processed_examples['input_ids'].append(encoded['input_ids'])
            processed_examples['attention_mask'].append(encoded['attention_mask'])
            processed_examples['labels'].append(labels) # 리스트로 추가

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} examples in this batch.")

        return processed_examples
    # --- 전처리 함수 정의 끝 ---

    # --- 전처리 (datasets.map 사용) ---
    logger.info("Preprocessing datasets...")

    # functools.partial 대신 lambda 사용
    tokenized_train_dataset = raw_train_dataset.map(
        lambda examples: preprocess_mrc_data(examples, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=raw_train_dataset.column_names # 원본 컬럼 제거
    )
    tokenized_val_dataset = raw_val_dataset.map(
        lambda examples: preprocess_mrc_data(examples, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=raw_val_dataset.column_names
    )
    logger.info(f"Tokenized datasets prepared: train={len(tokenized_train_dataset)}, validation={len(tokenized_val_dataset)}")

    # Data collator (Causal LM용)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments (기존 설정 유지)
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps", 
        eval_steps=200, 
        learning_rate=2e-5,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, 
        num_train_epochs=2, 
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
        report_to="none", 
        gradient_checkpointing=True, 
        optim="adamw_torch",
    )

    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset, # <-- 수정됨
        eval_dataset=tokenized_val_dataset,   # <-- 수정됨
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
        # compute_metrics는 Causal LM 학습 시 생략 가능
    )

    # Start training
    logger.info("Starting training")
    trainer.train()

    # Save final model and tokenizer
    final_model_path = os.path.join(model_config.output_dir, "final_model") # 경로 일관성
    logger.info(f"Saving final best model to: {final_model_path}")
    trainer.save_model(final_model_path) # save_pretrained 대신 사용
    tokenizer.save_pretrained(final_model_path)

    logger.info(f"Training completed for {model_config.name}")

# Main execution function
if __name__ == "__main__":
    logger.info("--- Starting KLUE-MRC Model Training Script ---")

    # Process each model configuration
    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        model = None; tokenizer = None # 초기화

        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            # Train model only
            train_model(model_config)
            logger.info(f"--- Completed training for {model_config.name} ---")

        except Exception as e:
            logger.error(f"Error training model {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
        finally:
            # Memory cleanup
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del model; del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info(f"Resource cleanup finished for {model_config.name}.")
            print("-" * 50)

    logger.info("--- KLUE-MRC Model Training Script Finished ---")