import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_nli_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE NLI Label Definitions
NLI_LABELS = ["entailment", "neutral", "contradiction"]
NUM_LABELS = len(NLI_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(NLI_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(NLI_LABELS)}
logger.info(f"Total number of KLUE-NLI labels: {NUM_LABELS}")

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="full-OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_nli_results/full-olmo1B-org-klue-nli",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-1b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_nli_results/full-olmo1B-v12-klue-nli",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_nli_results/full-olmo7B-org-klue-nli",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_nli_results/full-olmo7B-v13-klue-nli",
    #     is_local=True
    # ),
    #     ModelConfig(
    #     name="full-Llama-3.2-3b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_nli_results/full-llama3.2-3b-klue-nli",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
    #     output_dir="klue_nli_results/llama3.2-3b-it-klue-nli",
    #     is_local=True,
    # ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        output_dir="klue_nli_results/llama3.1-8b-it-klue-nli",
        is_local=True
    ),
    # ModelConfig(
    #     name="BERT-uncased-origin-translation",
    #     model_path="google-bert/bert-base-uncased",
    #     output_dir="klue_nli_results/BERT-uncased-origin-klue-nli",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="BERT-uncased-kr-eng-translation",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/bert-uncased-finetuned-kr-eng",
    #     output_dir="klue_nli_results/BERT-uncased-kr-eng-translation-klue-nli",
    #     is_local=True
    # ),
]
# Configuration parameters
DATA_CACHE_DIR = "./klue_nli_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_nli_train.json"
MAX_LENGTH = 512

#  --- Model and tokenizer loading function (PEFT 제거) ---
def load_model_and_tokenizer(model_config, num_labels=NUM_LABELS):
    logger.info(f"Load model: {model_config.model_path} for Sequence Classification (NLI Task - Full Fine-tuning)")
    is_local = model_config.is_local

    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, local_files_only=is_local, trust_remote_code=True)

    # --- 패딩 토큰 설정 ---
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(f"Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id})")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        else: raise ValueError("Cannot set padding token.")
    elif tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])
    logger.info(f"Using pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # 2. 모델 설정(Config) 로드 및 수정 (모든 모델 공통)
    #    먼저 config만 로드해서 분류 헤드 정보를 넣어줍니다.
    config = AutoConfig.from_pretrained(
        model_config.model_path,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=is_local,
        trust_remote_code=True
    )
    config.pad_token_id = tokenizer.pad_token_id
    logger.info("Model config loaded and modified for sequence classification.")

    # 3. 모델 로드 (Try-Except로 분기)
    model = None
    try:
        # --- 시도 1: AutoModelForSequenceClassification 직접 로드 (Llama, BERT 등) ---
        logger.info(f"Attempting direct load with AutoModelForSequenceClassification for {model_config.name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_path,
            config=config, # 수정된 config 전달
            torch_dtype=torch.bfloat16,
            device_map="auto", # 직접 로드 시 device_map 사용 가능
            local_files_only=is_local,
            trust_remote_code=True,
            # ignore_mismatched_sizes=True # 헤드 새로 추가 시 필요할 수 있음
        )
        logger.info(f"Successfully loaded {model.__class__.__name__} directly.")

    except ValueError as e:
        # --- 시도 2: 직접 로드 실패 시 (Unrecognized Config - OLMo 등) ---
        if "Unrecognized configuration class" in str(e):
            logger.warning(f"Direct load failed for {model_config.name} ({e}). Attempting fallback: Load CausalLM -> Adapt.")

            # 3a. 기본 Causal LM 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_path,
                config=config, # 이미 분류 정보가 포함된 config 사용
                torch_dtype=torch.bfloat16,
                local_files_only=is_local,
                trust_remote_code=True
                # device_map 여기서 사용하지 않음 (헤드 추가 후 이동)
            )
            logger.info("Base Causal LM model loaded for fallback.")

            # 3b. Sequence Classification 모델로 변환 (헤드 추가)
            #    **중요:** model_args 대신 로드된 base_model 객체를 첫 인자로 전달
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, # <--- 로드된 base_model 객체를 경로 대신 사용
                config=config,
                torch_dtype=torch.bfloat16,
                # ignore_mismatched_sizes=True # 필요 없을 수 있음
            )
            logger.info("Adapted base model to Sequence Classification model using fallback.")

            # 모델을 GPU로 이동
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logger.info(f"Fallback model moved to device: {model.device}")

        else:
            # 다른 종류의 ValueError이면 다시 발생시킴
            logger.error(f"Failed to load model {model_config.name} due to unexpected ValueError: {e}")
            raise e
    except Exception as e:
        # 로딩 중 발생한 다른 모든 예외 처리
        logger.error(f"Failed to load model {model_config.name}: {e}")
        raise

    # --- 모델 로딩 후 공통 설정 ---
    # use_cache 설정은 여전히 유용할 수 있음
    model.config.use_cache = False

    # 모델 config에 pad_token_id 재확인
    if hasattr(model, 'config') and model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id
         logger.info("Set pad_token_id in the final model config.")

    logger.info("Model and tokenizer prepared for full fine-tuning.")
    return model, tokenizer

# --- Training function (데이터 처리 부분은 이전 수정과 동일, Trainer/저장 방식 확인) ---
def train_model(model_config):
    """Train the model for NLI using sequence classification approach (Full Fine-tuning)."""
    logger.info(f"Starting training for {model_config.name} (Full Fine-tuning)")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load model and tokenizer (이제 Full Fine-tuning용)
    model, tokenizer = load_model_and_tokenizer(model_config)

    # --- 데이터 로딩 (datasets 라이브러리 사용) ---
    logger.info(f"Loading dataset using datasets library from: {JSON_TRAIN_DATASET_PATH}")
    try:
        raw_dataset = load_dataset("json", data_files={"train": JSON_TRAIN_DATASET_PATH}, cache_dir=DATA_CACHE_DIR, split="train")
        logger.info(f"Loaded raw dataset with {len(raw_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {JSON_TRAIN_DATASET_PATH}: {e}")
        raise

    # --- 전처리 함수 정의 (동일) ---
    def preprocess_function(examples, tokenizer, max_length=MAX_LENGTH):
        tokenized_inputs = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False
        )
        return tokenized_inputs
    # --- 전처리 함수 정의 끝 ---

    # --- 데이터셋 전처리 적용 (동일) ---
    logger.info("Applying preprocessing to the dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["premise", "hypothesis", "guid", "source"]
    )
    logger.info("Preprocessing finished.")

    # --- 데이터 분할 (동일) ---
    logger.info("Splitting the processed dataset into train and validation sets...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    logger.info(f"Data split - train: {len(train_data)} examples, validation: {len(val_data)} examples")

    # --- Data Collator (동일) ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # --- Training arguments (메모리 고려하여 조정 필요할 수 있음) ---
    # Full Fine-tuning은 LoRA보다 훨씬 많은 메모리를 사용하므로 배치 크기/축적 단계를 더 줄여야 할 수 있습니다.
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=5e-5, # Full fine-tuning 시 조금 낮추는 것 고려 (예: 2e-5, 1e-5)
        per_device_train_batch_size=2,  # <-- 매우 작게 시작 (OOM 발생 시 1로 줄임)
        per_device_eval_batch_size=4,   # <-- 평가 배치 크기도 줄임
        gradient_accumulation_steps=16, # <-- 배치 크기를 줄인 만큼 늘림 (유효 배치 32)
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2, # 저장 공간 고려
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False,
        bf16=True, # 사용 가능하면 유지
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # F1으로 변경 권장
        greater_is_better=True,         # F1은 높을수록 좋음
        report_to="none",
        gradient_checkpointing=True, # Full fine-tuning 시 메모리 절약 위해 필수적
        optim="adamw_torch",
    )

    # Compute metrics function (동일)
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # --- 표준 Trainer 사용 ---
    trainer = Trainer(
        model=model, # Full 모델 전달
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)], # 조기 종료 콜백 유용
    )

    # Train
    logger.info("Starting full fine-tuning")
    trainer.train()

    # --- Save final model (Best model based on eval metric) ---
    # load_best_model_at_end=True 이므로, trainer.save_model()은 최적 모델 저장
    final_model_path = os.path.join(model_config.output_dir, "final") # LoRA 때와 구분 X
    logger.info(f"Saving final best model to: {final_model_path}")
    trainer.save_model(final_model_path) # Trainer 사용 시 이 방식 권장
    tokenizer.save_pretrained(final_model_path)
    logger.info("Final best model and tokenizer saved.")


    logger.info(f"Training completed for {model_config.name}")

    
# Main execution function
if __name__ == "__main__":
    logger.info("--- Starting KLUE-NLI Model Training Script ---")

    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        model = None; tokenizer = None # 초기화
        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            os.makedirs(DATA_CACHE_DIR, exist_ok=True) # 캐시 디렉토리 생성 확인
            train_model(model_config) # 학습 함수만 호출
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
    logger.info("--- KLUE-NLI Model Training Script Finished ---")