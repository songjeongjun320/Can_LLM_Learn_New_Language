import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, # Regression으로 사용
    AutoTokenizer,
    AutoModelForCausalLM,               # OLMo 로딩 시 필요
    AutoConfig,                         # OLMo 로딩 시 필요
    TrainingArguments,
    Trainer,                            # 표준 Trainer 사용
    # EarlyStoppingCallback,           # 필요 시 사용
    DataCollatorWithPadding             # Regression/Classification에 사용
)

# from sklearn.metrics import mean_absolute_error, mean_squared_error # compute_metrics 구현 시 필요
import logging
from functools import partial

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration class (동일)
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_sts_results/olmo1B-org-klue-sts",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="OLMo-1b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_sts_results/olmo1B-v12-klue-sts",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_sts_results/olmo7B-org-klue-sts",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="OLMo-7b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_sts_results/olmo7B-v13-klue-sts",
    #     is_local=True
    # ),

    # llama models
    # ModelConfig(
    #     name="Llama-3.2-3B", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_sts_results/llama3.2-3b-klue-sts",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
    #     is_local=True,
    #     output_dir="klue_sts_results/llama3.2-3b-it-klue-sts",
    # ),
    # ModelConfig(
    #     name="Llama-3.1-8b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
    #     is_local=True,
    #     output_dir="klue_sts_results/llama3.1-8b-it-klue-sts",
    # ),

    # BERT Models
    # ModelConfig(
    #     name="BERT-base-uncased-origin",
    #     model_path="google-bert/bert-base-uncased",
    #     is_local=False,
    #     output_dir="klue_sts_results/BERT-base-uncased-origin-klue-sts",
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_sts_results/BERT-base-uncased-Tuned-klue-sts",
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Subtitle-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_sts_results/BERT-base-uncased-Subtitle-Tuned-klue-sts",
    # ),
    ModelConfig(
        name="bert-uncased-finetuned-subtitle_dt_v1",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt_v1",
        is_local=True, # Assuming this is local based on path pattern
        output_dir="klue_sts_results/bert-uncased-finetuned-subtitle_dt_v1-klue-sts",
    ),
    # ModelConfig(
    #     name="bert-uncased-finetuned-subtitle_dt_v2",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt_v2",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_sts_results/bert-uncased-finetuned-subtitle_dt_v2-klue-sts",
    # ),
    # ModelConfig(
    #     name="BERT-uncased-kr-eng-translation",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/bert-uncased-finetuned-kr-eng",
    #     output_dir="klue_sts_results/BERT-uncased-kr-eng-translation-klue-sts",
    #     is_local=True
    # ),
]


# 기본 설정
DATA_CACHE_DIR = "./klue_sts_regression_cache" # 캐시 경로 변경 권장
# --- !!! 중요: 숫자형 데이터셋 경로로 변경 !!! ---
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_train.json"
MAX_LENGTH = 512 # 필요시 조정

# Model and tokenizer loading function (Regression용으로 수정)
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 Sequence Classification 모델(num_labels=1)과 토크나이저를 로드합니다."""
    num_labels = 1 # Regression task
    logger.info(f"Load model: {model_config.model_path} for Sequence Regression (STS Task)")
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

    # 2. 모델 설정(Config) 로드 및 수정
    config = AutoConfig.from_pretrained(
        model_config.model_path,
        num_labels=num_labels, # 회귀이므로 1
        problem_type="regression", # <-- 명시적으로 회귀 문제임을 지정
        local_files_only=is_local,
        trust_remote_code=True
    )
    config.pad_token_id = tokenizer.pad_token_id
    logger.info("Model config loaded and modified for regression.")

    # 3. 모델 로드 (Try-Except 분기 유지)
    model = None
    try:
        # --- 시도 1: AutoModelForSequenceClassification 직접 로드 ---
        logger.info(f"Attempting direct load with AutoModelForSequenceClassification for {model_config.name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=is_local,
            trust_remote_code=True,
            ignore_mismatched_sizes=True # 헤드 크기 다를 수 있으므로 유지
        )
        logger.info(f"Successfully loaded {model.__class__.__name__} directly.")

    except ValueError as e:
        # --- 시도 2: 직접 로드 실패 시 (Unrecognized Config - OLMo 등) ---
        if "Unrecognized configuration class" in str(e) or "AutoModelForSequenceClassification" in str(e): # 오류 메시지 확인
            logger.warning(f"Direct load failed for {model_config.name}. Attempting fallback: Load CausalLM -> Adapt.")
            # 3a. 기본 Causal LM 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_path,
                config=config, # Regression 정보 포함된 config 사용
                torch_dtype=torch.bfloat16,
                local_files_only=is_local,
                trust_remote_code=True
            )
            logger.info("Base Causal LM model loaded for fallback.")
            # 3b. Sequence Classification 모델로 변환 (헤드 추가)
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                config=config,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True # 기본 LM 헤드 무시
            )
            logger.info("Adapted base model to Sequence Classification model using fallback.")
            # 모델을 GPU로 이동 (device_map 대신)
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logger.info(f"Fallback model moved to device: {model.device}")
        else:
            logger.error(f"Failed to load model {model_config.name} due to unexpected ValueError: {e}")
            raise e
    except Exception as e:
        logger.error(f"Failed to load model {model_config.name}: {e}")
        raise

    # --- 모델 로딩 후 공통 설정 ---
    model.config.use_cache = False # Gradient checkpointing 사용 시 권장
    if hasattr(model, 'config') and model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id
         logger.info("Set pad_token_id in the final model config.")

    logger.info("Model and tokenizer prepared for regression fine-tuning.")
    return model, tokenizer

# 메인 학습 함수 수정 (Regression용)
def train_model(model_config):
    """모델을 Regression 방식으로 학습시킵니다."""
    logger.info(f"Starting training for {model_config.name} (Regression)")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load model and tokenizer (Regression용으로 num_labels=1)
    model, tokenizer = load_model_and_tokenizer(model_config)

    # --- 데이터 로딩 (numeric 파일 사용) ---
    logger.info(f"Loading dataset to be split from: {JSON_TRAIN_DATASET_PATH}")
    try:
        full_train_val_dataset = load_dataset("json", data_files={"train": JSON_TRAIN_DATASET_PATH}, cache_dir=DATA_CACHE_DIR, split="train")
        logger.info(f"Loaded dataset with {len(full_train_val_dataset)} samples.")
        # 'output' 컬럼 이름을 'labels'로 변경하고 float 타입으로 변환
        full_train_val_dataset = full_train_val_dataset.map(lambda x: {'labels': float(x['output'])}, remove_columns=['output'])
        logger.info("Renamed 'output' column to 'labels' and converted to float.")
    except Exception as e:
        logger.error(f"Failed to load or process dataset from {JSON_TRAIN_DATASET_PATH}: {e}")
        raise

    # --- 전처리 함수 정의 (Input만 토크나이징) ---
    def preprocess_function_for_regression(examples, tokenizer, max_length=MAX_LENGTH):
        """Regression 학습용 데이터 전처리: input 필드만 토크나이징합니다."""
        # 'input' 필드 텍스트를 토크나이징
        model_inputs = tokenizer(
            examples["input"], # 'input' 컬럼 사용
            max_length=max_length,
            padding=False, # DataCollator가 처리
            truncation=True
        )
        # 'labels' 컬럼은 이미 숫자형이므로 그대로 유지됨
        return model_inputs
    # --- 전처리 함수 정의 끝 ---

    # --- 데이터셋 전처리 적용 ---
    logger.info("Preprocessing dataset for regression...")
    # 'input' 컬럼만 전처리 함수에 사용, 'labels' 컬럼은 유지
    tokenized_dataset = full_train_val_dataset.map(
        preprocess_function_for_regression,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': MAX_LENGTH},
        remove_columns=["input"] # 원본 'input' 텍스트 컬럼 제거
    )
    logger.info(f"Tokenized dataset prepared. Columns: {tokenized_dataset.column_names}")

    # --- 데이터 분할 ---
    logger.info("Splitting the processed dataset into train and validation sets...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(f"Split dataset: train={len(train_dataset)}, validation_during_training={len(eval_dataset)}")

    # --- 데이터 콜레이터 (Padding) ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # --- TrainingArguments (Regression용) ---
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=500, # 평가 빈도 조절
        learning_rate=1e-5, # 회귀는 조금 더 낮은 학습률 시도
        per_device_train_batch_size=8, # 메모리 상황 따라 조절
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, # 유효 배치 32
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=500, # 저장 빈도 조절
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # 또는 compute_metrics 구현 후 'eval_mae' 등
        greater_is_better=False, # loss는 낮을수록 좋음
        report_to="none",
        gradient_checkpointing=True, # 메모리 절약 위해 유지
        optim="adamw_torch",
    )

    # --- Trainer (표준 Trainer 사용) ---
    logger.info("Initializing Trainer for regression...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, # DataCollator 위해 전달
        data_collator=data_collator,
        # compute_metrics=compute_regression_metrics, # 필요 시 회귀 메트릭 함수 구현/전달
    )

    # 학습 실행
    logger.info("Starting model training (regression)...")
    trainer.train()
    logger.info("Training finished.")

    # 최종 모델 저장
    final_model_path = os.path.join(model_config.output_dir, "final_model")
    logger.info(f"Saving final best model (based on validation loss) to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("Training and saving best model completed!")


# 메인 실행 함수
if __name__ == "__main__":
    logger.info("--- Starting KLUE-STS Regression Model Training Script ---") # 로그 수정
    # 각 모델별로 학습 실행
    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        # 출력 디렉토리 생성 (경로 변경 권장됨)
        os.makedirs(model_config.output_dir, exist_ok=True)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        model = None
        tokenizer = None
        trainer = None

        try:
            # 모델 학습 실행
            train_model(model_config)
            logger.info(f"--- Completed training for {model_config.name} ---")
        except Exception as e:
            logger.error(f"Error processing model {model_config.name}: {e}")
            logger.exception("Exception details:")
        finally:
            # 메모리 정리
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del model, tokenizer, trainer # trainer도 명시적 삭제
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache.")
            logger.info(f"Resource cleanup finished for {model_config.name}.")
            print("-" * 50)
    logger.info("--- KLUE-STS Regression Training Script Finished ---") # 로그 수정