import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding

)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_re_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE RE Label Definitions (테이블에 맞게 수정)
RE_LABELS = [
    "no_relation",
    "org:dissolved",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]
NUM_LABELS = len(RE_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(RE_LABELS)}  # 레이블 -> 인덱스
ID2LABEL = {idx: label for idx, label in enumerate(RE_LABELS)}  # 인덱스 -> 레이블
logger.info(f"Total number of KLUE-RE labels: {NUM_LABELS}")


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
    #     output_dir="klue_re_results/full-olmo1B-org-klue-re",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-1b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_re_results/full-olmo1B-v12-klue-re",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_re_results/full-olmo7B-org-klue-re",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_re_results/full-olmo7B-v13-klue-re",
    #     is_local=True
    # ),
    #     ModelConfig(
    #     name="full-Llama-3.2-3b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_re_results/full-llama3.2-3b-klue-re",
    #     is_local=True
    # ),
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        output_dir="klue_re_results/lora-llama3.2-3b-it-klue-re",
        is_local=True,
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        output_dir="klue_re_results/lora-llama3.1-8b-it-klue-re",
        is_local=True
    ),
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_re_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_train.json"
MAX_LENGTH = 1024
MAX_EVAL_SAMPLES = 200

# Model and tokenizer loading function
def load_model_and_tokenizer(model_config, num_labels): # num_labels 인자 추가
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model: {model_config.model_path} for Sequence Classification")
    is_local = model_config.is_local

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path,
        local_files_only=is_local,
        trust_remote_code=True
    )

    # --- 패딩 토큰 설정 ---
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            logger.warning(f"Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id})")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
             logger.warning(f"Setting pad_token_id to unk_token_id ({tokenizer.unk_token_id})")
             tokenizer.pad_token_id = tokenizer.unk_token_id
             if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
        else: raise ValueError("Cannot set padding token.")
    elif tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])
    logger.info(f"Using pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    # --- 패딩 토큰 설정 끝 ---

    # --- 특수 마커 추가 (만약 토크나이저에 없다면) ---
    special_tokens_dict = {'additional_special_tokens': ['[SUBJ]', '[/SUBJ]', '[OBJ]', '[/OBJ]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_toks > 0:
        logger.info(f"Added {num_added_toks} special marker tokens to the tokenizer.")
    # --- 특수 마커 추가 끝 ---


    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_path,
        num_labels=num_labels, # 레이블 개수 설정
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True,
        ignore_mismatched_sizes=True # LM 헤드 무시
    )

    # 모델 config에도 pad_token_id 설정 (중요)
    model.config.pad_token_id = tokenizer.pad_token_id
    # 모델 임베딩 크기 조정 (특수 토큰 추가 시 필요)
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model embeddings to: {len(tokenizer)}")


    return model, tokenizer

# Training function
def train_model(model_config):
    # --- 전처리 함수 정의 (이전과 동일) ---
    def preprocess_re_data(examples, tokenizer, max_length=MAX_LENGTH):
        processed_examples = {'input_ids': [], 'attention_mask': [], 'labels': []}
        skipped_count = 0
        for i in range(len(examples['sentence'])):
            sentence = examples['sentence'][i]
            subject_entity = examples['subject_entity'][i]
            object_entity = examples['object_entity'][i]
            label_id = examples['label'][i] # 숫자 레이블 ID

            if not all([sentence, subject_entity, object_entity, label_id is not None]):
                skipped_count += 1; continue
            if not isinstance(label_id, int) or not (0 <= label_id < NUM_LABELS):
                skipped_count += 1; continue

            sub_start = subject_entity.get('start_idx'); sub_end = subject_entity.get('end_idx')
            obj_start = object_entity.get('start_idx'); obj_end = object_entity.get('end_idx')
            if None in [sub_start, sub_end, obj_start, obj_end]:
                skipped_count += 1; continue

            try:
                 if sub_start < obj_start:
                     marked_sentence = (sentence[:sub_start] + "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                                        sentence[sub_end + 1:obj_start] + "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                                        sentence[obj_end + 1:])
                 else:
                     marked_sentence = (sentence[:obj_start] + "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                                        sentence[obj_end + 1:sub_start] + "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                                        sentence[sub_end + 1:])
            except Exception as e:
                 skipped_count += 1; continue

            encoding = tokenizer(marked_sentence, truncation=True, max_length=max_length, padding="max_length")
            processed_examples['input_ids'].append(encoding['input_ids'])
            processed_examples['attention_mask'].append(encoding['attention_mask'])
            processed_examples['labels'].append(label_id)

        if skipped_count > 0: logger.warning(f"Skipped {skipped_count} examples in batch.")
        return processed_examples
    # --- 전처리 함수 정의 끝 ---

    """Train the model for RE using sequence classification approach."""
    logger.info(f"Starting training for {model_config.name} (RE Task)")
    logger.info(f"Loading data from {JSON_TRAIN_DATASET_PATH} and splitting into 90% train / 10% validation.") # 로그 메시지 수정

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_config, num_labels=NUM_LABELS)

    # --- Load **ONLY** train dataset and split it ---
    logger.info(f"Loading dataset to split from: {JSON_TRAIN_DATASET_PATH}")
    try:
        # 학습 데이터 파일만 로드
        full_dataset = load_dataset("json", data_files={"train": JSON_TRAIN_DATASET_PATH}, split="train", cache_dir=DATA_CACHE_DIR)
        # JSON_VAL_DATASET_PATH 로드 부분 삭제
        # raw_val_dataset = load_dataset("json", data_files=JSON_VAL_DATASET_PATH, split="train", cache_dir=DATA_CACHE_DIR)

        logger.info(f"Loaded dataset with {len(full_dataset)} samples to be split.")

        # 학습 데이터셋을 90/10으로 분할
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
        raw_train_dataset = split_dataset["train"]
        raw_val_dataset_for_training = split_dataset["test"] # 분할된 10%를 검증용으로 사용

        logger.info(f"Split dataset: train={len(raw_train_dataset)}, validation_during_training={len(raw_val_dataset_for_training)}")

    except Exception as e:
        logger.error(f"Failed to load or split dataset from {JSON_TRAIN_DATASET_PATH}: {e}")
        # 대체 로딩 로직도 동일하게 수정
        try:
            logger.info("Attempting alternative loading via json.load...")
            with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f: train_list = json.load(f)
            # 검증 파일 로드 제거
            # with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f: val_list = json.load(f)
            from datasets import Dataset
            full_dataset = Dataset.from_list(train_list)
            # 분할
            split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
            raw_train_dataset = split_dataset["train"]
            raw_val_dataset_for_training = split_dataset["test"] # 변수명 변경
            logger.info(f"Loaded via json.load and split: train={len(raw_train_dataset)}, validation={len(raw_val_dataset_for_training)}")
        except Exception as e2:
            logger.error(f"Alternative loading failed: {e2}")
            raise e

    # --- Preprocess datasets using .map ---
    logger.info("Preprocessing datasets...")
    train_dataset = raw_train_dataset.map(
        lambda examples: preprocess_re_data(examples, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True, remove_columns=raw_train_dataset.column_names)
    # 변수명 변경된 것을 사용
    val_dataset_for_training = raw_val_dataset_for_training.map(
        lambda examples: preprocess_re_data(examples, tokenizer=tokenizer, max_length=MAX_LENGTH),
        batched=True, remove_columns=raw_val_dataset_for_training.column_names)
    logger.info(f"Preprocessing complete: train={len(train_dataset)}, validation={len(val_dataset_for_training)}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps", 
        eval_steps=200,
        learning_rate=2e-5, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16, 
        gradient_accumulation_steps=2,
        num_train_epochs=3, 
        weight_decay=0.01, 
        save_total_limit=2,
        save_strategy="steps", 
        save_steps=200,
        logging_dir=os.path.join(model_config.output_dir, "logs"), 
        logging_steps=100,
        fp16=False, 
        bf16=True, 
        lr_scheduler_type="linear", 
        warmup_steps=500,
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        greater_is_better=True,
        report_to="none", 
        gradient_checkpointing=False, 
        optim="adamw_torch",
    )

    # --- Compute metrics function ---
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # --- Trainer ---
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset_for_training, # <-- 분할된 검증 데이터 사용
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    # --- Train ---
    logger.info("Starting training for Sequence Classification")
    trainer.train()

    # --- Save model ---
    final_model_path = os.path.join(model_config.output_dir, "final_model")
    logger.info(f"Saving final best model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info(f"Training completed for {model_config.name}")

# Main execution
if __name__ == "__main__":
    # ... (main 루프는 이전과 동일) ...
    logger.info("--- Starting KLUE-RE Model Training Script ---")
    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        model = None; tokenizer = None
        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            os.makedirs(DATA_CACHE_DIR, exist_ok=True)
            train_model(model_config) # 학습 함수만 호출
            logger.info(f"--- Completed training for {model_config.name} ---")
        except Exception as e:
            logger.error(f"Error training model {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
        finally:
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del model; del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info(f"Resource cleanup finished for {model_config.name}.")
            print("-" * 50)
    logger.info("--- KLUE-RE Model Training Script Finished ---")