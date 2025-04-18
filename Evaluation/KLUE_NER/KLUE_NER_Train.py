import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_ner_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE NER Label Definitions
NER_TAGS = [
    "B-LC",     # 0
    "I-LC",     # 1
    "B-DT",     # 2
    "I-DT",     # 3
    "B-OG",     # 4
    "I-OG",     # 5
    "B-PS",     # 6
    "I-PS",     # 7
    "B-QT",     # 8
    "I-QT",     # 9
    "B-TI",     # 10
    "I-TI",     # 11
    "O"         # 12
]
NUM_LABELS = len(NER_TAGS)
LABEL2ID = {label: idx for idx, label in enumerate(NER_TAGS)}
ID2LABEL = {idx: label for label, idx in enumerate(NER_TAGS)}
logger.info(f"Total number of KLUE-NER labels: {NUM_LABELS}")

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
    #     output_dir="klue_ner_results/full-olmo1B-org-klue-ner",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-1b-tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_ner_results/full-olmo1B-klue-ner",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_ner_results/full-olmo7B-org-klue-ner",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_ner_results/full-olmo7B-klue-ner",
    #     is_local=True
    # ),
    #     ModelConfig(
    #     name="full-Llama-3.2:3B", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_ner_results/full-llama3.2-3b-klue-ner",
    #     is_local=True
    # ),
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        output_dir="klue_ner_results/lora-llama3.2-3b-it-klue-ner",
        is_local=True,
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        output_dir="klue_ner_results/lora-llama3.1-8b-it-klue-ner",
        is_local=True
    ),
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_ner_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_ner_train.json"
MAX_LENGTH = 256
MAX_EVAL_SAMPLES = 200

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

    # --- 모델 로드 끝 ---
    peft_params = LoraConfig(
        lora_alpha=16, # Often set to 2*r
        lora_dropout=0.1,
        r=64,          # Rank, higher might capture more complex patterns but uses more memory
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_params)
    model.config.use_cache = False # Important for training stability with LoRA/gradient checkpointing
    model.print_trainable_parameters()

    return model, tokenizer

# Custom Dataset for KLUE-NER
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading NER dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples for NER")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        
        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Align labels with tokenized input
        word_ids = encoding.word_ids()
        labels = [-100] * len(encoding["input_ids"][0])
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels[i] = -100
            elif word_idx != previous_word_idx:
                labels[i] = ner_tags[word_idx]
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels)
        }

# Training function
def train_model(model_config):
    """Train the model for NER using token classification approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    full_train_data = NERDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.1,  # Val 20%
        random_state=42,  # 재현성 보장
        shuffle=True
    )
    logger.info(f"Loaded data - train: {len(train_data)} examples, validation: {len(val_data)} examples")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # 배치 크기 증가
        per_device_eval_batch_size=1,  # 배치 크기 증가
        gradient_accumulation_steps=16,  # 축적 단계 감소
        num_train_epochs=3,
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
    
    # Compute metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_labels = []
        pred_labels = []
        for pred, label in zip(predictions, labels):
            for p, l in zip(pred, label):
                if l != -100:
                    true_labels.append(l)
                    pred_labels.append(p)
        
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="micro", zero_division=0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save model (Merge LoRA weights into the base model and save the full model)
    final_model_path = os.path.join(model_config.output_dir, "final_merged") # 경로 이름 변경 권장
    logger.info(f"Merging LoRA adapter into base model...")

    # --- 모델 병합 및 언로드 ---
    try:
        # merge_and_unload()는 병합된 베이스 모델을 반환하므로, 다시 할당해야 함
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged and unloaded successfully.")

        # --- 병합된 전체 모델 저장 ---
        logger.info(f"Saving the full merged model to: {final_model_path}")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path) # 토크나이저는 동일하게 저장
        logger.info("Full merged model and tokenizer saved.")

    except Exception as e:
        logger.error(f"Error during model merging or saving: {e}")
        logger.warning("Attempting to save the adapter only as a fallback...")
        # Fallback: 어댑터만 저장 시도 (오류 발생 시)
        adapter_only_path = os.path.join(model_config.output_dir, "final_adapter_fallback")
        try:
            # merge_and_unload 전에 원본 PEFT 모델 상태가 필요하므로,
            # 실제로는 오류 발생 시 여기서 다시 로드하거나 해야 함.
            # 여기서는 일단 merge_and_unload 전에 저장하는 것으로 가정하거나,
            # 오류 처리를 더 견고하게 만들어야 함.
            # peft_model.save_pretrained(adapter_only_path) # peft_model 변수가 있다면
            logger.warning(f"Fallback save path (adapter only): {adapter_only_path}. Manual merge might be needed later.")
            # tokenizer.save_pretrained(adapter_only_path)
        except Exception as fallback_e:
            logger.error(f"Fallback adapter save also failed: {fallback_e}")


    logger.info(f"Training completed for {model_config.name}")

# Main execution
if __name__ == "__main__":
    logger.info("--- Starting KLUE-NER Model Training Script ---")
    for model_config in MODEL_CONFIGS:
        logger.info(f"--- Processing model: {model_config.name} ---")
        model = None; tokenizer = None # 초기화
        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            os.makedirs(DATA_CACHE_DIR, exist_ok=True)
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
    logger.info("--- KLUE-NER Model Training Script Finished ---")