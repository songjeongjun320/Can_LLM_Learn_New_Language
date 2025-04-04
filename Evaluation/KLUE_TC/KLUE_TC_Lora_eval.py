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
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_tc_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE TC Label Definitions
TC_LABELS = [
    "IT/SCIENCE",  # 0
    "ECONOMY",     # 1
    "SOCIETY",     # 2
    "CULTURE", # 3
    "WORLD",     # 4
    "SPORTS",    # 5
    "POLITICS"      # 6
]
NUM_LABELS = len(TC_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(TC_LABELS)}
ID2LABEL = {idx: label for label, idx in enumerate(TC_LABELS)}
logger.info(f"Total number of KLUE-TC labels: {NUM_LABELS}")

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations (주어진 MODEL_CONFIGS 사용)
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_tc_results/lora-olmo1B-org-klue-tc",
        is_local=False
    ),
    ModelConfig(
        name="lora-OLMo-1b-Tuned", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
        output_dir="klue_tc_results/lora-olmo1B-v12-klue-tc",
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_tc_results/lora-olmo7B-org-klue-tc",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_tc_results/lora-olmo7B-v13-klue-tc",
    #     is_local=True
    # ),
        ModelConfig(
        name="lora-Llama-3.2-3b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_tc_results/lora-llama3.2-3b-klue-tc",
        is_local=True
    )
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_tc_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_tc_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_tc_validation.json"
MAX_LENGTH = 64
MAX_EVAL_SAMPLES = 500

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

# Training function
def train_model(model_config):
    """Train the model for TC using sequence classification approach."""
    logger.info("============================================")
    logger.info(f"Starting training for {model_config.name}")
    logger.info("============================================")
    
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load and process the dataset directly
    logger.info(f"Loading TC dataset from {JSON_TRAIN_DATASET_PATH}")
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} samples for TC")
    
    # Tokenization function
    def tokenize_function(example):
        encoding = tokenizer(
            example["title"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }
    
    # Process all examples
    processed_dataset = [tokenize_function(item) for item in dataset]
    
    # Convert to Hugging Face Dataset
    processed_dataset = Dataset.from_list(processed_dataset)
    
    # Split into train and validation sets using datasets' train_test_split
    split_dataset = processed_dataset.train_test_split(
        test_size=0.2,  # 20% for validation
        seed=42,        # Ensure reproducibility
        shuffle=True
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    logger.info(f"Processed data - train: {len(train_dataset)} examples, validation: {len(val_dataset)} examples")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # For tensor core efficiency
    )
    
    # LoRA 설정 추가
    peft_params = LoraConfig(
        lora_alpha=16,  # LoRA 스케일링 팩터
        lora_dropout=0.1,  # LoRA 드롭아웃 비율
        r=16,  # LoRA 랭크
        bias="none",  
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"]
    )    
    if (model_config.name == "lora-Llama-3.2-3b"):
        peft_params = LoraConfig(
            lora_alpha=16,  # LoRA 스케일링 팩터
            lora_dropout=0.1,  # LoRA 드롭아웃 비율
            r=16,  # LoRA 랭크
            bias="none",  
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj"]
        )

    # 모델 및 토크나이저 로드 시 LoRA 설정 적용
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=300,
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # 배치 크기 증가
        per_device_eval_batch_size=4,  # 배치 크기 증가
        gradient_accumulation_steps=2,  # 축적 단계 감소
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=600,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=True,  # FP16으로 전환
        bf16=False,  # BF16 비활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Warmup 비율 감소
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,  # 체크포인팅 비활성화
        optim="adamw_torch",  # 필요 시 "adamw_8bit"로 변경
        eval_accumulation_steps=10
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # SFTTrainer 초기화 시 tokenizer와 packing 제거
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
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

# Evaluation function
def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-TC metrics."""
    logger.info("============================================")
    logger.info(f"Evaluating model: {model_config.name}")
    logger.info("============================================")

    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    val_subset = val_data[:MAX_EVAL_SAMPLES]
    
    model.eval()
    device = model.device
    
    true_labels = []
    pred_labels = []
    logs = []
    
    for item in tqdm(val_subset, desc="Evaluating"):
        title = item["title"]
        gold_label = item["label"]
        
        # Tokenize
        encoding = tokenizer(
            title,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        true_labels.append(gold_label)
        pred_labels.append(prediction)
        
        # Log details
        logs.append({
            "title": title,
            "gold_label": ID2LABEL[gold_label],
            "pred_label": ID2LABEL[prediction]
        })
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    per_class_metrics = {
        ID2LABEL[i]: {"precision": p, "recall": r, "f1": f, "support": s}
        for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class))
    }
    
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro Precision: {precision:.4f}")
    logger.info(f"Macro Recall: {recall:.4f}")
    logger.info(f"Macro F1: {f1:.4f}")
    logger.info("Per-class metrics:")
    logger.info(json.dumps(per_class_metrics, indent=2))
    
    # Save logs and results
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    results = {
        "model": model_config.name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(val_subset),
        "per_class_metrics": per_class_metrics
    }
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")
    
    return results

# Main execution
if __name__ == "__main__":
    logger.info("Starting KLUE-TC training and evaluation")
    
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")
        
        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            
            # Train
            model, tokenizer = train_model(model_config)
            
            # Evaluate
            results = evaluate_model(model, tokenizer, model_config)
            all_results[model_config.name] = results
            
            logger.info(f"Completed processing for {model_config.name}")
            
        except Exception as e:
            logger.error(f"Error processing {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
    
    # Save combined results
    combined_results_path = "klue_tc_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-TC training and evaluation completed")