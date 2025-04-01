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
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from datasets import load_dataset

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
        self.is_local = is_

# 모델 설정들 (기본 OLMo 1B, OLMo 7B)
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


# Configuration parameters
DATA_CACHE_DIR = "./klue_mrc_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_validation.json"
MAX_LENGTH = 1024  # Increased length for MRC which often has longer contexts
MAX_EVAL_SAMPLES = 200

# Function to check if dataset exists and is valid
def check_dataset():
    """Check if the KLUE MRC dataset exists and is valid."""
    if not os.path.exists(JSON_TRAIN_DATASET_PATH) or not os.path.exists(JSON_VAL_DATASET_PATH):
        logger.error(f"Dataset files not found at {JSON_TRAIN_DATASET_PATH} or {JSON_VAL_DATASET_PATH}")
        return False
    
    try:
        with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        
        logger.info(f"Dataset files loaded successfully. Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Verify structure of a sample
        sample = train_data[0]
        required_fields = ["title", "context", "question", "answers"]
        if not all(field in sample for field in required_fields):
            logger.error(f"Dataset format is not as expected, missing required fields")
            return False
            
        logger.info("Dataset format is valid")
        return True
    except Exception as e:
        logger.error(f"Error checking dataset: {str(e)}")
        return False

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

# Custom Dataset for KLUE-MRC
class MachineReadingComprehensionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading MRC dataset from {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples for MRC")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract data
        title = item.get("title", "")
        context = item.get("context", "")
        question = item.get("question", "")
        answers = item.get("answers", {"text": [""]})
        answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
        
        # Create prompt and completion
        prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        completion = f" {answer_text}"
        
        # Combine prompt and completion for training
        full_text = prompt + completion
        
        # Tokenize the full text
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize only the prompt to create labels
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels, masking the prompt portion with -100
        labels = encoded["input_ids"].clone().squeeze(0)
        prompt_length = prompt_encoded["input_ids"].shape[1]
        labels[:prompt_length] = -100
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels
        }

# Custom Trainer for language modeling approach
from transformers import DataCollatorForLanguageModeling

def train_model(model_config):
    """Train the model for machine reading comprehension using language modeling approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    # Create output directory
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Check dataset
    if not check_dataset():
        logger.error("Dataset check failed. Aborting training.")
        return None, None
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    logger.info("Loading train and validation datasets")
    train_dataset = MachineReadingComprehensionDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    val_dataset = MachineReadingComprehensionDataset(JSON_VAL_DATASET_PATH, tokenizer)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    # Data collator for language modeling
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
        num_train_epochs=5,
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
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3
    )
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save final model and tokenizer
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Training completed for {model_config.name}")
    return model, tokenizer

# MRC evaluation metrics
def compute_exact_match(prediction, ground_truth):
    """
    Calculate exact match score between prediction and ground truth.
    """
    prediction = prediction.strip().lower()
    ground_truths = [gt.strip().lower() for gt in ground_truth]
    return max(int(prediction == gt) for gt in ground_truths)

def compute_f1_score(prediction, ground_truth):
    """
    Calculate token-level F1 score between prediction and ground truth.
    """
    def get_tokens(s):
        return s.strip().lower().split()
    
    prediction_tokens = get_tokens(prediction)
    f1_scores = []
    
    for gt in ground_truth:
        gt_tokens = get_tokens(gt)
        
        common_tokens = set(prediction_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        precision = len(common_tokens) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return max(f1_scores)

def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-MRC metrics."""
    logger.info(f"Evaluating model: {model_config.name}")
    
    # Load MRC dataset
    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Limit number of samples for faster evaluation
    val_subset = val_data[:MAX_EVAL_SAMPLES]
    
    model.eval()
    
    # Results tracking
    exact_match_scores = []
    f1_scores = []
    
    # Generate detailed logs
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    logs = []
    
    for item in tqdm(val_subset, desc="Evaluating"):
        title = item.get("title", "")
        context = item.get("context", "")
        question = item.get("question", "")
        
        # Skip items with missing fields
        if not context or not question:
            continue
        
        ground_truth_answers = item.get("answers", {}).get("text", [""])
        
        # Create prompt for model
        prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=64,  # Limiting output tokens for answers
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        answer = generated_text[len(prompt):].strip()
        
        # Calculate metrics
        em_score = compute_exact_match(answer, ground_truth_answers)
        f1 = compute_f1_score(answer, ground_truth_answers)
        
        exact_match_scores.append(em_score)
        f1_scores.append(f1)
        
        # Log details
        log_entry = {
            "title": title,
            "context": context[:200] + "..." if len(context) > 200 else context,  # Truncate for log
            "question": question,
            "ground_truth": ground_truth_answers,
            "prediction": answer,
            "em_score": em_score,
            "f1_score": f1
        }
        logs.append(log_entry)
    
    # Calculate final metrics
    avg_em = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Log results
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Exact Match: {avg_em:.4f}")
    logger.info(f"F1 Score: {avg_f1:.4f}")
    logger.info(f"Total examples evaluated: {len(exact_match_scores)}")
    
    # Save logs and results
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        
    results = {
        "model": model_config.name,
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "total_examples": len(exact_match_scores)
    }
    
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")
    
    return results

# Main execution function
if __name__ == "__main__":
    logger.info("Starting KLUE-MRC training and evaluation")
    
    # Process each model configuration
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")
        
        try:
            # Create output directories
            os.makedirs(model_config.output_dir, exist_ok=True)
            
            # Train model
            model, tokenizer = train_model(model_config)
            
            if model is not None:
                # Evaluate model
                results = evaluate_model(model, tokenizer, model_config)
                all_results[model_config.name] = results
            
            logger.info(f"Completed processing for {model_config.name}")
            
        except Exception as e:
            logger.error(f"Error processing {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
    
    # Save combined results
    combined_results_path = "klue_mrc_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-MRC training and evaluation completed")