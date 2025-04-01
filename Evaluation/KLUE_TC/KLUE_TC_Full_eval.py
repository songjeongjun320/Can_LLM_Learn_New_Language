import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_f1_support, accuracy_score
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
        logging.FileHandler("klue_tc_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE TC Label Definitions
TC_LABELS = [
    "IT/과학",  # 0
    "경제",     # 1
    "사회",     # 2
    "생활/문화", # 3
    "세계",     # 4
    "스포츠",    # 5
    "정치"      # 6
]
NUM_LABELS = len(TC_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(TC_LABELS)}
ID2LABEL = {idx: label for label, idx in enumerate(TC_LABELS)}
logger.info(f"Total number of KLUE-TC labels: {NUM_LABELS}")

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir

# Model configurations (주어진 MODEL_CONFIGS 사용)
MODEL_CONFIGS = [
    ModelConfig(
        name="full-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_tc_results/full-olmo1B-org-klue-tc"
    ),
    ModelConfig(
        name="full-OLMo-1b-v12", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo1B-v12", 
        output_dir="klue_tc_results/full-olmo1B-v12-klue-tc"
    ),
    ModelConfig(
        name="full-OLMo-7b-org", 
        model_path="allenai/OLMo-7B", 
        output_dir="klue_tc_results/full-olmo7B-org-klue-tc"
    ),
    ModelConfig(
        name="full-OLMo-7b-v13", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo7B-v13", 
        output_dir="klue_tc_results/full-olmo7B-v13-klue-tc"
    ),
    ModelConfig(
        name="full-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_tc_results/full-llama3.2-3b-klue-tc"
    )
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_tc_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_tc_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_tc_validation.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

# Model and tokenizer loading function
def load_model_and_tokenizer(model_config):
    """Load model and tokenizer based on model configuration."""
    logger.info(f"Loading model: {model_config.model_path}")

    is_local = os.path.exists(model_config.model_path)
    logger.info(f"Model is local: {is_local}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, 
        trust_remote_code=True,
        local_files_only=is_local
    )
    
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model with bfloat16 precision for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Model loaded successfully: {model_config.name}")
    return model, tokenizer

# Custom Dataset for KLUE-TC
class TCDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading TC dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples for TC")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        title = item["title"]
        label = item["label"]
        
        # Tokenize the title
        encoding = self.tokenizer(
            title,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_model(model_config):
    """Train the model for TC using sequence classification approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    train_dataset = TCDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    val_dataset = TCDataset(JSON_VAL_DATASET_PATH, tokenizer)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
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
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # Compute metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_f1_support(labels, predictions, average="macro", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Training completed for {model_config.name}")
    return model, tokenizer

# Evaluation function
def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-TC metrics."""
    logger.info(f"Evaluating model: {model_config.name}")
    
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
    precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="macro", zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_f1_support(
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