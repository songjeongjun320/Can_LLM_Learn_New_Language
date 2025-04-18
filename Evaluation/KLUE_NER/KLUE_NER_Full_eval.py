import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_f1_support
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
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir

# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        name="full-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_sts_results/full-olmo1B-org-klue-sts"
    ),
    ModelConfig(
        name="full-OLMo-1b-v12", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo1B-v12", 
        output_dir="klue_sts_results/full-olmo1B-v12-klue-sts"
    ),
    ModelConfig(
        name="full-OLMo-7b-org", 
        model_path="allenai/OLMo-7B", 
        output_dir="klue_sts_results/full-olmo7B-org-klue-sts"
    ),
    ModelConfig(
        name="full-OLMo-7b-v13", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo7B-v13", 
        output_dir="klue_sts_results/full-olmo7B-v13-klue-sts"
    ),
        ModelConfig(
        name="full-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_sts_results/full-llama3.2-3b-klue-sts"
    )
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_ner_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_ner_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_ner_validation.json"
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
    
    logger.info(f"Loading model with bfloat16 precision for token classification...")
    model = AutoModelForTokenClassification.from_pretrained(
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
    train_dataset = NERDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    val_dataset = NERDataset(JSON_VAL_DATASET_PATH, tokenizer)
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
        metric_for_best_model="eval_f1",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
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
        
        precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="micro", zero_division=0)
        return {
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
    """Evaluate the model on KLUE-NER metrics."""
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
        tokens = item["tokens"]
        gold_labels = item["ner_tags"]
        
        # Tokenize
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        word_ids = encoding.word_ids()
        
        # Align predictions with original tokens
        aligned_preds = []
        aligned_golds = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx and word_idx < len(gold_labels):
                aligned_preds.append(predictions[i])
                aligned_golds.append(gold_labels[word_idx])
            previous_word_idx = word_idx
        
        true_labels.extend(aligned_golds)
        pred_labels.extend(aligned_preds)
        
        # Log details
        logs.append({
            "sentence": " ".join(tokens),
            "gold_labels": [ID2LABEL.get(l, "O") for l in aligned_golds],
            "pred_labels": [ID2LABEL.get(p, "O") for p in aligned_preds]
        })
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_f1_support(true_labels, pred_labels, average="micro", zero_division=0)
    
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1: {f1:.4f}")
    
    # Save logs and results
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    results = {
        "model": model_config.name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(val_subset)
    }
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")
    
    return results

# Main execution
if __name__ == "__main__":
    logger.info("Starting KLUE-NER training and evaluation")
    
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
    combined_results_path = "klue_ner_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-NER training and evaluation completed")