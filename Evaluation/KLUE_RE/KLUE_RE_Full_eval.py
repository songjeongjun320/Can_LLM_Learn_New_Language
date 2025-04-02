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
from sklearn.model_selection import train_test_split 
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
DATA_CACHE_DIR = "./klue_re_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_validation.json"
MAX_LENGTH = 512
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
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True
    )
    
    return model, tokenizer

# Custom Dataset for KLUE-RE
class REDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading RE dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples for RE")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]
        label = item["label"]
        
        # Add entity markers to the sentence
        # Ensure correct ordering of entities based on their positions
        sub_start, sub_end = subject_entity["start_idx"], subject_entity["end_idx"]
        obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]
        
        # Split the sentence into parts and insert markers
        if sub_start < obj_start:
            marked_sentence = (
                sentence[:sub_start] +
                "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                sentence[sub_end + 1:obj_start] +
                "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                sentence[obj_end + 1:]
            )
        else:
            marked_sentence = (
                sentence[:obj_start] +
                "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                sentence[obj_end + 1:sub_start] +
                "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                sentence[sub_end + 1:]
            )
        
        # Tokenize the marked sentence
        encoding = self.tokenizer(
            marked_sentence,
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
    """Train the model for RE using sequence classification approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    full_train_data = REDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.2,  # Val 20%
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
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # 배치 크기 증가
        per_device_eval_batch_size=8,  # 배치 크기 증가
        gradient_accumulation_steps=2,  # 축적 단계 감소
        num_train_epochs=2,
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
        data_collator=data_collator,
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
    """Evaluate the model on KLUE-RE metrics."""
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
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]
        gold_label = item["label"]
        
        # Add entity markers to the sentence
        sub_start, sub_end = subject_entity["start_idx"], subject_entity["end_idx"]
        obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]
        
        if sub_start < obj_start:
            marked_sentence = (
                sentence[:sub_start] +
                "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                sentence[sub_end + 1:obj_start] +
                "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                sentence[obj_end + 1:]
            )
        else:
            marked_sentence = (
                sentence[:obj_start] +
                "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                sentence[obj_end + 1:sub_start] +
                "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                sentence[sub_end + 1:]
            )
        
        # Tokenize
        encoding = tokenizer(
            marked_sentence,
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
            "sentence": sentence,
            "subject_entity": subject_entity["word"],
            "object_entity": object_entity["word"],
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
    logger.info("Starting KLUE-RE training and evaluation")
    
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
    combined_results_path = "klue_re_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-RE training and evaluation completed")