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
from torch.utils.data import Dataset
import torch.nn.functional as F
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_dp_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE-DP Label Definitions
SYNTAX_TAGS = ['NP', 'VP', 'AP', 'VNP', 'DP', 'IP', 'X', 'L', 'R']
FUNCTION_TAGS = ['SBJ', 'OBJ', 'MOD', 'AJT', 'CMP', 'CNJ']

# Generate all possible label combinations
LABELS = []
for syntax in SYNTAX_TAGS:
    LABELS.append(syntax)
    for function in FUNCTION_TAGS:
        LABELS.append(f"{syntax}_{function}")
LABELS += ['ROOT']
LABELS = sorted(list(set(LABELS)))
NUM_LABELS = len(LABELS)
logger.info(f"Total number of KLUE-DP labels: {NUM_LABELS}")

# Model configuration class (from first code)
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations (from first code)
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_dp_results/olmo1B-org-klue-dp",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="OLMo-1b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_dp_results/olmo1B-v12-klue-dp",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_dp_results/olmo7B-org-klue-dp",
    #     is_local=False
    # ),
    ModelConfig(
        name="OLMo-7b-Tuned", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
        output_dir="klue_dp_results/olmo7B-v13-klue-dp",
        is_local=True
    ),
    ModelConfig(
        name="Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_dp_results/llama3.2-3b-klue-dp",
        is_local=True
    ),
    # ModelConfig(
    #     name="Llama-3.2-3b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
    #     output_dir="klue_dp_results/lora-llama3.2-3b-it-klue-dp",
    #     is_local=True,
    # ),
    # ModelConfig(
    #     name="Llama-3.1-8b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
    #     output_dir="klue_dp_results/lora-llama3.1-8b-it-klue-dp",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased",
    #     model_path="bert-base-uncased",
    #     is_local=False,
    #     output_dir="klue_dp_results/BERT-base-uncased-klue-dp",
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_dp_results/BERT-base-uncased-Tuned-klue-dp",
    # ),
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_dp_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dp_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dp_validation.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

# Function to prepare the dataset in JSON format if not already available
def prepare_dataset_json():
    """Convert existing KLUE-DP dataset to the required input/output format."""
    if os.path.exists(JSON_TRAIN_DATASET_PATH) and os.path.exists(JSON_VAL_DATASET_PATH):
        logger.info(f"Dataset already exists: {JSON_TRAIN_DATASET_PATH} and {JSON_VAL_DATASET_PATH}")
        return
    
    logger.info("Converting KLUE DP dataset to input/output format...")
    
    # Paths to your existing data files
    original_train_path = JSON_TRAIN_DATASET_PATH
    original_val_path = JSON_VAL_DATASET_PATH
    
    # Load original data
    with open(original_train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(original_val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Prepare lists for training and validation data
    train_samples = []
    val_samples = []
    
    # Function to create prompt and completion
    def create_prompt(item):
        tokens = item["word_form"]
        token_str = " ".join(tokens)
        return f"Perform dependency parsing on the following Korean sentence. Provide head indices (0-based) and dependency relations for each token. Sentence: {token_str}"

    def create_completion(item):
        tokens = item["word_form"]
        heads = item["head"]
        deprels = item["deprel"]
        results = []
        for i, (token, head, deprel) in enumerate(zip(tokens, heads, deprels)):
            results.append(f"{token}: head={head}, relation={deprel}")
        return " " + "\n".join(results)
    
    # Process training data
    logger.info("Creating training data JSON...")
    for item in tqdm(train_data):
        if not item.get("word_form"):  # Skip empty samples
            continue
            
        sample = {
            "input": create_prompt(item),
            "output": create_completion(item)
        }
        train_samples.append(sample)
    
    # Process validation data
    logger.info("Creating validation data JSON...")
    for item in tqdm(val_data):
        if not item.get("word_form"):  # Skip empty samples
            continue
            
        sample = {
            "input": create_prompt(item),
            "output": create_completion(item)
        }
        val_samples.append(sample)
    
    # Save JSON files
    os.makedirs(os.path.dirname(JSON_TRAIN_DATASET_PATH), exist_ok=True)
    
    logger.info(f"Saving JSON datasets... (train: {len(train_samples)}, valid: {len(val_samples)})")
    with open(JSON_TRAIN_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
    with open(JSON_VAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created KLUE-DP datasets: {JSON_TRAIN_DATASET_PATH} and {JSON_VAL_DATASET_PATH}")

# Model and tokenizer loading function (from first code)
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

# Custom Dataset for KLUE-DP
class DependencyParsingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading DP dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {label: i for i, label in enumerate(LABELS)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        logger.info(f"Loaded {len(self.data)} samples for dependency parsing")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract data from the current format
        tokens = item["word_form"]
        heads = item["head"]
        deprels = item["deprel"]
        
        # Create prompt and completion
        prompt = f"Perform dependency parsing on the following Korean sentence. Provide head indices (0-based) and dependency relations for each token. Sentence: {' '.join(tokens)}"
        
        completion = ""
        for i, (token, head, deprel) in enumerate(zip(tokens, heads, deprels)):
            completion += f"{token}: head={head}, relation={deprel}\n"
        
        # Combine prompt and completion for training
        full_text = prompt + " " + completion
        
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

def train_model(model_config):
    """Train the model for dependency parsing using language modeling approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    # Create output directory
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Prepare dataset if needed
    prepare_dataset_json()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 1. 원본 데이터 로드
    full_train_data = DependencyParsingDataset(JSON_TRAIN_DATASET_PATH, tokenizer)  # 전체 훈련 데이터

    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.1,  # Val 20%
        random_state=42,  # 재현성 보장
        shuffle=True
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # TrainingArguments (이전 설정 사용)
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps", # eval_strategy 오타 수정
        eval_steps=400,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
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
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5
    )
    
    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
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

# Function to parse model outputs and extract heads and relations
def parse_model_output(output_text, tokens):
    heads = [-1] * len(tokens)  # Initialize with -1
    deprels = [""] * len(tokens)  # Initialize with empty strings
    
    lines = output_text.strip().split("\n")
    for line in lines:
        parts = line.split(": ", 1)
        if len(parts) < 2:
            continue
            
        token, info = parts
        if token not in tokens:
            continue
            
        idx = tokens.index(token)
        
        # Extract head
        head_match = re.search(r'head=(\d+)', info)
        if head_match:
            try:
                heads[idx] = int(head_match.group(1))
            except (ValueError, IndexError):
                pass
                
        # Extract relation
        rel_match = re.search(r'relation=(\w+(?:_\w+)?)', info)
        if rel_match:
            deprels[idx] = rel_match.group(1)
            
    return heads, deprels

# Function to evaluate model on KLUE-DP metrics
import re
def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-DP metrics using local JSON data."""
    logger.info(f"Evaluating model: {model_config.name}")
    
    # 로컬 JSON 파일 로드
    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # 최대 평가 샘플 수로 제한
    val_subset = val_data[:MAX_EVAL_SAMPLES]  # JSON 리스트에서 슬라이싱
    
    model.eval()
    
    # Results tracking
    uas_correct = 0  # Unlabeled attachment score
    las_correct = 0  # Labeled attachment score
    total_tokens = 0
    
    # Generate detailed logs
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    logs = []
    
    for item in tqdm(val_subset, desc="Evaluating"):
        tokens = item["word_form"]
        gold_heads = item["head"]
        gold_deprels = item["deprel"]
        
        if not tokens:  # Skip empty samples
            continue
            
        # Create prompt for model
        prompt = f"Perform dependency parsing on the following Korean sentence. Provide head indices (0-based) and dependency relations for each token. Sentence: {' '.join(tokens)}"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract completion part
        completion = generated_text[len(prompt):]
        
        # Parse output to get heads and dependency relations
        pred_heads, pred_deprels = parse_model_output(completion, tokens)
        
        # Calculate metrics
        for i in range(len(tokens)):
            if i < len(gold_heads) and i < len(pred_heads):
                total_tokens += 1
                
                # UAS: correct head
                if pred_heads[i] == gold_heads[i]:
                    uas_correct += 1
                    
                    # LAS: correct head and dependency relation
                    if pred_deprels[i] == gold_deprels[i]:
                        las_correct += 1
        
        # Log details
        log_entry = {
            "sentence": " ".join(tokens),
            "gold_heads": gold_heads,
            "gold_deprels": gold_deprels,
            "pred_heads": pred_heads,
            "pred_deprels": pred_deprels,
            "generated_text": completion
        }
        logs.append(log_entry)
    
    # Calculate final metrics
    uas = uas_correct / total_tokens if total_tokens > 0 else 0
    las = las_correct / total_tokens if total_tokens > 0 else 0
    
    # Log results
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"UAS: {uas:.4f}")
    logger.info(f"LAS: {las:.4f}")
    logger.info(f"Total tokens evaluated: {total_tokens}")
    
    # Save logs and results
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        
    results = {
        "model": model_config.name,
        "UAS": uas,
        "LAS": las,
        "total_tokens": total_tokens
    }
    
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")
    
    return results

# Main execution function
if __name__ == "__main__":
    logger.info("Starting KLUE-DP training and evaluation")
    
    # Process each model configuration
    all_results = {}
    
    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")
        
        try:
            # Create output directories
            os.makedirs(model_config.output_dir, exist_ok=True)
            
            # Train model
            model, tokenizer = train_model(model_config)
            
            # Evaluate model
            results = evaluate_model(model, tokenizer, model_config)
            all_results[model_config.name] = results
            
            logger.info(f"Completed processing for {model_config.name}")
            
        except Exception as e:
            logger.error(f"Error processing {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
    
    # Save combined results
    combined_results_path = "klue_dp_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-DP training and evaluation completed")