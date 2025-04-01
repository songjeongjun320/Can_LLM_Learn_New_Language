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
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir

# Model configurations (from first code)
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_dp_results/full-olmo7B-org-klue-dp"
    # ),
    ModelConfig(
        name="full-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_dp_results/full-olmo1B-org-klue-dp"
    ),
    ModelConfig(
        name="full-OLMo-1b-v12", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo1B-v12", 
        output_dir="klue_dp_results/full-olmo1B-v12-klue-dp"
    ),
    ModelConfig(
        name="full-OLMo-7b-v13", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo7B-v13", 
        output_dir="klue_dp_results/full-olmo7B-v13-klue-dp"
    ),
    ModelConfig(
        name="full-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_dp_results/full-llama3.2-3b-klue-dp"
    )
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
    """Load model and tokenizer based on model configuration."""
    logger.info(f"Loading model: {model_config.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, 
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model with bfloat16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Model loaded successfully: {model_config.name}")
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

# Biaffine layer for arc scoring
class Biaffine(torch.nn.Module):
    def __init__(self, in1_dim, in2_dim, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in1_dim, in2_dim))
        self.bias = bias
        if bias:
            self.bias1 = torch.nn.Parameter(torch.randn(in1_dim))
            self.bias2 = torch.nn.Parameter(torch.randn(in2_dim))
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias:
            torch.nn.init.zeros_(self.bias1)
            torch.nn.init.zeros_(self.bias2)
    
    def forward(self, x1, x2):
        batch, len1, dim1 = x1.size()
        batch, len2, dim2 = x2.size()
        
        if self.bias:
            x1 = x1 + self.bias1.unsqueeze(0).unsqueeze(0)
            x2 = x2 + self.bias2.unsqueeze(0).unsqueeze(0)
        
        part1 = torch.matmul(x1.reshape(-1, dim1), self.weight)
        part1 = part1.reshape(batch, len1, dim2)
        scores = torch.bmm(part1, x2.transpose(1, 2))
        
        return scores

# Dependency Parser model
class DependencyParser(torch.nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.hidden_size = base_model.config.hidden_size
        
        # Add dependency parsing specific layers
        self.arc_biaffine = Biaffine(self.hidden_size, self.hidden_size, bias=True)
        self.label_classifier = torch.nn.Linear(self.hidden_size, NUM_LABELS)
        self.init_weights()
        
        logger.info(f"Initialized dependency parser with hidden size: {self.hidden_size}")
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.label_classifier.weight)
        self.label_classifier.bias.data.zero_()
        
    def gradient_checkpointing_enable(self):
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not available for this model")
        
    def forward(self, input_ids, attention_mask, word_attention_mask=None, head_indices=None, deprel_labels=None, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # If training with language modeling loss, return base model outputs
        if labels is not None:
            return outputs
        
        # For dependency parsing specific tasks
        hidden_states = outputs.hidden_states[-1]
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get arc scores and label logits
        arc_scores = self.arc_biaffine(hidden_states, hidden_states)
        label_logits = self.label_classifier(hidden_states)
        
        loss = None
        if head_indices is not None and deprel_labels is not None:
            arc_loss = F.cross_entropy(
                arc_scores.view(-1, seq_len),
                head_indices.view(-1),
                ignore_index=-100
            )
            
            label_loss = F.cross_entropy(
                label_logits.view(-1, NUM_LABELS),
                deprel_labels.view(-1),
                ignore_index=-100
            )
            
            loss = arc_loss + label_loss
        
        return {
            "loss": loss,
            "arc_scores": arc_scores,
            "label_logits": label_logits
        }

# Custom Trainer for language modeling approach
from transformers import DataCollatorForLanguageModeling

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
    
    # Load datasets
    logger.info("Loading train and validation datasets")
    train_dataset = DependencyParsingDataset(JSON_TRAIN_DATASET_PATH, tokenizer)
    val_dataset = DependencyParsingDataset(JSON_VAL_DATASET_PATH, tokenizer)
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
        fp16=True,  # FP16으로 전환
        bf16=False,  # BF16 비활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Warmup 비율 감소
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=False,  # 체크포인팅 비활성화
        optim="adamw_torch",  # 필요 시 "adamw_8bit"로 변경
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