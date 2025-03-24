from datasets import Dataset
import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """Load a single JSON file and return its data or None if error occurs."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error reading file: {os.path.basename(file_path)}")
        return None

def load_all_data(folder_path):
    """Load all JSON files from a folder using parallel processing."""
    logger.info("Starting data loading...")
    all_data = {"input": [], "output": []}
    file_count = 0
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Use process pool for parallel file reading
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(load_json_file, file_path): file_path for file_path in json_files}
        
        for future in tqdm(as_completed(futures), total=len(json_files), desc="Loading files"):
            file_path = futures[future]
            data = future.result()
            if data:
                file_count += 1
                all_data["input"].extend([item["input"] for item in data])
                all_data["output"].extend([item["output"] for item in data])
    
    logger.info(f"Loaded {len(all_data['input'])} samples from {file_count} files.")
    return all_data

def preprocess_function(examples):
    """Preprocess function for tokenizing and combining input/output pairs."""
    # Format: "Input: {input} Output: {output}"
    prompts = [f"Input: {input}\nOutput: " for input in examples["input"]]
    
    # Tokenize inputs and outputs separately
    inputs = tokenizer(prompts, truncation=True, max_length=1024, padding=False)
    outputs = tokenizer(examples["output"], truncation=True, max_length=1024, padding=False)
    
    result = {
        "input_ids": [],
        "attention_mask": []
    }
    
    max_length = 1024
    for i in range(len(prompts)):
        input_ids = inputs["input_ids"][i]
        output_ids = outputs["input_ids"][i]
        
        # Add EOS token if not present
        if output_ids[-1] != tokenizer.eos_token_id:
            output_ids.append(tokenizer.eos_token_id)
        
        # Combine input and output
        combined_ids = input_ids + output_ids
        attention_mask = [1] * len(combined_ids)
        
        # Truncate if necessary
        if len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        
        result["input_ids"].append(combined_ids)
        result["attention_mask"].append(attention_mask)
    
    return result

if __name__ == "__main__":
    # Set paths and constants
    FOLDER_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/DB/Refined_Datas/v1/Data_Final_Reversed"
    BASE_MODEL = "allenai/OLMo-7B"
    OUTPUT_DIR = "Fine_Tuned_Results/olmo7B-v13"
    
    # Load and prepare data
    all_data = load_all_data(FOLDER_PATH)
    
    # Verify input and output length match
    assert len(all_data["input"]) == len(all_data["output"]), "Input and output data counts do not match."
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(all_data)
    
    # Split into train and validation sets (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Training dataset: {len(dataset['train'])} samples, Validation dataset: {len(dataset['test'])} samples")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Memory-efficient model loading
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Automatically distribute model across available GPUs
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_train = dataset["train"].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count(),  # Use all available CPU cores
        desc="Tokenizing training data"
    )
    tokenized_eval = dataset["test"].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing validation data"
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,  # Use bfloat16 precision
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Disable external logging
        # Additional optimizations
        gradient_checkpointing=True,  # Save memory with gradient checkpointing
        optim="adamw_torch",  # Use more efficient optimizer
    )
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create checkpoint directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize trainer and start training
    logger.info("Initializing trainer and starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Run training
    trainer.train()
    
    # Save final model and tokenizer
    logger.info(f"Saving final model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Fine-tuning completed successfully!")