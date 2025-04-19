import json
import os
import torch
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM, # Added for encoder models
    AutoModelForSeq2SeqLM, # Added for encoder-decoder models
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, # Added for seq2seq models
    PreTrainedModel
)
from transformers.utils import WEIGHTS_NAME # "pytorch_model.bin"
from transformers.utils.import_utils import is_peft_available

from dataclasses import dataclass, field
from typing import Optional
import functools # Needed for partial function application in map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool
    model_type: str # "causal", "encoder", or "encoder-decoder"
    max_length: int = 1024 # Default max length, can be overridden

# --- Updated MODEL_CONFIGS list ---
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="OLMo-1b-org",
    #     model_path="allenai/OLMo-1B",
    #     is_local=False,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="OLMo-7b-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B",
    #     is_local=True,
    #     model_type="causal"
    # ),

    # --- Encoder Model ---
    # Note: Fine-tuning BERT on a seq2seq task like this is non-standard.
    # It might learn *something*, but it's not its primary design.
    # Often, encoders are fine-tuned with a classification head.
    # We'll treat it like a Causal LM for simplicity given the data format.

    ModelConfig(
        name="Full_BERT-base-uncased",
        model_path="google-bert/bert-base-uncased",
        is_local=False,
        model_type="encoder", # Will use LM collator for this setup
        max_length=512 # BERT's typical max length
    ),

    # --- Encoder-Decoder Model ---
    # ModelConfig(
    #     name="Full_T5-base-Tuned",
    #     model_path="t5-base",
    #     is_local=False,
    #     model_type="encoder-decoder",
    #     max_length=512 # T5's typical max length
    # ),
]

# --- Data Loading Functions ---
def load_json_file(file_path):
    """Load a single JSON file and return its data or None if error occurs."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error reading file: {os.path.basename(file_path)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {os.path.basename(file_path)}: {e}")
        return None


def load_all_data(folder_path):
    """Load all JSON files from a folder using parallel processing."""
    logger.info(f"Starting data loading from: {folder_path}")
    all_data = {"input": [], "output": []}
    file_count = 0
    json_files = []
    if os.path.isdir(folder_path):
        json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    else:
        logger.error(f"Folder not found: {folder_path}")
        return None # Return None if folder doesn't exist

    if not json_files:
        logger.warning(f"No JSON files found in {folder_path}")
        return all_data # Return empty data if no files found

    # Use process pool for parallel file reading
    # Adjust max_workers if needed, especially on systems with many cores but limited memory
    num_workers = min(os.cpu_count(), 8) # Limit workers to avoid potential memory issues
    logger.info(f"Using {num_workers} workers for data loading.")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_json_file, file_path): file_path for file_path in json_files}

        for future in tqdm(as_completed(futures), total=len(json_files), desc="Loading files"):
            # file_path = futures[future] # Keep track if needed for debugging
            try:
                data = future.result()
                if data:
                    # Basic validation of data structure
                    if isinstance(data, list) and all(isinstance(item, dict) and "input" in item and "output" in item for item in data):
                        file_count += 1
                        all_data["input"].extend([item["input"] for item in data])
                        all_data["output"].extend([item["output"] for item in data])
                    else:
                        logger.warning(f"Skipping file with unexpected format: {futures[future]}")
            except Exception as e:
                logger.error(f"Error processing future for {futures[future]}: {e}")


    if not all_data["input"]:
         logger.error("No valid data loaded. Please check JSON files format and content.")
         return None

    logger.info(f"Loaded {len(all_data['input'])} samples from {file_count} files.")
    return all_data

# --- Preprocessing Functions ---

def preprocess_causal(examples, tokenizer, max_length):
    """Preprocess function for Causal LM: combines input/output."""
    # Format: "Input: {input} Output: {output}<eos>"
    texts = [f"Input: {inp}\nOutput: {out}{tokenizer.eos_token}"
             for inp, out in zip(examples["input"], examples["output"])]

    # Tokenize the combined texts
    tokenized_output = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False # Trainer/Collator will handle padding
    )

    # For Causal LM, the labels are the input_ids shifted
    # The DataCollatorForLanguageModeling handles this automatically if labels are not provided
    # So we just need input_ids and attention_mask
    return {
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"],
    }

def preprocess_seq2seq(examples, tokenizer, max_length):
    """Preprocess function for Encoder-Decoder models."""
    # Format inputs: "Input: {input}" (or just "{input}" depending on model expectations)
    # T5 often uses prefixes, but "Input: ..." is a reasonable generic approach.
    inputs = [f"Input: {inp}" for inp in examples["input"]]
    targets = [out for out in examples["output"]] # Keep outputs separate

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False # Collator handles padding
    )

    # Tokenize targets (labels)
    # Ensure EOS token is added if necessary, although T5 tokenizers might handle this.
    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding=False # Collator handles padding
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class CustomTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Saves the model checkpoint using pytorch_model.bin instead of safetensors
        by explicitly setting safe_serialization=False.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save the model
        if not hasattr(self.model, "save_pretrained"):
             raise ValueError("The model doesn't support save_pretrained functionality.")

        # --- 여기가 핵심: safe_serialization=False 추가 ---
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=False # Force saving in pytorch .bin format
        )
        # --- ---

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

# --- Main Execution ---
if __name__ == "__main__":
    # Set paths and constants
    FOLDER_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/DB/Refined_Datas/v2/Data_Final_Reversed"
    BASE_OUTPUT_DIR = "Fine_Tuned_Results" # Base directory for all model outputs

    # Load and prepare data ONCE
    logger.info("--- Starting Data Loading ---")
    all_data = load_all_data(FOLDER_PATH)

    if all_data is None or not all_data["input"]:
        logger.error("Failed to load data or data is empty. Exiting.")
        exit()

    # Verify input and output length match
    assert len(all_data["input"]) == len(all_data["output"]), "Input and output data counts do not match."

    # Convert to Hugging Face Dataset
    full_dataset = Dataset.from_dict(all_data)

    # Split into train and validation sets (90/10) ONCE
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Total dataset size: {len(full_dataset)}")
    logger.info(f"Training dataset: {len(split_dataset['train'])} samples, Validation dataset: {len(split_dataset['test'])} samples")


    # --- Loop through each model configuration ---
    for config in MODEL_CONFIGS:
        logger.info(f"--- Starting Fine-Tuning for: {config.name} ---")
        logger.info(f"Model Path: {config.model_path}, Type: {config.model_type}")

        output_dir = os.path.join(BASE_OUTPUT_DIR, config.name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. Load Tokenizer for the current model
            logger.info(f"Loading tokenizer for {config.name} from {config.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True,
                # use_fast=True # Generally faster, but check compatibility
            )

            # Set pad token if needed (important!)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")
                else:
                    # Add a pad token if EOS is also missing (unlikely for most models)
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.warning(f"Added pad_token: {tokenizer.pad_token}")
                    # Important: Resize model embeddings if adding tokens AFTER loading model
                    # We'll handle this after model loading if needed.


            # 2. Load Model based on type
            logger.info(f"Loading model: {config.name}")
            model_load_args = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16, # Use bfloat16
                "device_map": "auto" # Distribute across GPUs
            }
            if config.is_local and os.path.exists(os.path.join(config.model_path, "model.safetensors")):
                 model_load_args["use_safetensors"] = True
            elif config.is_local:
                 model_load_args["use_safetensors"] = False # Prefer .bin if .safetensors doesn't exist locally

            model = None
            added_pad_token = False
            if config.model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_load_args)
            elif config.model_type == "encoder-decoder":
                model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path, **model_load_args)
            elif config.model_type == "encoder":
                # Using MaskedLM, but will use LM collator as per our decision
                # This is non-standard for seq2seq tasks.
                logger.warning(f"Using AutoModelForMaskedLM for {config.name} but applying a Causal LM style training objective.")
                model = AutoModelForMaskedLM.from_pretrained(config.model_path, **model_load_args)
            else:
                logger.error(f"Unsupported model type '{config.model_type}' for {config.name}. Skipping.")
                continue # Skip to the next model

            # Resize embeddings if a new pad token was added to the tokenizer *after* it was loaded
            if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
                 logger.warning(f"Resizing model token embeddings to accommodate new pad token ({tokenizer.pad_token}).")
                 model.resize_token_embeddings(len(tokenizer))
                 added_pad_token = True # Flag to potentially avoid issues if pad is same as eos


            # 3. Select and Apply Preprocessing Function
            logger.info("Tokenizing datasets...")
            preprocess_fn = None
            if config.model_type == "causal" or config.model_type == "encoder": # Treat encoder like causal here
                preprocess_fn = functools.partial(preprocess_causal, tokenizer=tokenizer, max_length=config.max_length)
            elif config.model_type == "encoder-decoder":
                 preprocess_fn = functools.partial(preprocess_seq2seq, tokenizer=tokenizer, max_length=config.max_length)

            if preprocess_fn:
                 # Use functools.partial to pass the specific tokenizer and max_length
                 tokenized_train = split_dataset["train"].map(
                     preprocess_fn,
                     batched=True,
                     remove_columns=split_dataset["train"].column_names,
                     num_proc=min(os.cpu_count(), 8), # Limit parallelism
                     desc=f"Tokenizing train data for {config.name}"
                 )
                 tokenized_eval = split_dataset["test"].map(
                     preprocess_fn,
                     batched=True,
                     remove_columns=split_dataset["test"].column_names,
                     num_proc=min(os.cpu_count(), 8), # Limit parallelism
                     desc=f"Tokenizing eval data for {config.name}"
                 )
            else:
                 logger.error(f"No preprocessing function defined for model type {config.model_type}. Skipping {config.name}.")
                 continue

            # 4. Select Data Collator
            data_collator = None
            if config.model_type == "causal" or config.model_type == "encoder":
                # For Causal LM, the collator handles padding and optionally creates labels by shifting inputs.
                # mlm=False means standard language modeling (predict next token).
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )
            elif config.model_type == "encoder-decoder":
                # For Seq2Seq, the collator pads inputs and labels independently.
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=tokenizer,
                    model=model # Recommended for potential model-specific padding
                )
            else: # Should not happen due to earlier checks
                logger.error(f"Cannot determine data collator for model type {config.model_type}. Skipping {config.name}.")
                continue

            # 5. Set up Training Arguments
            # Adjust parameters potentially based on model size if needed
            training_args = TrainingArguments(
                output_dir=output_dir,
                eval_strategy="steps",
                eval_steps=500,
                learning_rate=5e-5, # May need tuning per model
                per_device_train_batch_size=16, # Adjust based on GPU memory
                per_device_eval_batch_size=16,  # Adjust based on GPU memory
                gradient_accumulation_steps=2, # Effective batch size = 4 * 8 * num_gpus = 32 * num_gpus
                num_train_epochs=3,
                weight_decay=0.01,
                save_total_limit=2, # Save fewer checkpoints to save space
                save_strategy="steps",
                save_steps=500,
                logging_dir=os.path.join(output_dir, "logs"), # Log within model's output dir
                logging_steps=100,
                fp16=False, # Disabled as we use bf16
                bf16=True,  # Use bfloat16 precision (ensure hardware support)
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="none",  # Disable wandb/tensorboard reporting unless configured
                gradient_checkpointing=True, # Enable gradient checkpointing
                optim="adamw_torch", # Use efficient AdamW
                # Consider adding:
                # remove_unused_columns=False, # Important if preprocess adds extra columns accidentally
                # dataloader_num_workers=2, # Can speed up data loading
            )

            # 6. Initialize Trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                data_collator=data_collator,
                tokenizer=tokenizer, # Pass tokenizer for saving convenience
            )

            # 7. Start Training
            logger.info(f"Starting training for {config.name}...")
            train_result = trainer.train()
            logger.info(f"Training finished for {config.name}.")

            # Log metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # Evaluate final model
            logger.info(f"Evaluating final model for {config.name}...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)


            # 8. Save Final Model & Tokenizer
            logger.info(f"Saving final model and tokenizer for {config.name} to: {output_dir}")
            # Trainer saves the best model if load_best_model_at_end=True
            # We save again here explicitly to ensure the latest state (or best) is saved clearly
            trainer.save_model(output_dir) # Saves model and tokenizer
            # model.save_pretrained(output_dir) # Redundant if using trainer.save_model
            # tokenizer.save_pretrained(output_dir) # Redundant if using trainer.save_model
            trainer.save_state() # Save trainer state (optimizer, scheduler, etc.)

            logger.info(f"Fine-tuning for {config.name} completed successfully!")

        except Exception as e:
            logger.error(f"An error occurred during fine-tuning {config.name}: {e}", exc_info=True) # Log traceback
            # Optionally continue to the next model or break
            # continue

        finally:
            # Clean up GPU memory before next iteration
            del model
            del tokenizer
            del trainer
            # del tokenized_train # Optional, depends on memory
            # del tokenized_eval # Optional, depends on memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Cleaned up resources for {config.name}")


    logger.info("--- All Model Fine-Tuning Attempts Completed ---")
