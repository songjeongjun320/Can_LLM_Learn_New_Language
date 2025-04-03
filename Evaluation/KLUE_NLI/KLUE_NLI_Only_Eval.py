import sys
print(f"Python executable path: {sys.executable}")
# Standard library imports
import json
import logging
import os
import re
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
from datasets import Dataset # Keep Dataset for potential use if needed, though not strictly required for this eval loop
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # Import metrics functions
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig # Keep in case needed for base model loading variations
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_nli_evaluation.log") # Log file for evaluation
    ]
)
logger = logging.getLogger(__name__)

# KLUE NLI Label Definitions
NLI_LABELS = ["entailment", "neutral", "contradiction"]
NUM_LABELS = len(NLI_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(NLI_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(NLI_LABELS)}
logger.info(f"Total number of KLUE-NLI labels: {NUM_LABELS}")

# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path # Path to the ORIGINAL base model
        self.output_dir = output_dir # Directory where 'final' adapter is saved
        self.is_local = is_local

# Model configurations - Make sure these paths are correct
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-OLMo-1b-org",
        model_path="allenai/OLMo-1B", # Base model path
        output_dir="klue_nli_results/lora-olmo1B-org-klue-nli", # Where 'final' adapter lives
        is_local=False
    ),
    ModelConfig(
        name="lora-OLMo-1b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", # Base model path
        output_dir="klue_nli_results/lora-olmo1B-v12-klue-nli", # Where 'final' adapter lives
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org",
    #     model_path="allenai/OLMo-7B", # Base model path
    #     output_dir="klue_nli_results/lora-olmo7B-org-klue-nli", # Where 'final' adapter lives
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", # Base model path
    #     output_dir="klue_nli_results/lora-olmo7B-v13-klue-nli", # Where 'final' adapter lives
    #     is_local=True
    # ),
    ModelConfig(
        name="lora-Llama-3.2-3b",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", # Base model path
        output_dir="klue_nli_results/lora-llama3.2-3b-klue-nli", # Where 'final' adapter lives
        is_local=True
    )
]

# Configuration parameters
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_nli_validation.json"
MAX_LENGTH = 300
MAX_EVAL_SAMPLES = 200 # Limit evaluation samples if needed, set to None for full evaluation
EVAL_BATCH_SIZE = 8 # Batch size for evaluation inference

# Function to load fine-tuned PEFT model and tokenizer
def load_finetuned_model_and_tokenizer(model_config):
    """Loads the base model and attaches the PEFT adapter."""
    adapter_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Loading fine-tuned model for {model_config.name}")
    logger.info(f"Base model path: {model_config.model_path}")
    logger.info(f"Adapter path: {adapter_path}")

    if not os.path.exists(adapter_path):
         raise FileNotFoundError(f"Adapter directory not found: {adapter_path}. Ensure training completed and saved the model.")

    is_local = model_config.is_local

    # Load tokenizer from the adapter path (where it was saved during training)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True
    )
    logger.info(f"Tokenizer loaded from {adapter_path}")

    # Check and set pad token
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    logger.info(f"Loading base model from {model_config.model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for consistency
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True
    )
    logger.info("Base model loaded.")

    # Load the PEFT model by attaching the adapter to the base model
    logger.info(f"Loading PEFT adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("PEFT adapter loaded and attached to the base model.")

    # Optional: Merge adapter weights into the base model for potentially faster inference
    # logger.info("Merging adapter weights into base model...")
    # model = model.merge_and_unload()
    # logger.info("Adapter merged and unloaded.")
    # Note: If merged, the model object is no longer a PeftModel but the base model type with updated weights.
    # Keeping it unmerged is often fine for evaluation.

    return model, tokenizer
# Evaluation function (수정됨)
def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-NLI metrics."""
    logger.info("============================================")
    logger.info(f"Evaluating model: {model_config.name}")
    logger.info("============================================")

    # Ensure output directory exists for saving results
    os.makedirs(model_config.output_dir, exist_ok=True)

    logger.info(f"Loading validation data from: {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        logger.info(f"Loaded {len(val_data)} validation samples.")
    except FileNotFoundError:
        logger.error(f"Validation data file not found: {JSON_VAL_DATASET_PATH}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {JSON_VAL_DATASET_PATH}")
        return None

    if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES > 0:
        val_subset = val_data[:MAX_EVAL_SAMPLES]
        logger.info(f"Using a subset of {len(val_subset)} samples for evaluation.")
    else:
        val_subset = val_data
        logger.info("Using the full validation dataset.")

    if not val_subset:
        logger.warning("Validation subset is empty. Skipping evaluation.")
        return None

    model.eval()
    try:
        device = next(model.parameters()).device
        logger.info(f"Model placed on device: {device}")
    except StopIteration:
        logger.error("Could not determine model device. Assuming CPU.")
        device = torch.device("cpu") # Fallback

    true_labels = []
    pred_labels = []
    logs = []

    # --- Batch Processing Implementation ---
    for i in tqdm(range(0, len(val_subset), EVAL_BATCH_SIZE), desc="Evaluating"):
        batch_items = val_subset[i:i+EVAL_BATCH_SIZE]
        batch_premises = [item["premise"] for item in batch_items]
        batch_hypotheses = [item["hypothesis"] for item in batch_items]
        batch_gold_labels = [item["label"] for item in batch_items]

        # Tokenize batch
        try:
            encodings = tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length", # Pad to max_length
                return_tensors="pt"
            ) # Move tensors to device later, after potential modification

            # ================== 수정 시작 ==================
            # OLMo 모델은 token_type_ids를 사용하지 않으므로 제거
            if 'token_type_ids' in encodings:
                del encodings['token_type_ids']
            # ================== 수정 끝 ====================

            # Move tensors to the correct device
            encodings = {k: v.to(device) for k, v in encodings.items()}

        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            logger.error(f"Problematic premises: {batch_premises}")
            logger.error(f"Problematic hypotheses: {batch_hypotheses}")
            continue

        # Predict batch
        with torch.no_grad():
            try:
                outputs = model(**encodings) # 수정된 encodings 전달
                logits = outputs.logits
                sequence_lengths = torch.ne(encodings['input_ids'], tokenizer.pad_token_id).sum(-1) - 1
                last_token_logits = logits[torch.arange(logits.shape[0], device=device), sequence_lengths]

            except Exception as e:
                 logger.error(f"Error during model inference: {e}")
                 continue

        batch_predictions = torch.argmax(last_token_logits, dim=-1).cpu().numpy()

        true_labels.extend(batch_gold_labels)
        pred_labels.extend(batch_predictions)

        # Log details for this batch
        for premise, hypothesis, gold, pred in zip(batch_premises, batch_hypotheses, batch_gold_labels, batch_predictions):
             if 0 <= pred < NUM_LABELS : # Ensure prediction is within valid label indices
                logs.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "gold_label": ID2LABEL.get(gold, "INVALID_GOLD"),
                    "pred_label": ID2LABEL.get(pred, "INVALID_PRED")
                })
             else:
                 logs.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "gold_label": ID2LABEL.get(gold, "INVALID_GOLD"),
                    "pred_label": f"OUT_OF_BOUNDS ({pred})"
                 })
                 logger.warning(f"Prediction index {pred} is out of bounds for labels {list(ID2LABEL.keys())}.")


    # --- End Batch Processing ---

    if not true_labels:
        logger.warning("No evaluation results generated. Cannot calculate metrics.")
        return None

    # Filter out-of-bounds predictions if any occurred
    aligned_true = []
    aligned_pred = []
    for true, pred in zip(true_labels, pred_labels):
        if 0 <= pred < NUM_LABELS:
            aligned_true.append(true)
            aligned_pred.append(pred)
        else:
             logger.warning(f"Excluding out-of-bounds prediction {pred} from metrics calculation.")

    true_labels = aligned_true
    pred_labels = aligned_pred


    if not true_labels:
        logger.warning("No valid predictions remaining after filtering. Cannot calculate metrics.")
        return None

    # Calculate metrics
    try:
        accuracy = accuracy_score(true_labels, pred_labels)
        # ================== 수정 시작 (함수 이름 오타 수정) ==================
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, labels=list(ID2LABEL.keys()), zero_division=0
        )
        # ================== 수정 끝 ====================
        per_class_metrics = {
            ID2LABEL[i]: {"precision": p, "recall": r, "f1": f, "support": int(s)} # Cast support to int for JSON
            for i, (p, r, f, s) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, support_per_class))
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        logger.exception("Metrics calculation failed.") # 스택 트레이스 포함 로깅
        return None


    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Total Samples Evaluated: {len(true_labels)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro Precision: {precision_macro:.4f}")
    logger.info(f"Macro Recall: {recall_macro:.4f}")
    logger.info(f"Macro F1: {f1_macro:.4f}")
    logger.info("Per-class metrics:")
    # Ensure per_class_metrics is serializable
    try:
        logger.info(json.dumps(per_class_metrics, indent=2))
    except TypeError as e:
        logger.error(f"Could not serialize per_class_metrics: {e}")
        logger.info(str(per_class_metrics)) # Fallback to string representation


    # Save logs and results
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
         logger.error(f"Failed to save evaluation logs to {log_file_path}: {e}")


    # Convert numpy types to standard python types for JSON serialization
    results = {
        "model": model_config.name,
        "accuracy": float(accuracy) if isinstance(accuracy, np.generic) else accuracy,
        "precision_macro": float(precision_macro) if isinstance(precision_macro, np.generic) else precision_macro,
        "recall_macro": float(recall_macro) if isinstance(recall_macro, np.generic) else recall_macro,
        "f1_macro": float(f1_macro) if isinstance(f1_macro, np.generic) else f1_macro,
        "total_samples_evaluated": len(true_labels),
        "per_class_metrics": per_class_metrics # Already converted inside the dict comprehension
    }
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    try:
        with open(results_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to: {results_file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results to {results_file_path}: {e}")


    logger.info(f"Evaluation logs saved to: {log_file_path}")


    return results
    
# Main execution block for evaluation
if __name__ == "__main__":
    logger.info("Starting KLUE-NLI evaluation for pre-tuned models")

    all_results = {}

    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")

        try:
            # Load the fine-tuned model and tokenizer
            model, tokenizer = load_finetuned_model_and_tokenizer(model_config)

            # Evaluate the loaded model
            results = evaluate_model(model, tokenizer, model_config)
            if results:
                 all_results[model_config.name] = results

            logger.info(f"Completed evaluation for {model_config.name}")

            # Clean up memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Cleaned up resources for {model_config.name}")


        except FileNotFoundError as e:
             logger.error(f"Skipping {model_config.name} due to missing file/directory: {e}")
        except Exception as e:
            logger.error(f"Error evaluating {model_config.name}: {str(e)}")
            logger.exception("Exception details:")

    # Save combined results
    combined_results_path = "klue_nli_results/combined_evaluation_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"All evaluation results saved to: {combined_results_path}")
    logger.info("KLUE-NLI evaluation completed")