# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Literal
import statistics
import traceback

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Updated ModelConfig dataclass ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"

# --- Updated MODEL_CONFIGS list ---
MODEL_CONFIGS = [
    # --- Causal Models ---
    ModelConfig(
        name="OLMo-1b-org",
        model_path="allenai/OLMo-1B",
        is_local=False,
        model_type="causal"
    ),
    ModelConfig(
        name="OLMo-1b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B",
        is_local=True,
        model_type="causal"
    ),
    ModelConfig(
        name="OLMo-7b-org",
        model_path="allenai/OLMo-7B",
        is_local=False,
        model_type="causal"
    ),
    ModelConfig(
        name="OLMo-7b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B",
        is_local=True,
        model_type="causal"
    ),
    ModelConfig(
        name="Llama-3.2-3b",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b",
        is_local=True,
        model_type="causal"
    ),
    ModelConfig(
        name="Llama-4-Scout-17B-16E",
        model_path="meta-llama/Llama-4-Scout-17B-16E",
        is_local=False,
        model_type="causal"
    ),
    # --- Encoder Model ---
    ModelConfig(
        name="BERT-base-uncased",
        model_path="bert-base-uncased",
        is_local=False,
        model_type="encoder"
    ),
    ModelConfig(
        name="BERT-base-uncased-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
    # --- Encoder-Decoder Model ---
    ModelConfig(
        name="T5-base",
        model_path="t5-base",
        is_local=False,
        model_type="encoder-decoder"
    ),
    ModelConfig(
        name="T5-base-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_T5-base-Tuned",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder-decoder"
    ),
]

# --- Configuration ---
WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_ch.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAFE_MAX_LENGTH = 20  # Define a reasonable max length for tokenization
# NUM_SAMPLES removed, will be determined per model

# --- *** Dynamic Path Generation *** ---
try:
    word_pairs_basename = os.path.basename(WORD_PAIRS_PATH)
    # Extract language pair identifier (e.g., "en_ch" from "en_ch.json")
    lang_pair, _ = os.path.splitext(word_pairs_basename)
    if not lang_pair:
        raise ValueError("Could not extract language pair from WORD_PAIRS_PATH filename.")

    # Create dynamic folder and file names
    INDIVIDUAL_RESULTS_FOLDER = f'embedding_results_{lang_pair}'
    # Combined results path now reflects the language pair, not a specific model
    COMBINED_RESULTS_PATH = f'combined_embedding_results_{lang_pair}.json'

    os.makedirs(INDIVIDUAL_RESULTS_FOLDER, exist_ok=True)
    logger.info(f"Using language pair: {lang_pair}")
    logger.info(f"Individual results will be saved in: {INDIVIDUAL_RESULTS_FOLDER}")
    logger.info(f"Combined results will be saved to: {COMBINED_RESULTS_PATH}")

except Exception as e:
    logger.error(f"Failed to determine dynamic paths from WORD_PAIRS_PATH '{WORD_PAIRS_PATH}': {e}")
    logger.error("Please ensure WORD_PAIRS_PATH has a filename like 'lang1_lang2.json'.")
    exit()

logger.info(f"Using device: {DEVICE}")

# --- Helper Function to Get Word Embedding (with explicit max_length) ---
def get_word_embedding(model, tokenizer, word, device, max_len, model_type: str):
    """
    Gets the average embedding of the last hidden state for a given word.
    Handles different model types, specifically calling only the encoder for T5.
    """
    if not word:
        logger.warning("Skipping empty word.")
        return None
    try:
        inputs = tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(device)

        if inputs.input_ids.shape[1] == 0:
            logger.warning(f"Tokenization of '{word}' resulted in empty input_ids. Skipping.")
            return None

        with torch.no_grad():
            model_inputs = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs.get('attention_mask'), # Use .get for safety
                # Request hidden states for all types, though we might use last_hidden_state directly
                "output_hidden_states": True
            }
            # Remove None attention mask if tokenizer didn't provide one
            if model_inputs["attention_mask"] is None:
                del model_inputs["attention_mask"]

            # --- Conditional Model Execution ---
            last_hidden_state = None # Initialize
            if model_type == "encoder-decoder":
                # For T5 (and potentially other enc-dec models loaded with AutoModel),
                # call ONLY the encoder part.
                encoder_args = {k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'output_hidden_states']}
                outputs = model.encoder(**encoder_args) # Pass relevant args
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'): # Fallback if needed
                     last_hidden_state = outputs.hidden_states[-1]
                else:
                    logger.error(f"Could not retrieve last_hidden_state or hidden_states from encoder for word '{word}'.")
                    return None
            elif model_type == "encoder":
                 # For BERT-like models loaded with AutoModel
                 outputs = model(**model_inputs)
                 if hasattr(outputs, 'last_hidden_state'):
                     last_hidden_state = outputs.last_hidden_state
                 elif hasattr(outputs, 'hidden_states'): # Fallback
                     last_hidden_state = outputs.hidden_states[-1]
                 else:
                     logger.error(f"Could not retrieve hidden_states from encoder model for word '{word}'.")
                     return None
            elif model_type == "causal":
                 # For CausalLM models loaded with AutoModelForCausalLM
                 outputs = model(**model_inputs)
                 if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                     last_hidden_state = outputs.hidden_states[-1]
                 else:
                      logger.error(f"Could not retrieve hidden_states from causal model for word '{word}'.")
                      return None
            else:
                logger.error(f"Unsupported model_type '{model_type}' for word '{word}'.")
                return None
            # --- End Conditional Model Execution ---

            # --- Averaging using attention mask (if available) ---
            if last_hidden_state is None: # Should not happen if checks above work, but safety first
                logger.error(f"Cannot compute embedding as last_hidden_state is None after model execution for word '{word}'.")
                return None

            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None :
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9) # Avoid division by zero
                embedding = sum_embeddings / sum_mask
            else:
                 # If no attention mask, just average all tokens (less ideal for padded sequences)
                 logger.warning(f"No attention mask found for word '{word}'. Averaging all tokens.")
                 embedding = last_hidden_state.mean(dim=1)

        return embedding.squeeze(0) # Remove batch dimension

    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Embedding error for '{word}' ({error_type}): {e}")
        logger.error("--- Traceback ---")
        logger.error(traceback.format_exc())
        logger.error("--- End Traceback ---")
        return None

# --- Main Execution ---
all_model_summaries: Dict[str, Dict[str, Any]] = {}

# --- Load Full Word Pairs ONCE ---
try:
    with open(WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
        full_word_pairs_dict = json.load(f) # Load the full dictionary
    full_word_pairs_items = list(full_word_pairs_dict.items()) # Convert to list of items once
    logger.info(f"Loaded {len(full_word_pairs_items)} total word pairs from {WORD_PAIRS_PATH}")
except Exception as e:
    logger.error(f"Failed to load word pairs from {WORD_PAIRS_PATH}: {e}")
    exit()
# --- End Load Full Word Pairs ---

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Processing Model: {config.name} (Path: {config.model_path}) ---")
    model_individual_results: Dict[str, Dict[str, Optional[float]]] = {}
    model = None
    tokenizer = None
    model_similarities: List[float] = []
    processed_pairs = 0
    successful_pairs = 0
    error_occurred = False

    # --- *** Determine Number of Samples for This Model *** ---
    num_samples_for_this_model = -1 # Default: process all
    if config.name == "Llama-4-Scout-17B-16E":
        num_samples_for_this_model = 500
        logger.info(f"Specific model '{config.name}' detected. Setting NUM_SAMPLES = {num_samples_for_this_model}")
    else:
        logger.info(f"Processing all available word pairs for model '{config.name}'.")

    # --- *** Select Word Pairs for This Model *** ---
    if num_samples_for_this_model > 0 and num_samples_for_this_model < len(full_word_pairs_items):
        word_pairs_items_to_process = full_word_pairs_items[:num_samples_for_this_model]
        total_pairs_to_process_this_model = num_samples_for_this_model
        logger.info(f"Sampling the first {num_samples_for_this_model} word pairs for processing.")
    else:
        # If num_samples_for_this_model is -1 or >= total, process all pairs
        word_pairs_items_to_process = full_word_pairs_items # Use all pairs
        total_pairs_to_process_this_model = len(full_word_pairs_items)
        logger.info(f"Processing all {total_pairs_to_process_this_model} loaded word pairs.")
    # --- *** End Select Word Pairs *** ---


    try:
        logger.info(f"Loading tokenizer from: {config.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=config.is_local
        )
        original_max_len = getattr(tokenizer, 'model_max_length', 'N/A')
        logger.info(f"Tokenizer original model_max_length: {original_max_len}")

        if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set EOS token as PAD token.")
        elif tokenizer.pad_token is None:
             logger.warning(f"Tokenizer for {config.name} does not have a pad_token or eos_token. Padding might behave unexpectedly.")

        logger.info(f"Loading model from: {config.model_path} using appropriate AutoModel class (bfloat16)...")
        if config.model_type == "causal":
            ModelClass = AutoModelForCausalLM
        elif config.model_type == "encoder" or config.model_type == "encoder-decoder":
             ModelClass = AutoModel
        else:
             logger.error(f"Unknown model_type '{config.model_type}' for {config.name}. Skipping.")
             raise ValueError(f"Unknown model_type: {config.model_type}")

        loaded_object = ModelClass.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Using bfloat16 as in original
            local_files_only=config.is_local,
            output_hidden_states=True,
            device_map="auto"
        )
        logger.info(f"Type of loaded object from {config.model_path}: {type(loaded_object)}")
        if not isinstance(loaded_object, torch.nn.Module):
             logger.error(f"Failed to load model correctly from {config.model_path}. Received type: {type(loaded_object)}")
             raise TypeError(f"Expected a PyTorch model from {config.model_path}, but got {type(loaded_object)}")

        model = loaded_object # Already on device due to device_map="auto"

        if tokenizer.pad_token_id is not None:
            if hasattr(model, 'config') and model.config is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")
            else:
                logger.warning(f"Could not set pad_token_id on model config for {config.name}.")
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
             if hasattr(model, 'config') and model.config is not None:
                logger.warning(f"Attempting to set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) as pad_token_id was None.")
                model.config.pad_token_id = tokenizer.eos_token_id
             else:
                logger.warning(f"Could not set pad_token_id using eos_token_id on model config for {config.name}.")

        model.eval()
        logger.info(f"Model from {config.model_path} evaluation mode set.")

        # --- Process selected pairs for this model ---
        for korean_word, english_word in tqdm(word_pairs_items_to_process, desc=f"Processing pairs for {config.name}", total=total_pairs_to_process_this_model):
            processed_pairs += 1

            # korean_word_lower = korean_word.lower() # Keep original case as it might matter for some tokenizers/models
            # english_word_lower = english_word.lower()
            embedding_k = get_word_embedding(model, tokenizer, korean_word, DEVICE, SAFE_MAX_LENGTH, config.model_type)
            embedding_e = get_word_embedding(model, tokenizer, english_word, DEVICE, SAFE_MAX_LENGTH, config.model_type)

            similarity = None
            if embedding_k is not None and embedding_e is not None:
                try:
                    # Ensure embeddings are float32 for cosine similarity if needed (though bfloat16 might work)
                    embedding_k_float = embedding_k.float()
                    embedding_e_float = embedding_e.float()
                    similarity = F.cosine_similarity(embedding_k_float.unsqueeze(0), embedding_e_float.unsqueeze(0)).item()
                    model_similarities.append(similarity)
                    successful_pairs += 1
                except Exception as e:
                    logger.error(f"Similarity calculation error for '{korean_word}' & '{english_word}': {e}")
                    similarity = None
            else:
                logger.warning(f"Skipping similarity calculation for pair ('{korean_word}', '{english_word}') due to embedding error.")
                similarity = None

            # Store result for this pair, even if similarity calculation failed
            model_individual_results[korean_word] = {"english_word": english_word, "similarity": similarity}

        # --- Save Individual Results (using dynamic folder name) ---
        individual_filename = os.path.join(INDIVIDUAL_RESULTS_FOLDER, f"{config.name}_similarity_{lang_pair}.json") # Added lang_pair to filename too for clarity

        try:
            with open(individual_filename, 'w', encoding='utf-8') as f_ind:
                json.dump(model_individual_results, f_ind, ensure_ascii=False, indent=4)
            logger.info(f"Saved individual results for model {config.name} to {individual_filename}")
        except Exception as e:
            logger.error(f"Failed to save individual results for model {config.name}: {e}")

    except Exception as e:
        logger.error(f"Critical error processing model {config.name} (Path: {config.model_path}): {e}")
        logger.error("--- Critical Error Traceback ---")
        logger.error(traceback.format_exc())
        logger.error("--- End Critical Error Traceback ---")
        error_occurred = True
        all_model_summaries[config.name] = {"error": str(e), "model_path": config.model_path, "lang_pair": lang_pair}

    finally:
        # Calculate Summary Metrics
        if not error_occurred:
            avg_similarity = np.mean(model_similarities) if model_similarities else 0.0
            std_dev_similarity = np.std(model_similarities) if len(model_similarities) > 1 else 0.0
            median_similarity = statistics.median(model_similarities) if model_similarities else 0.0
            success_rate = (successful_pairs / processed_pairs) * 100 if processed_pairs > 0 else 0.0
            all_model_summaries[config.name] = {
                "model_path": config.model_path,
                "lang_pair": lang_pair,
                "processed_pairs": processed_pairs,
                "total_pairs_in_file": len(full_word_pairs_items), # Add total available pairs for context
                "successful_pairs": successful_pairs,
                "success_rate_percent": round(success_rate, 2),
                "average_similarity": round(float(avg_similarity), 6),
                "median_similarity": round(float(median_similarity), 6),
                "std_dev_similarity": round(float(std_dev_similarity), 6)
            }
            logger.info(f"Summary for model {config.name}: Avg Sim={avg_similarity:.4f}, Median Sim={median_similarity:.4f}, StdDev={std_dev_similarity:.4f}, Success Rate={success_rate:.2f}% ({successful_pairs}/{processed_pairs})")
        else:
             logger.warning(f"Skipping summary calculation for model {config.name} due to critical error.")

        # Clean up memory
        del model
        del tokenizer
        if 'loaded_object' in locals(): del loaded_object
        if 'embedding_k' in locals(): del embedding_k
        if 'embedding_e' in locals(): del embedding_e
        if 'embedding_k_float' in locals(): del embedding_k_float
        if 'embedding_e_float' in locals(): del embedding_e_float
        if 'outputs' in locals(): del outputs
        if 'last_hidden_state' in locals(): del last_hidden_state
        if 'inputs' in locals(): del inputs

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for model {config.name}.")

# --- Save Combined Summary Results (using dynamic filename) ---
try:
    with open(COMBINED_RESULTS_PATH, 'w', encoding='utf-8') as f_comb:
        json.dump(all_model_summaries, f_comb, ensure_ascii=False, indent=4)
    logger.info(f"\nSuccessfully saved combined summary results for language pair '{lang_pair}' to {COMBINED_RESULTS_PATH}")
except Exception as e:
    logger.error(f"\nError saving combined summary results to {COMBINED_RESULTS_PATH}: {e}")

logger.info("\n--- Word Embedding Similarity Evaluation Complete ---")