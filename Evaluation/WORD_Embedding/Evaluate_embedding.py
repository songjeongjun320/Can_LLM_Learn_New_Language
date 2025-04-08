# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass # <--- Import dataclass
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
    # ModelConfig(
    #     name="OLMo-1b-org",
    #     model_path="allenai/OLMo-1B",
    #     is_local=False,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="OLMo-1b-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B",
    #     is_local=True,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="OLMo-7b-org",
    #     model_path="allenai/OLMo-7B",
    #     is_local=False,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="OLMo-7b-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B",
    #     is_local=True,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="Llama-3.2-3b",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b",
    #     is_local=True,
    #     model_type="causal"
    # ),
    ModelConfig(
        name="Llama-4-Scout-17B-16E",
        model_path="meta-llama/Llama-4-Scout-17B-16E",
        is_local=False,
        model_type="causal"
    ),
    # --- Encoder Model ---
    # ModelConfig(
    #     name="BERT-base-uncased",
    #     model_path="bert-base-uncased",
    #     is_local=False,
    #     model_type="encoder" # <-- Specify type
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=False,
    #     model_type="encoder" # <-- Specify type
    # ),

    # # --- Encoder-Decoder Model ---
    # ModelConfig(
    #     name="T5-base",
    #     model_path="t5-base",
    #     is_local=False,
    #     model_type="encoder-decoder" # <-- Specify type
    # ),
    # ModelConfig(
    #     name="T5-base-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_T5-base-Tuned",
    #     is_local=False,
    #     model_type="encoder-decoder" # <-- Specify type
    # ),
]

# --- Configuration ---
WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/ch_kr.json'
INDIVIDUAL_RESULTS_FOLDER = 'embedding_results_ch_kr'
COMBINED_RESULTS_PATH = 'llama4_results_ch_kr.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAFE_MAX_LENGTH = 512  # Define a reasonable max length for tokenization
NUM_SAMPLES = 500

os.makedirs(INDIVIDUAL_RESULTS_FOLDER, exist_ok=True)
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

            # --- *** Conditional Model Execution *** ---
            if model_type == "encoder-decoder":
                # For T5 (and potentially other enc-dec models loaded with AutoModel),
                # call ONLY the encoder part.
                # We pass the regular 'input_ids' and 'attention_mask' intended for the encoder.
                outputs = model.encoder(**{k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'output_hidden_states']}) # Pass relevant args
                # The encoder output usually directly contains 'last_hidden_state'
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
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None and last_hidden_state is not None: # Check last_hidden_state is not None
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9) # Avoid division by zero
                embedding = sum_embeddings / sum_mask
            elif last_hidden_state is not None: # Check last_hidden_state is not None
                 # If no attention mask, just average all tokens (less ideal for padded sequences)
                 logger.warning(f"No attention mask found for word '{word}'. Averaging all tokens.")
                 embedding = last_hidden_state.mean(dim=1)
            else:
                 # This case should ideally not be reached if checks above work
                 logger.error(f"Cannot compute embedding as last_hidden_state is None for word '{word}'.")
                 return None


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

try:
    with open(WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
        full_word_pairs = json.load(f) # Load the full dictionary
    logger.info(f"Loaded {len(full_word_pairs)} total word pairs from {WORD_PAIRS_PATH}")

    # *** Sample the first NUM_SAMPLES items ***
    if NUM_SAMPLES > 0 and NUM_SAMPLES < len(full_word_pairs):
        word_pairs_items = list(full_word_pairs.items())[:NUM_SAMPLES]
        # Convert back to dictionary if needed, or keep as list of tuples for iteration
        # word_pairs = dict(word_pairs_items) # Optional: convert back to dict
        logger.info(f"Sampling the first {NUM_SAMPLES} word pairs for processing.")
        total_pairs_to_process = NUM_SAMPLES
    else:
        # If NUM_SAMPLES is 0 or >= total, process all pairs
        word_pairs_items = list(full_word_pairs.items()) # Use all pairs
        # word_pairs = full_word_pairs # Use the full dictionary
        logger.info("Processing all loaded word pairs.")
        total_pairs_to_process = len(full_word_pairs)
except Exception as e:
    logger.error(f"Failed to load word pairs: {e}")
    exit()

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Processing Model Path: {config.model_path} (Name: {config.name}) ---")
    model_individual_results: Dict[str, Dict[str, Optional[float]]] = {}
    model = None
    tokenizer = None
    model_similarities: List[float] = []
    processed_pairs = 0
    successful_pairs = 0
    error_occurred = False

    try:
        logger.info(f"Loading tokenizer from: {config.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=config.is_local
        )
        # Log the potentially problematic value before fixing it
        original_max_len = getattr(tokenizer, 'model_max_length', 'N/A')
        logger.info(f"Tokenizer original model_max_length: {original_max_len}")

        if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set EOS token as PAD token.")
        elif tokenizer.pad_token is None:
             logger.warning(f"Tokenizer for {config.name} does not have a pad_token or eos_token. Padding might behave unexpectedly.")

        # --- Select appropriate AutoModel class ---
        logger.info(f"Loading model from: {config.model_path} using appropriate AutoModel class (float32)...")
        if config.model_type == "causal":
            ModelClass = AutoModelForCausalLM
        elif config.model_type == "encoder" or config.model_type == "encoder-decoder":
             # Use AutoModel to get base embeddings for both encoder and enc-dec
             ModelClass = AutoModel
        else:
             logger.error(f"Unknown model_type '{config.model_type}' for {config.name}. Skipping.")
             raise ValueError(f"Unknown model_type: {config.model_type}")

        loaded_object = ModelClass.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            # torch_dtype=torch.float32, # Default is float32
            torch_dtype=torch.bfloat16,
            local_files_only=config.is_local,
            output_hidden_states=True, # Ensure hidden states are outputted by default if possible
            device_map="auto" 
        )

        logger.info(f"Type of loaded object from {config.model_path}: {type(loaded_object)}")
        if not isinstance(loaded_object, torch.nn.Module):
             logger.error(f"Failed to load model correctly from {config.model_path}. Received type: {type(loaded_object)}")
             raise TypeError(f"Expected a PyTorch model from {config.model_path}, but got {type(loaded_object)}")

        # model = loaded_object.to(DEVICE)
        model = loaded_object
        # logger.info(f"Model from {config.model_path} successfully moved to {DEVICE}.")

        # Set pad_token_id in model config if tokenizer has pad_token_id
        if tokenizer.pad_token_id is not None:
            if hasattr(model, 'config') and model.config is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}")
            else:
                logger.warning(f"Could not set pad_token_id on model config for {config.name}.")
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            # Fallback to EOS if PAD is not set but EOS is
             if hasattr(model, 'config') and model.config is not None:
                logger.warning(f"Attempting to set model.config.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id}) as pad_token_id was None.")
                model.config.pad_token_id = tokenizer.eos_token_id
             else:
                logger.warning(f"Could not set pad_token_id using eos_token_id on model config for {config.name}.")

        model.eval()
        logger.info(f"Model from {config.model_path} evaluation mode set.")

        # Process pairs using the SAFE_MAX_LENGTH
        for korean_word, english_word in tqdm(word_pairs_items, desc=f"Processing pairs for {config.name}", total=total_pairs_to_process):
            processed_pairs += 1

            korean_word_lower = korean_word.lower()
            english_word_lower = english_word.lower()
            # Pass SAFE_MAX_LENGTH to the embedding function
            embedding_k = get_word_embedding(model, tokenizer, korean_word, DEVICE, SAFE_MAX_LENGTH, config.model_type)
            embedding_e = get_word_embedding(model, tokenizer, english_word, DEVICE, SAFE_MAX_LENGTH, config.model_type)

            similarity = None
            if embedding_k is not None and embedding_e is not None:
                try:
                    # Ensure embeddings are on the correct device and have compatible shapes
                    # unsqueeze(0) adds a batch dimension for cosine_similarity
                    similarity = F.cosine_similarity(embedding_k.unsqueeze(0), embedding_e.unsqueeze(0)).item()
                    model_similarities.append(similarity)
                    successful_pairs += 1
                except Exception as e:
                    logger.error(f"Similarity calculation error for '{korean_word}' & '{english_word}': {e}")
                    similarity = None # Ensure similarity is None if calculation fails
            else:
                logger.warning(f"Skipping similarity calculation for pair ('{korean_word}', '{english_word}') due to embedding error.")
                # Ensure similarity is None if embeddings couldn't be generated
                similarity = None

            # Store result for this pair, even if similarity calculation failed (similarity will be None)
            model_individual_results[korean_word] = {"english_word": english_word, "similarity": similarity}

        # Save Individual Results
        individual_filename = os.path.join(INDIVIDUAL_RESULTS_FOLDER, f"{config.name}_similarity.json")

        try:
            with open(individual_filename, 'w', encoding='utf-8') as f_ind:
                json.dump(model_individual_results, f_ind, ensure_ascii=False, indent=4)
            logger.info(f"Saved individual results for model path {config.model_path} (name: {config.name}) to {individual_filename}")
        except Exception as e:
            logger.error(f"Failed to save individual results for model path {config.model_path} (name: {config.name}): {e}")

    except Exception as e:
        logger.error(f"Critical error processing model path {config.model_path} (name: {config.name}): {e}")
        logger.error("--- Critical Error Traceback ---")
        logger.error(traceback.format_exc())
        logger.error("--- End Critical Error Traceback ---")
        error_occurred = True
        all_model_summaries[config.name] = {"error": str(e), "model_path": config.model_path}

    finally:
        # Calculate Summary Metrics
        if not error_occurred:
            avg_similarity = np.mean(model_similarities) if model_similarities else 0.0
            std_dev_similarity = np.std(model_similarities) if len(model_similarities) > 1 else 0.0
            median_similarity = statistics.median(model_similarities) if model_similarities else 0.0
            success_rate = (successful_pairs / processed_pairs) * 100 if processed_pairs > 0 else 0.0
            all_model_summaries[config.name] = {
                "model_path": config.model_path,
                "processed_pairs": processed_pairs,
                "successful_pairs": successful_pairs,
                "success_rate_percent": round(success_rate, 2),
                "average_similarity": round(float(avg_similarity), 6),
                "median_similarity": round(float(median_similarity), 6),
                "std_dev_similarity": round(float(std_dev_similarity), 6)
            }
            logger.info(f"Summary for model path {config.model_path} (name: {config.name}): Avg Sim={avg_similarity:.4f}, Median Sim={median_similarity:.4f}, StdDev={std_dev_similarity:.4f}, Success Rate={success_rate:.2f}%")
        else:
             logger.warning(f"Skipping summary calculation for model path {config.model_path} (name: {config.name}) due to critical error.")

        # Clean up memory
        del model
        del tokenizer
        if 'loaded_object' in locals(): del loaded_object
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for model path {config.model_path} (name: {config.name}).")

# Save Combined Summary Results
try:
    with open(COMBINED_RESULTS_PATH, 'w', encoding='utf-8') as f_comb:
        json.dump(all_model_summaries, f_comb, ensure_ascii=False, indent=4)
    logger.info(f"\nSuccessfully saved combined summary results to {COMBINED_RESULTS_PATH}")
except Exception as e:
    logger.error(f"\nError saving combined summary results: {e}")

logger.info("\n--- Word Embedding Similarity Evaluation Complete ---")