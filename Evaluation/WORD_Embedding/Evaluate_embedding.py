# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass # <--- Import dataclass
from typing import List, Dict, Optional, Any
import statistics
import traceback

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# --- Updated MODEL_CONFIGS list ---
MODEL_CONFIGS = [
    ModelConfig(
        name="OLMo-1b-org",
        model_path="allenai/OLMo-1B",
        is_local=False
    ),
    ModelConfig(
        name="OLMo-1b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B",
        is_local=True
    ),
    ModelConfig(
        name="OLMo-7b-org",
        model_path="allenai/OLMo-7B",
        is_local=False
    ),
    ModelConfig(
        name="OLMo-7b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B",
        is_local=True
    ),
    ModelConfig(
        name="Llama-3.2-3b",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b",
        is_local=True
    )
]

# --- Configuration ---
WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/translated_words.json'
INDIVIDUAL_RESULTS_FOLDER = 'embedding_results'
COMBINED_RESULTS_PATH = 'combined_word_embedding_results.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAFE_MAX_LENGTH = 2048 # Define a reasonable max length for tokenization
NUM_SAMPLES = 5282

os.makedirs(INDIVIDUAL_RESULTS_FOLDER, exist_ok=True)
logger.info(f"Using device: {DEVICE}")

# --- Helper Function to Get Word Embedding (with explicit max_length) ---
def get_word_embedding(model, tokenizer, word, device, max_len):
    """
    Gets the average embedding of the last hidden state for a given word.
    Uses an explicit max_length and includes detailed traceback logging.
    """
    if not word:
        logger.warning("Skipping empty word.")
        return None
    try:
        # Use the provided explicit max_len instead of tokenizer.model_max_length
        inputs = tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
            max_length=max_len # <--- Use explicit max_len
        ).to(device)

        if inputs.input_ids.shape[1] == 0:
            logger.warning(f"Tokenization of '{word}' resulted in empty input_ids. Skipping.")
            return None
        # No need to check against tokenizer.model_max_length here

        with torch.no_grad():
            # *** Explicitly pass input_ids and attention_mask ***
            # Check if attention_mask exists in inputs, pass if it does
            model_inputs = {
                "input_ids": inputs['input_ids'],
                "output_hidden_states": True
            }
            if 'attention_mask' in inputs:
                 model_inputs['attention_mask'] = inputs['attention_mask']

            outputs = model(**model_inputs) # <--- Pass the curated dictionary

        if not outputs.hidden_states:
             logger.error(f"Could not retrieve hidden states for word '{word}'.")
             return None
        last_hidden_state = outputs.hidden_states[-1]

        if 'attention_mask' in inputs:
            mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        else:
             embedding = last_hidden_state.mean(dim=1)

        return embedding.squeeze(0)

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

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Added EOS token as PAD token.")

        logger.info(f"Loading model from: {config.model_path} using AutoModelForCausalLM (using float32)...")
        loaded_object = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16, # Keep using float32
            local_files_only=config.is_local
        )

        logger.info(f"Type of loaded object from {config.model_path}: {type(loaded_object)}")
        if not isinstance(loaded_object, torch.nn.Module):
             logger.error(f"Failed to load model correctly from {config.model_path}. Received type: {type(loaded_object)}")
             raise TypeError(f"Expected a PyTorch model from {config.model_path}, but got {type(loaded_object)}")

        logger.info(f"Moving model from {config.model_path} to device: {DEVICE}...")
        model = loaded_object.to(DEVICE)
        logger.info(f"Model from {config.model_path} successfully moved to {DEVICE}.")

        if tokenizer.pad_token_id is None and hasattr(tokenizer, 'eos_token_id'):
             if hasattr(model, 'config') and model.config is not None: model.config.pad_token_id = tokenizer.eos_token_id
             else: logger.warning(f"Could not set pad_token_id on model config for {config.model_path}.")
        model.eval()
        logger.info(f"Model from {config.model_path} evaluation mode set.")

        # Process pairs using the SAFE_MAX_LENGTH
        for korean_word, english_word in tqdm(word_pairs_items, desc=f"Processing pairs for {config.name}", total=total_pairs_to_process):
            processed_pairs += 1

            korean_word_lower = korean_word.lower()
            english_word_lower = english_word.lower()
            # Pass SAFE_MAX_LENGTH to the embedding function
            embedding_k = get_word_embedding(model, tokenizer, korean_word, DEVICE, SAFE_MAX_LENGTH)
            embedding_e = get_word_embedding(model, tokenizer, english_word, DEVICE, SAFE_MAX_LENGTH)

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