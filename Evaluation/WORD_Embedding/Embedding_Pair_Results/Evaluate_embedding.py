# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Literal, Tuple # Added Tuple
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
    #     model_type="encoder"
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=True, # Assuming this is local based on path pattern
    #     model_type="encoder"
    # ),
    # # --- Encoder-Decoder Model ---
    # ModelConfig(
    #     name="T5-base",
    #     model_path="t5-base",
    #     is_local=False,
    #     model_type="encoder-decoder"
    # ),
    # ModelConfig(
    #     name="T5-base-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_T5-base-Tuned",
    #     is_local=True, # Assuming this is local based on path pattern
    #     model_type="encoder-decoder"
    # ),
]

# --- Configuration ---
WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_kr.json'
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
def get_word_embedding(model, tokenizer, word, device, max_len, model_type: str) -> Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[List[str]]]:
    """
    Gets the average embedding of the last hidden state for a given word,
    excluding padding AND specific special tokens ([CLS], [SEP], [EOS]/</s>, _, <|begin_of_text|>)
    from the average. <unk> token is INCLUDED by default.
    Also returns token IDs and token strings. Handles different model types.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[List[str]]]: A tuple containing:
            - The embedding tensor (or None if error).
            - A list of token IDs (or None if error/empty).
            - A list of token strings (or None if error/empty).
    """
    if not word:
        logger.warning("Skipping empty word.")
        return None, None, None
    token_ids_list = None
    token_strings_list = None
    embedding = None
    try:
        inputs = tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(device)

        if inputs.input_ids.shape[1] == 0:
            logger.warning(f"Tokenization of '{word}' resulted in empty input_ids. Skipping.")
            return None, None, None

        token_ids_tensor = inputs['input_ids']
        token_ids_list = token_ids_tensor.squeeze(0).tolist()

        try:
            token_strings_list = tokenizer.convert_ids_to_tokens(token_ids_list)
        except Exception as te:
            logger.error(f"Failed to convert token IDs to strings for word '{word}': {te}")
            token_strings_list = ["CONVERSION_ERROR"] * len(token_ids_list)

        with torch.no_grad():
            model_inputs = {
                "input_ids": token_ids_tensor,
                "attention_mask": inputs.get('attention_mask'),
                "output_hidden_states": True
            }
            if model_inputs["attention_mask"] is None:
                # If no attention mask, create a default one assuming all tokens are valid initially
                model_inputs["attention_mask"] = torch.ones_like(token_ids_tensor)
                logger.warning(f"No attention mask provided by tokenizer for '{word}'. Creating default mask.")
                # Also update the inputs dictionary for consistency in pooling logic below
                inputs["attention_mask"] = model_inputs["attention_mask"]


            last_hidden_state = None
            # --- Conditional Model Execution ---
            try:
                if model_type == "encoder-decoder":
                    encoder_args = {k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'output_hidden_states']}
                    if "output_hidden_states" not in encoder_args:
                       encoder_args["output_hidden_states"] = True
                    outputs = model.encoder(**encoder_args)
                    if hasattr(outputs, 'last_hidden_state'):
                        last_hidden_state = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states'):
                         last_hidden_state = outputs.hidden_states[-1]
                    else:
                        logger.error(f"Could not retrieve hidden states from encoder for word '{word}'.")
                        return None, token_ids_list, token_strings_list
                elif model_type == "encoder":
                     outputs = model(**model_inputs)
                     if hasattr(outputs, 'last_hidden_state'):
                         last_hidden_state = outputs.last_hidden_state
                     elif hasattr(outputs, 'hidden_states'):
                         last_hidden_state = outputs.hidden_states[-1]
                     else:
                         logger.error(f"Could not retrieve hidden states from encoder model for word '{word}'.")
                         return None, token_ids_list, token_strings_list
                elif model_type == "causal":
                     outputs = model(**model_inputs)
                     if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                         last_hidden_state = outputs.hidden_states[-1]
                     else:
                          logger.error(f"Could not retrieve hidden states from causal model for word '{word}'.")
                          return None, token_ids_list, token_strings_list
                else:
                    logger.error(f"Unsupported model_type '{model_type}' for word '{word}'.")
                    return None, token_ids_list, token_strings_list

                if last_hidden_state is None:
                    logger.error(f"Cannot compute embedding as last_hidden_state is None after model execution for word '{word}'.")
                    return None, token_ids_list, token_strings_list

                # --- *** MODIFIED: Mean Pooling Excluding Specific Special Tokens *** ---
                attention_mask = inputs.get('attention_mask') # Get mask again (or default one)

                # 1. Start with attention_mask (handles padding)
                valid_token_mask = attention_mask.bool() # Mask for non-padding tokens (True if not pad)

                # 2. Identify special token IDs to EXCLUDE
                special_tokens_to_exclude_ids = set()
                # Add standard special tokens if they exist
                if tokenizer.cls_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.cls_token_id)
                if tokenizer.sep_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.sep_token_id)
                if tokenizer.eos_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.eos_token_id)
                # Add specific tokens by string -> ID conversion
                try:
                    # Try to get ID for T5's space marker "_" (ID 3 in T5) - be cautious
                    # t5_space_id = tokenizer.encode(" ", add_special_tokens=False) # T5 often uses ID 3 for space marker
                    # if len(t5_space_id) == 1: # Make sure it's a single ID
                    #     logger.debug(f"ID for T5 space marker ' ': {t5_space_id[0]}")
                    #     special_tokens_to_exclude_ids.add(t5_space_id[0])
                    # Let's check the actual token string for ID 3 instead, if it exists
                    if 3 in token_ids_list:
                        idx_of_3 = token_ids_list.index(3)
                        if idx_of_3 < len(token_strings_list) and token_strings_list[idx_of_3] == " ":
                             logger.debug("Found token ID 3 corresponding to ' '. Adding to exclude list for pooling.")
                             special_tokens_to_exclude_ids.add(3)

                    # Try to get ID for Llama's BOS token
                    llama_bos_token = "<|begin_of_text|>"
                    llama_bos_id = tokenizer.encode(llama_bos_token, add_special_tokens=False)
                    if len(llama_bos_id) == 1: # Make sure it's a single ID
                         logger.debug(f"ID for Llama BOS token '{llama_bos_token}': {llama_bos_id[0]}")
                         special_tokens_to_exclude_ids.add(llama_bos_id[0])
                except Exception as enc_e:
                    logger.warning(f"Could not encode specific special tokens (' ', '{llama_bos_token}') for exclusion: {enc_e}")

                logger.debug(f"Excluding special token IDs from pooling: {special_tokens_to_exclude_ids}")


                # 3. Create mask for non-special tokens (True if NOT special)
                non_special_token_mask = torch.ones_like(token_ids_tensor, dtype=torch.bool)
                if special_tokens_to_exclude_ids: # Only iterate if set is not empty
                    for special_id in special_tokens_to_exclude_ids:
                        non_special_token_mask &= (token_ids_tensor != special_id)

                # 4. Combine masks: Include = Not Padding AND Not (Excluded Special Token)
                final_pooling_mask = valid_token_mask & non_special_token_mask

                # 5. Perform Mean Pooling using the final mask
                num_valid_tokens_for_pooling = final_pooling_mask.sum().item() # Use .item() to get Python number
                if num_valid_tokens_for_pooling == 0:
                    logger.warning(f"No valid tokens left for averaging after excluding padding and special tokens for word '{word}'. IDs: {token_ids_list}, Tokens: {token_strings_list}, FinalMask: {final_pooling_mask.tolist()}. Returning None embedding.")
                    embedding = None
                else:
                    # Expand final_pooling_mask only once
                    final_mask_expanded = final_pooling_mask.unsqueeze(-1).expand(last_hidden_state.size())
                    # Use the expanded boolean mask directly for element-wise multiplication with 0 for False
                    # Need to convert mask to float for multiplication
                    masked_hidden_states = last_hidden_state * final_mask_expanded.float()
                    sum_embeddings = torch.sum(masked_hidden_states, dim=1)
                    # Divide by the actual count of True values in the boolean mask
                    sum_mask_count = torch.clamp(final_pooling_mask.sum(dim=1, keepdim=True), min=1e-9) # keepdim for broadcasting
                    embedding = sum_embeddings / sum_mask_count
                # --- *** END MODIFIED *** ---


                if embedding is not None:
                     embedding = embedding.squeeze(0)

                return embedding, token_ids_list, token_strings_list

            except Exception as model_exec_e:
                 error_type = type(model_exec_e).__name__
                 logger.error(f"Model execution or embedding error for '{word}' ({error_type}): {model_exec_e}")
                 return None, token_ids_list, token_strings_list

    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"General error processing word '{word}' ({error_type}): {e}")
        logger.error(traceback.format_exc())
        return None, None, None


# --- Main Execution ---
all_model_summaries: Dict[str, Dict[str, Any]] = {}

# --- Load Full Word Pairs ONCE ---
try:
    with open(WORD_PAIRS_PATH, 'r', encoding='utf-8') as f:
        full_word_pairs_dict = json.load(f)
    full_word_pairs_items = list(full_word_pairs_dict.items())
    logger.info(f"Loaded {len(full_word_pairs_items)} total word pairs from {WORD_PAIRS_PATH}")
except Exception as e:
    logger.error(f"Failed to load word pairs from {WORD_PAIRS_PATH}: {e}")
    exit()
# --- End Load Full Word Pairs ---

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Processing Model: {config.name} (Path: {config.model_path}) ---")
    model_individual_results_list: List[Dict[str, Any]] = []
    model = None
    tokenizer = None
    model_similarities: List[float] = []
    processed_pairs = 0
    successful_embedding_pairs = 0
    successful_similarity_calcs = 0
    error_occurred = False

    # --- Sample Selection Logic (Keep as is) ---
    num_samples_for_this_model = -1
    if config.name == "Llama-4-Scout-17B-16E": # Example, adjust if needed
        num_samples_for_this_model = 500
        logger.info(f"Specific model '{config.name}' detected. Setting NUM_SAMPLES = {num_samples_for_this_model}")
    else:
        logger.info(f"Processing all available word pairs for model '{config.name}'.")

    if num_samples_for_this_model > 0 and num_samples_for_this_model < len(full_word_pairs_items):
        word_pairs_items_to_process = full_word_pairs_items[:num_samples_for_this_model]
        total_pairs_to_process_this_model = num_samples_for_this_model
        logger.info(f"Sampling the first {num_samples_for_this_model} word pairs for processing.")
    else:
        word_pairs_items_to_process = full_word_pairs_items
        total_pairs_to_process_this_model = len(full_word_pairs_items)
        logger.info(f"Processing all {total_pairs_to_process_this_model} loaded word pairs.")
    # --- End Sample Selection ---

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
            torch_dtype=torch.bfloat16,
            local_files_only=config.is_local,
            output_hidden_states=True,
            device_map="auto"
        )
        logger.info(f"Type of loaded object from {config.model_path}: {type(loaded_object)}")
        if not isinstance(loaded_object, torch.nn.Module):
             logger.error(f"Failed to load model correctly from {config.model_path}. Received type: {type(loaded_object)}")
             raise TypeError(f"Expected a PyTorch model from {config.model_path}, but got {type(loaded_object)}")

        model = loaded_object

        # --- Pad Token ID Configuration (Keep as is) ---
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
        # --- End Pad Token ID Configuration ---

        model.eval()
        logger.info(f"Model from {config.model_path} evaluation mode set.")

        # --- Process selected pairs for this model ---
        for english_word_key, korean_word_value in tqdm(word_pairs_items_to_process, desc=f"Processing pairs for {config.name}", total=total_pairs_to_process_this_model):
            processed_pairs += 1

            # --- *** MODIFIED: Get embedding, IDs, AND tokens *** ---
            embedding_k, token_ids_k, tokens_k = get_word_embedding(model, tokenizer, korean_word_value, DEVICE, SAFE_MAX_LENGTH, config.model_type)
            embedding_e, token_ids_e, tokens_e = get_word_embedding(model, tokenizer, english_word_key, DEVICE, SAFE_MAX_LENGTH, config.model_type)
            # --- *** END MODIFIED *** ---

            similarity = None
            embeddings_successful = embedding_k is not None and embedding_e is not None

            if embeddings_successful:
                successful_embedding_pairs += 1
                try:
                    embedding_k_float = embedding_k.float()
                    embedding_e_float = embedding_e.float()
                    similarity = F.cosine_similarity(embedding_k_float.unsqueeze(0), embedding_e_float.unsqueeze(0)).item()
                    model_similarities.append(similarity)
                    successful_similarity_calcs += 1
                except Exception as e:
                    logger.error(f"Similarity calculation error for '{korean_word_value}' & '{english_word_key}': {e}")
                    similarity = None
            else:
                logger.warning(f"Skipping similarity calculation for pair ('{korean_word_value}', '{english_word_key}') due to embedding error.")
                similarity = None

            # --- *** MODIFIED: Store result including token strings *** ---
            result_entry = {
                "english_word": english_word_key,
                "english_tokens": tokens_e,          # Add english tokens
                "english_token_ids": token_ids_e,
                "korean_word": korean_word_value,
                "korean_tokens": tokens_k,           # Add korean tokens
                "korean_token_ids": token_ids_k,
                "similarity": similarity
            }
            model_individual_results_list.append(result_entry)
            # --- *** END MODIFIED *** ---

        # --- Save Individual Results ---
        individual_filename = os.path.join(INDIVIDUAL_RESULTS_FOLDER, f"{config.name}_similarity_{lang_pair}.json")
        try:
            with open(individual_filename, 'w', encoding='utf-8') as f_ind:
                json.dump(model_individual_results_list, f_ind, ensure_ascii=False, indent=4)
            logger.info(f"Saved individual results for model {config.name} to {individual_filename}")
        except Exception as e:
            logger.error(f"Failed to save individual results for model {config.name}: {e}")

    except Exception as e: # Catch critical errors during setup or loop
        logger.error(f"Critical error processing model {config.name} (Path: {config.model_path}): {e}")
        logger.error(traceback.format_exc())
        error_occurred = True
        # Ensure summary captures the error state correctly
        all_model_summaries[config.name] = {
            "error": str(e),
            "model_path": config.model_path,
            "lang_pair": lang_pair,
            "processed_pairs": processed_pairs,
            "total_pairs_in_file": len(full_word_pairs_items),
            "successful_embedding_pairs": successful_embedding_pairs,
            "successful_similarity_calcs": successful_similarity_calcs
        }

    finally:
        # --- Summary Calculation (Keep as is) ---
        if not error_occurred:
            avg_similarity = np.mean(model_similarities) if model_similarities else None
            std_dev_similarity = np.std(model_similarities) if len(model_similarities) > 1 else 0.0
            median_similarity = statistics.median(model_similarities) if model_similarities else None
            sim_success_rate = (successful_similarity_calcs / processed_pairs) * 100 if processed_pairs > 0 else 0.0

            all_model_summaries[config.name] = {
                "model_path": config.model_path,
                "lang_pair": lang_pair,
                "processed_pairs": processed_pairs,
                "total_pairs_in_file": len(full_word_pairs_items),
                "successful_embedding_pairs": successful_embedding_pairs,
                "successful_similarity_calcs": successful_similarity_calcs,
                "similarity_success_rate_percent": round(sim_success_rate, 2),
                "average_similarity": round(float(avg_similarity), 6) if avg_similarity is not None else None,
                "median_similarity": round(float(median_similarity), 6) if median_similarity is not None else None,
                "std_dev_similarity": round(float(std_dev_similarity), 6)
            }
            avg_sim_str = f"{avg_similarity:.4f}" if avg_similarity is not None else "N/A"
            med_sim_str = f"{median_similarity:.4f}" if median_similarity is not None else "N/A"
            logger.info(f"Summary for model {config.name}: Avg Sim={avg_sim_str}, Median Sim={med_sim_str}, StdDev={std_dev_similarity:.4f}, Sim Success Rate={sim_success_rate:.2f}% ({successful_similarity_calcs}/{processed_pairs})")
        elif config.name in all_model_summaries:
             logger.warning(f"Summary calculation skipped for model {config.name} due to critical error during processing. Error details saved.")
        else:
            logger.error(f"Model {config.name} encountered an error, but no error summary was recorded.")
        # --- End Summary Calculation ---

        # --- Memory Cleanup ---
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
        if 'token_ids_k' in locals(): del token_ids_k
        if 'token_ids_e' in locals(): del token_ids_e
        # *** MODIFIED: Clean up token string variables ***
        if 'tokens_k' in locals(): del tokens_k
        if 'tokens_e' in locals(): del tokens_e
        # *** END MODIFIED ***
        if 'model_individual_results_list' in locals(): del model_individual_results_list

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for model {config.name}.")

# --- Save Combined Summary Results ---
try:
    with open(COMBINED_RESULTS_PATH, 'w', encoding='utf-8') as f_comb:
        json.dump(all_model_summaries, f_comb, ensure_ascii=False, indent=4)
    logger.info(f"\nSuccessfully saved combined summary results for language pair '{lang_pair}' to {COMBINED_RESULTS_PATH}")
except Exception as e:
    logger.error(f"\nError saving combined summary results to {COMBINED_RESULTS_PATH}: {e}")

logger.info("\n--- Word Embedding Similarity Evaluation Complete ---")