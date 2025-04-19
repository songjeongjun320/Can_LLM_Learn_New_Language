# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Literal, Tuple
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
    # ModelConfig(
    #     name="Llama-4-Scout-17B-16E",
    #     model_path="meta-llama/Llama-4-Scout-17B-16E",
    #     is_local=False,
    #     model_type="causal"
    # ),

    # --- Encoder Model ---
    # ModelConfig(
    #      name="BERT-base-uncased",
    #      model_path="bert-base-uncased",
    #      is_local=False,
    #      model_type="encoder"
    #  ),
    # ModelConfig(
    #      name="BERT-base-uncased",
    #      model_path="bert-base-uncased",
    #      is_local=False,
    #      model_type="encoder"
    #  ),
    # ModelConfig(
    #     name="bert-uncased-finetuned-kr-eng-v2",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-kr-eng-v2",
    #     is_local=True, # Assuming this is local based on path pattern
    #     model_type="encoder"
    # ),
    
    # ModelConfig(
    #     name="bert-base-uncased-CAUSAL-LM-Tuning",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/BERT-base-uncased-CAUSAL-LM-Tuning",
    #     is_local=True, # Assuming this is local based on path pattern
    #     model_type="encoder"
    # ),
    # ModelConfig(
    #     name="bert-uncased-finetuned-subtitle_dt",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt",
    #     is_local=True, # Assuming this is local based on path pattern
    #     model_type="encoder"
    # ),
    ModelConfig(
        name="bert-uncased-finetuned-subtitle_dt-used-reverse-v2",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt_v2",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
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
    # ModelConfig(
    #     name="Llama-3.2-3b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
    #     is_local=True,
    #     model_type="causal"
    # ),
    # ModelConfig(
    #     name="Llama-3.1-8b-it",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
    #     is_local=True,
    #     model_type="causal"
    # ),
]

# --- Configuration ---
# WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_ch.json'
WORD_PAIRS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_kr.json' # Using KR for testing T5 again
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAFE_MAX_LENGTH = 512

# --- *** Dynamic Path Generation *** ---
try:
    word_pairs_basename = os.path.basename(WORD_PAIRS_PATH)
    lang_pair, _ = os.path.splitext(word_pairs_basename)
    if not lang_pair:
        raise ValueError("Could not extract language pair from WORD_PAIRS_PATH filename.")

    # Indicate Raw Token Embedding + Last Hidden calculation
    INDIVIDUAL_RESULTS_FOLDER = f'embedding_results_{lang_pair}_RAW_LAST'
    COMBINED_RESULTS_PATH = f'combined_embedding_results_{lang_pair}_RAW_LAST.json'

    os.makedirs(INDIVIDUAL_RESULTS_FOLDER, exist_ok=True)
    logger.info(f"Using language pair: {lang_pair}")
    logger.info(f"Individual results will be saved in: {INDIVIDUAL_RESULTS_FOLDER}")
    logger.info(f"Combined results will be saved to: {COMBINED_RESULTS_PATH}")

except Exception as e:
    logger.error(f"Failed to determine dynamic paths from WORD_PAIRS_PATH '{WORD_PAIRS_PATH}': {e}")
    logger.error("Please ensure WORD_PAIRS_PATH has a filename like 'lang1_lang2.json'.")
    exit()

logger.info(f"Using device: {DEVICE}")

# --- *** NEW Helper Function to Get RAW Token Embeddings and LAST Hidden State Embeddings *** ---
def get_raw_and_last_embeddings(model, tokenizer, word, device, max_len, model_type: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]], Optional[List[str]]]:
    """
    *** NEW: Gets the average embedding from BOTH the RAW TOKEN EMBEDDING LAYER (lookup before pos encoding)
    AND the LAST HIDDEN STATE for a given word. ***
    Excludes padding AND specific special tokens from the average.
    Also returns token IDs and token strings. Handles different model types.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]], Optional[List[str]]]: A tuple containing:
            - The RAW token embedding tensor (mean pooled lookup from embedding matrix) (or None if error).
            - The LAST hidden state tensor (mean pooled hidden_states[-1]) (or None if error).
            - A list of token IDs (or None if error/empty).
            - A list of token strings (or None if error/empty).
    """
    if not word:
        logger.warning("Skipping empty word.")
        return None, None, None, None
    token_ids_list = None
    token_strings_list = None
    embedding_raw = None # Changed name
    embedding_last = None
    try:
        inputs = tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(device)

        if inputs.input_ids.shape[1] == 0:
            logger.warning(f"Tokenization of '{word}' resulted in empty input_ids. Skipping.")
            return None, None, None, None

        token_ids_tensor = inputs['input_ids']
        token_ids_list = token_ids_tensor.squeeze(0).tolist()

        try:
            token_strings_list = tokenizer.convert_ids_to_tokens(token_ids_list)
        except Exception as te:
            logger.error(f"Failed to convert token IDs to strings for word '{word}': {te}")
            token_strings_list = ["CONVERSION_ERROR"] * len(token_ids_list)

        with torch.no_grad():
            # --- Get RAW Token Embeddings (Before Positional Encoding) ---
            raw_embeddings_tensor = None
            try:
                embedding_layer = model.get_input_embeddings()
                if embedding_layer is not None:
                    # Directly look up token IDs in the embedding matrix
                    raw_embeddings_tensor = embedding_layer(token_ids_tensor)
                    logger.debug(f"Extracted raw token embeddings shape: {raw_embeddings_tensor.shape}")
                else:
                    logger.error(f"Could not get input embedding layer for word '{word}' using model.get_input_embeddings().")
                    # Cannot proceed without raw embeddings if requested this way
                    # We might still be able to get last hidden state, let's try
            except Exception as emb_e:
                logger.error(f"Error getting raw token embeddings for word '{word}': {emb_e}")
                # Continue to try getting last hidden state


            # --- Get Model Outputs for Last Hidden State ---
            # We still need to run the model forward pass to get the last hidden state
            model_inputs = {
                "input_ids": token_ids_tensor,
                "attention_mask": inputs.get('attention_mask'),
                "output_hidden_states": True # Still need this for last hidden state
            }
            if model_inputs["attention_mask"] is None:
                model_inputs["attention_mask"] = torch.ones_like(token_ids_tensor)
                logger.warning(f"No attention mask provided by tokenizer for '{word}'. Creating default mask.")
                inputs["attention_mask"] = model_inputs["attention_mask"] # Keep attention mask consistent

            # --- Get it from Hidden state layers
            hidden_state_last = None
            try:
                outputs = None
                if model_type == "encoder-decoder":
                    # For T5, we get encoder outputs
                    encoder_args = {k: v for k, v in model_inputs.items() if k in ['input_ids', 'attention_mask', 'output_hidden_states']}
                    if "output_hidden_states" not in encoder_args:
                         encoder_args["output_hidden_states"] = True
                    # Ensure we are calling the encoder directly if using AutoModel for T5
                    if hasattr(model, 'encoder'):
                        outputs = model.encoder(**encoder_args)
                    else: # Should not happen if ModelClass=AutoModel for T5
                         logger.error("Model object for T5 does not have 'encoder' attribute.")
                         outputs = model(**encoder_args) # Fallback attempt
                elif model_type == "encoder":
                    outputs = model(**model_inputs)
                elif model_type == "causal":
                    outputs = model(**model_inputs)
                else:
                    logger.error(f"Unsupported model_type '{model_type}' for word '{word}'.")
                    # Return what we have so far (potentially raw embedding)
                    return embedding_raw, None, token_ids_list, token_strings_list # embedding_last is None here

                # --- Extract Last Hidden State ---
                if outputs is None:
                    logger.error(f"Model outputs are None for word '{word}' (last hidden state).")
                else:
                    all_hidden_states = None
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        all_hidden_states = outputs.hidden_states
                        logger.debug(f"Found {len(all_hidden_states)} hidden states for word '{word}' (for last state).")
                        if len(all_hidden_states) > 0:
                            hidden_state_last = all_hidden_states[-1]
                            logger.debug(f"Extracted hidden_states[-1] (LastHidden) shape: {hidden_state_last.shape}")
                        else:
                             logger.error(f"outputs.hidden_states is empty for word '{word}'.")
                    elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                         # Handle cases where only last_hidden_state is returned (e.g., AutoModel for BERT)
                         hidden_state_last = outputs.last_hidden_state
                         logger.debug(f"Extracted last_hidden_state shape: {hidden_state_last.shape}")
                    else:
                         logger.error(f"Could not retrieve hidden_states or last_hidden_state for word '{word}'.")

            except Exception as model_exec_e:
                error_type = type(model_exec_e).__name__
                logger.error(f"Model execution error for '{word}' ({error_type}) when getting last hidden state: {model_exec_e}")
                # Continue to pooling with whatever we have

            # --- Mean Pooling Logic (Applied to both raw and last if available) ---
            attention_mask = inputs.get('attention_mask')
            valid_token_mask = attention_mask.bool()

            # Calculate mask once (same logic as before)
            special_tokens_to_exclude_ids = set()
            if tokenizer.cls_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.cls_token_id)
            if tokenizer.sep_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.sep_token_id)
            if tokenizer.eos_token_id is not None: special_tokens_to_exclude_ids.add(tokenizer.eos_token_id)
            # Add specific checks for space tokens or BOS tokens if necessary
            try:
                # Special Tokens
                if 3 in token_ids_list: # Example for T5 _ token ID
                    idx_of_3 = token_ids_list.index(3)
                    if idx_of_3 < len(token_strings_list) and token_strings_list[idx_of_3] == " ":
                         special_tokens_to_exclude_ids.add(3)
                llama_bos_token = "<|begin_of_text|>" # Example for Llama BOS
                llama_bos_id = tokenizer.encode(llama_bos_token, add_special_tokens=False)
                if len(llama_bos_id) == 1:
                     special_tokens_to_exclude_ids.add(llama_bos_id[0])
            except Exception as enc_e:
                 logger.warning(f"Could not encode specific special tokens (' ', '{llama_bos_token}') for exclusion: {enc_e}")


            non_special_token_mask = torch.ones_like(token_ids_tensor, dtype=torch.bool)
            if special_tokens_to_exclude_ids:
                for special_id in special_tokens_to_exclude_ids:
                    non_special_token_mask &= (token_ids_tensor != special_id)

            final_pooling_mask = valid_token_mask & non_special_token_mask
            num_valid_tokens_for_pooling = final_pooling_mask.sum().item()

            if num_valid_tokens_for_pooling == 0:
                logger.warning(f"No valid tokens left for averaging for word '{word}'. IDs: {token_ids_list}, Tokens: {token_strings_list}. Returning None embeddings.")
                # Return None for both raw and last
                return None, None, token_ids_list, token_strings_list
            else:
                final_mask_expanded_base = final_pooling_mask.unsqueeze(-1) # Expand once
                sum_mask_count = torch.clamp(final_pooling_mask.sum(dim=1, keepdim=True), min=1e-9)

                # Pool raw_embeddings_tensor
                if raw_embeddings_tensor is not None:
                    final_mask_expanded = final_mask_expanded_base.expand(raw_embeddings_tensor.size())
                    masked_states = raw_embeddings_tensor * final_mask_expanded.float()
                    sum_embeddings = torch.sum(masked_states, dim=1)
                    embedding_raw = (sum_embeddings / sum_mask_count).squeeze(0)
                    logger.debug(f"Calculated raw token embedding for '{word}'.")
                else:
                    logger.warning(f"Could not calculate raw token embedding for '{word}' as raw_embeddings_tensor was None.")

                # Pool hidden_state_last
                if hidden_state_last is not None:
                    final_mask_expanded = final_mask_expanded_base.expand(hidden_state_last.size())
                    masked_states = hidden_state_last * final_mask_expanded.float()
                    sum_embeddings = torch.sum(masked_states, dim=1)
                    embedding_last = (sum_embeddings / sum_mask_count).squeeze(0)
                    logger.debug(f"Calculated last hidden state embedding for '{word}'.")
                else:
                    logger.warning(f"Could not calculate last hidden state embedding for '{word}' as hidden_state_last was None.")

            return embedding_raw, embedding_last, token_ids_list, token_strings_list

    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"General error processing word '{word}' ({error_type}): {e}")
        logger.error(traceback.format_exc())
        # Return None for both embeddings in case of general error
        return None, None, None, None


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

for config in MODEL_CONFIGS:
    logger.info(f"\n--- Processing Model: {config.name} (Path: {config.model_path}) ---")
    model_individual_results_list: List[Dict[str, Any]] = []
    model = None
    tokenizer = None
    # --- Store similarities separately ---
    model_similarities_raw: List[float] = [] # Renamed
    model_similarities_last: List[float] = []
    processed_pairs = 0
    successful_pairs_raw_emb = 0      # Renamed
    successful_pairs_last_hidden = 0
    successful_similarity_calcs_raw = 0 # Renamed
    successful_similarity_calcs_last = 0
    error_occurred = False

    # --- Sample Selection Logic ---
    num_samples_for_this_model = -1
    # if config.name == "Llama-4-Scout-17B-16E": num_samples_for_this_model = 500
    if num_samples_for_this_model > 0 and num_samples_for_this_model < len(full_word_pairs_items):
        word_pairs_items_to_process = full_word_pairs_items[:num_samples_for_this_model]
        total_pairs_to_process_this_model = num_samples_for_this_model
        logger.info(f"Sampling the first {num_samples_for_this_model} word pairs.")
    else:
        word_pairs_items_to_process = full_word_pairs_items
        total_pairs_to_process_this_model = len(full_word_pairs_items)
        logger.info(f"Processing all {total_pairs_to_process_this_model} loaded word pairs.")

    try:
        # --- Model and Tokenizer Loading ---
        logger.info(f"Loading tokenizer from: {config.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True, local_files_only=config.is_local
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token; logger.info("Set EOS token as PAD token.")

        logger.info(f"Loading model from: {config.model_path} (bfloat16)...")
        # For T5 (encoder-decoder), AutoModel gives access to the encoder.
        # For BERT (encoder), AutoModel gives the base model outputs.
        # For CausalLM, AutoModelForCausalLM gives access to hidden states.
        if config.model_type == "causal": ModelClass = AutoModelForCausalLM
        elif config.model_type in ["encoder", "encoder-decoder"]: ModelClass = AutoModel
        else: raise ValueError(f"Unknown model_type: {config.model_type}")

        model = ModelClass.from_pretrained(
            config.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            local_files_only=config.is_local,
            output_hidden_states=True, # Still needed for last hidden state
            device_map="auto"
        )
        # Configure pad token ID in the model config if possible
        if hasattr(model, 'config') and model.config is not None:
             if tokenizer.pad_token_id is not None:
                 model.config.pad_token_id = tokenizer.pad_token_id
             elif tokenizer.eos_token_id is not None:
                 model.config.pad_token_id = tokenizer.eos_token_id


        model.eval()
        logger.info(f"Model {config.name} loaded and in eval mode.")


        # --- Process selected pairs for this model ---
        for english_word_key, foreign_word_value in tqdm(word_pairs_items_to_process, desc=f"Processing pairs for {config.name}", total=total_pairs_to_process_this_model):
            processed_pairs += 1

            # --- *** Get RAW and LAST embeddings *** ---
            emb_f_raw, emb_f_last, ids_f, tok_f = get_raw_and_last_embeddings(model, tokenizer, foreign_word_value, DEVICE, SAFE_MAX_LENGTH, config.model_type)
            emb_e_raw, emb_e_last, ids_e, tok_e = get_raw_and_last_embeddings(model, tokenizer, english_word_key, DEVICE, SAFE_MAX_LENGTH, config.model_type)

            similarity_raw = None  # Renamed
            similarity_last = None

            # --- Calculate Similarity for RAW Token Embeddings ---
            if emb_f_raw is not None and emb_e_raw is not None:
                successful_pairs_raw_emb += 1
                try:
                    similarity_raw = F.cosine_similarity(emb_f_raw.float().unsqueeze(0), emb_e_raw.float().unsqueeze(0)).item()
                    model_similarities_raw.append(similarity_raw)
                    successful_similarity_calcs_raw += 1
                except Exception as e:
                    logger.error(f"RawEmb Sim calculation error for '{foreign_word_value}' & '{english_word_key}': {e}")
                    similarity_raw = None # Ensure it's None on error
            else:
                logger.warning(f"Skipping RawEmb similarity for pair ('{foreign_word_value}', '{english_word_key}') due to embedding error.")

            # --- Calculate Similarity for Last Hidden States ---
            if emb_f_last is not None and emb_e_last is not None:
                successful_pairs_last_hidden += 1
                try:
                    similarity_last = F.cosine_similarity(emb_f_last.float().unsqueeze(0), emb_e_last.float().unsqueeze(0)).item()
                    model_similarities_last.append(similarity_last)
                    successful_similarity_calcs_last += 1
                except Exception as e:
                    logger.error(f"LastHidden Sim calculation error for '{foreign_word_value}' & '{english_word_key}': {e}")
                    similarity_last = None # Ensure it's None on error
            else:
                logger.warning(f"Skipping LastHidden similarity for pair ('{foreign_word_value}', '{english_word_key}') due to embedding error.")


            # --- Store result including BOTH similarities ---
            result_entry = {
                "english_word": english_word_key,
                "english_tokens": tok_e,
                "english_token_ids": ids_e,
                "foreign_word": foreign_word_value,
                "foreign_tokens": tok_f,
                "foreign_token_ids": ids_f,
                "similarity_raw_emb": similarity_raw,     # Renamed
                "similarity_last_hidden": similarity_last
            }
            model_individual_results_list.append(result_entry)
            # --- *** END MODIFIED *** ---

        # --- Save Individual Results ---
        individual_filename = os.path.join(INDIVIDUAL_RESULTS_FOLDER, f"{config.name}_similarity_{lang_pair}_RAW_LAST.json") # Filename reflects content
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
        all_model_summaries[config.name] = {
            "error": str(e),
            "model_path": config.model_path,
            "lang_pair": lang_pair,
            "embedding_source": "raw_and_last", # Updated source info
            "processed_pairs": processed_pairs,
            "total_pairs_in_file": len(full_word_pairs_items),
            "successful_pairs_raw_emb": successful_pairs_raw_emb,
            "successful_pairs_last_hidden": successful_pairs_last_hidden,
            "successful_similarity_calcs_raw": successful_similarity_calcs_raw,
            "successful_similarity_calcs_last": successful_similarity_calcs_last
        }

    finally:
        # --- Summary Calculation (Calculate for both methods) ---
        if not error_occurred:
            # Raw Token Embedding Stats
            avg_sim_raw = np.mean(model_similarities_raw) if model_similarities_raw else None
            std_dev_sim_raw = np.std(model_similarities_raw) if len(model_similarities_raw) > 1 else 0.0
            median_sim_raw = statistics.median(model_similarities_raw) if model_similarities_raw else None
            sim_success_rate_raw = (successful_similarity_calcs_raw / processed_pairs) * 100 if processed_pairs > 0 else 0.0

            # Last Hidden State Stats (remains the same)
            avg_sim_last = np.mean(model_similarities_last) if model_similarities_last else None
            std_dev_sim_last = np.std(model_similarities_last) if len(model_similarities_last) > 1 else 0.0
            median_sim_last = statistics.median(model_similarities_last) if model_similarities_last else None
            sim_success_rate_last = (successful_similarity_calcs_last / processed_pairs) * 100 if processed_pairs > 0 else 0.0

            all_model_summaries[config.name] = {
                "model_path": config.model_path,
                "lang_pair": lang_pair,
                "processed_pairs": processed_pairs,
                "total_pairs_in_file": len(full_word_pairs_items),
                "successful_pairs_raw_emb": successful_pairs_raw_emb,
                "successful_pairs_last_hidden": successful_pairs_last_hidden,
                "successful_similarity_calcs_raw": successful_similarity_calcs_raw,
                "successful_similarity_calcs_last": successful_similarity_calcs_last,

                "raw_emb_similarity_success_rate_percent": round(sim_success_rate_raw, 2),
                "raw_emb_average_similarity": round(float(avg_sim_raw), 6) if avg_sim_raw is not None else None,
                "raw_emb_median_similarity": round(float(median_sim_raw), 6) if median_sim_raw is not None else None,
                "raw_emb_std_dev_similarity": round(float(std_dev_sim_raw), 6),

                "last_hidden_similarity_success_rate_percent": round(sim_success_rate_last, 2),
                "last_hidden_average_similarity": round(float(avg_sim_last), 6) if avg_sim_last is not None else None,
                "last_hidden_median_similarity": round(float(median_sim_last), 6) if median_sim_last is not None else None,
                "last_hidden_std_dev_similarity": round(float(std_dev_sim_last), 6)
            }
            # Log summary for both
            avg_sim_raw_str = f"{avg_sim_raw:.4f}" if avg_sim_raw is not None else "N/A"
            med_sim_raw_str = f"{median_sim_raw:.4f}" if median_sim_raw is not None else "N/A"
            avg_sim_last_str = f"{avg_sim_last:.4f}" if avg_sim_last is not None else "N/A"
            med_sim_last_str = f"{median_sim_last:.4f}" if median_sim_last is not None else "N/A"
            logger.info(f"Summary for model {config.name}:")
            logger.info(f"  RawEmb: AvgSim={avg_sim_raw_str}, MedianSim={med_sim_raw_str}, StdDev={std_dev_sim_raw:.4f}, SuccessRate={sim_success_rate_raw:.2f}% ({successful_similarity_calcs_raw}/{processed_pairs})")
            logger.info(f"  LastHidden: AvgSim={avg_sim_last_str}, MedianSim={med_sim_last_str}, StdDev={std_dev_sim_last:.4f}, SuccessRate={sim_success_rate_last:.2f}% ({successful_similarity_calcs_last}/{processed_pairs})")

        elif config.name in all_model_summaries:
             all_model_summaries[config.name]["embedding_source"] = "raw_and_last" # Add source info even on error
             logger.warning(f"Summary calculation skipped for model {config.name} (Raw & Last Hidden) due to critical error. Error details saved.")
        else:
            logger.error(f"Model {config.name} encountered an error, but no error summary was recorded.")
        # --- End Summary Calculation ---

        # --- Memory Cleanup ---
        # (Cleanup logic remains largely the same, ensure new variables like emb_..._raw are included if necessary, although they are mostly local to the loop)
        del model
        del tokenizer
        if 'emb_f_raw' in locals(): del emb_f_raw
        if 'emb_f_last' in locals(): del emb_f_last
        if 'emb_e_raw' in locals(): del emb_e_raw
        if 'emb_e_last' in locals(): del emb_e_last
        # ... other potential variables ...
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Cleaned up resources for model {config.name}.")

# --- Save Combined Summary Results ---
try:
    with open(COMBINED_RESULTS_PATH, 'w', encoding='utf-8') as f_comb:
        json.dump(all_model_summaries, f_comb, ensure_ascii=False, indent=4)
    logger.info(f"\nSuccessfully saved combined summary results (Raw Emb & Last Hidden) for language pair '{lang_pair}' to {COMBINED_RESULTS_PATH}")
except Exception as e:
    logger.error(f"\nError saving combined summary results (Raw Emb & Last Hidden) to {COMBINED_RESULTS_PATH}: {e}")

logger.info("\n--- Word Embedding Similarity Evaluation (Raw Emb & Last Hidden) Complete ---")