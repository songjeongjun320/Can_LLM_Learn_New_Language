# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

# Third-party imports
import torch
# 필요한 클래스들을 명시적으로 임포트하면 타입 힌팅과 자동완성에 도움이 됩니다.
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Model Configurations ---
@dataclass
class ModelInfo:
    name: str
    path: str
    is_local: bool

MODEL_INFOS = [
    ModelInfo(name="OLMo-1b-org", path="allenai/OLMo-1B", is_local=False),
    ModelInfo(name="OLMo-7b-org", path="allenai/OLMo-7B", is_local=False),
    ModelInfo(name="BERT-base-uncased", path="bert-base-uncased", is_local=False),
    ModelInfo(name="T5-base", path="t5-base", is_local=False),
    ModelInfo(name="Llama-3.2-3b", path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", is_local=True),
    # Ensure this path is correct or use the HF identifier if available and you have access
    ModelInfo(name="Llama-4-Scout-17B-16E", path="meta-llama/Llama-4-Scout-17B-16E", is_local=False),
]

# --- File Paths ---
KOREAN_WORDS_PATH = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/words.json'
OUTPUT_BASE_DIR = 'Check_Tokenization' # Base directory for output files

# --- Helper Function ---
def analyze_tokenization(tokens: List[str], unk_token: str | None) -> Tuple[str, List[str]]:
    """Analyzes the tokenization result to determine the status."""
    if not tokens:
        return "ERROR", ["Tokenization resulted in empty list"]

    num_tokens = len(tokens)
    contains_unk = unk_token is not None and unk_token in tokens

    if num_tokens == 1:
        if contains_unk:
            status = "UNK_ONLY"
        else:
            status = "DIRECT"
    else: # num_tokens > 1
        if contains_unk:
            status = "MIXED"
        else:
            status = "SUBWORD"

    return status, tokens

# --- Main Logic ---
def check_korean_tokenization(model_infos: List[ModelInfo], words_path: str, output_dir: str):
    """
    Checks how Korean words are tokenized by specified models and saves results
    into separate files per model in the output directory. Logs tokenizer class info.

    Args:
        model_infos: A list of ModelInfo objects specifying models to check.
        words_path: Path to the JSON file containing Korean words under the "kor" key.
        output_dir: The base directory where individual model result files will be saved.
    """
    # 1. Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Could not create output directory '{output_dir}'. Error: {e}")
        return

    # 2. Load Korean words
    try:
        with open(words_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "kor" not in data or not isinstance(data["kor"], list):
            logger.error(f"Error: Input JSON '{words_path}' must contain a list under the key 'kor'.")
            return
        korean_words: List[str] = data["kor"]
        logger.info(f"Loaded {len(korean_words)} Korean words from {words_path}")
    except FileNotFoundError:
        logger.error(f"Error: Input word file not found at {words_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {words_path}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading words: {e}")
        return

    if not korean_words:
        logger.warning("No Korean words found in the input file. Exiting.")
        return

    # 3. Process each model
    for model_info in model_infos:
        logger.info(f"\n--- Processing Model: {model_info.name} ({model_info.path}) ---")
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
        model_results: Dict[str, Any] = {}
        unk_token: str | None = None
        model_load_error = False

        try:
            # Load tokenizer
            logger.info(f"Attempting to load tokenizer for {model_info.name} from '{model_info.path}'...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.path,
                trust_remote_code=True,
                local_files_only=model_info.is_local,
                use_fast=True
            )

            # *** ADDED: Log the type (class) of the loaded tokenizer ***
            tokenizer_class_name = type(tokenizer).__name__
            logger.info(f"Successfully loaded tokenizer for {model_info.name}. Type: {tokenizer_class_name}")
            # ***********************************************************

            unk_token = tokenizer.unk_token
            if unk_token is None:
                logger.warning(f"Tokenizer '{tokenizer_class_name}' for {model_info.name} does not have an explicit unk_token defined.")
            logger.info(f"UNK token for {model_info.name}: '{unk_token}'")

            # Check each Korean word
            word_count = len(korean_words)
            for i, word in enumerate(korean_words, 1):
                if i % 1000 == 0:
                    logger.info(f"  Processing word {i}/{word_count} for {model_info.name}...")

                if word and isinstance(word, str):
                    try:
                        tokens = tokenizer.tokenize(word)
                        status, analyzed_tokens = analyze_tokenization(tokens, unk_token)
                        model_results[word] = {"status": status, "tokens": analyzed_tokens}
                    except Exception as tokenize_err:
                        logger.error(f"Error tokenizing word '{word}' with {model_info.name} ({tokenizer_class_name}): {tokenize_err}", exc_info=False)
                        model_results[word] = {"status": "ERROR", "tokens": [f"Tokenization error: {tokenize_err}"]}
                elif not isinstance(word, str):
                     logger.warning(f"Skipping non-string item in word list: {word}")
                     model_results[f"INVALID_ITEM_{i}"] = {"status": "ERROR", "tokens": ["Item was not a string"]}
                else:
                    model_results[""] = {"status": "EMPTY", "tokens": []}

            logger.info(f"Finished checking tokenization for {word_count} words with {model_info.name}.")

        except (OSError, ValueError) as e: # Catch specific loading errors
            logger.error(f"Failed to load or process tokenizer for {model_info.name} from {model_info.path}. Error: {e}")
            model_results["_MODEL_ERROR_"] = f"Error loading/processing tokenizer: {e}"
            model_load_error = True
        except Exception as e: # Catch other unexpected errors during loading/setup
            logger.error(f"An unexpected error occurred processing {model_info.name} before tokenization loop: {e}", exc_info=True)
            model_results["_MODEL_ERROR_"] = f"Unexpected setup error: {e}"
            model_load_error = True

        finally:
            # --- Save results for the CURRENT model ---
            model_output_filename = f"{model_info.name.replace('/', '_')}_tokenization.json"
            model_output_path = os.path.join(output_dir, model_output_filename)

            try:
                logger.info(f"Attempting to save results for {model_info.name} to {model_output_path}...")
                with open(model_output_path, 'w', encoding='utf-8') as f_model_out:
                    json.dump(model_results, f_model_out, ensure_ascii=False, indent=4)
                logger.info(f"Successfully saved tokenization results for {model_info.name} to {model_output_path}")
            except Exception as save_err:
                logger.error(f"Error saving results for {model_info.name} to {model_output_path}: {save_err}")

            # Clean up memory
            # Check if tokenizer was successfully assigned before deleting
            if tokenizer is not None:
                 del tokenizer
                 tokenizer = None # Ensure reference is removed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Cleaned up resources for {model_info.name}.")

    logger.info("\n--- All models processed ---")

# --- Run the check ---
if __name__ == "__main__":
    check_korean_tokenization(MODEL_INFOS, KOREAN_WORDS_PATH, OUTPUT_BASE_DIR)