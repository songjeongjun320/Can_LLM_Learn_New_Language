import json
from transformers import AutoTokenizer
import logging
import os
import time # 시간 측정을 위해 추가

# 로깅 설정 (간단하게)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Suppress unnecessary warnings from transformers/huggingface
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- Configuration ---
JSON_FILE_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/ch_kr.json" # 확인할 단어 쌍 파일 경로

# 확인할 토크나이저 이름과 경로(또는 Hugging Face ID) 딕셔너리
TOKENIZER_PATHS = {
    "t5-base": "t5-base",
    "mt5-base": "google/mt5-base",
    "olmo-1b": "allenai/olmo-1b",
    "bert-base-uncased": "bert-base-uncased",
    "llama3.2-3b": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"
}

# 결과를 저장할 JSON 파일 경로
RESULTS_JSON_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/Check_Tokenization/vocab_check_results_all.json"

# 로그 출력 간격 설정 (예: 1000개 쌍마다 로그 출력)
LOG_INTERVAL = 1000

# --- Load Word Pairs from JSON ---
# (이전과 동일)
try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        word_pairs = json.load(f)
    if not isinstance(word_pairs, dict):
        logger.error(f"{JSON_FILE_PATH} 파일은 딕셔너리 형태여야 합니다.")
        exit()
    logger.info(f"{JSON_FILE_PATH}에서 {len(word_pairs)}개의 단어 쌍 로드 완료.")
except FileNotFoundError:
    logger.error(f"단어 쌍 파일을 찾을 수 없습니다: {JSON_FILE_PATH}")
    exit()
except json.JSONDecodeError:
    logger.error(f"단어 쌍 JSON 파일 디코딩 오류: {JSON_FILE_PATH}")
    exit()
except Exception as e:
    logger.error(f"단어 쌍 파일 읽기 오류 {JSON_FILE_PATH}: {e}")
    exit()


# --- Helper Function ---
# (check_token_in_vocab 함수 이전과 동일)
def check_token_in_vocab(tokenizer, token):
    token_with_space = " " + token
    vocab = tokenizer.get_vocab()
    return token in vocab or token_with_space in vocab

# --- Load Tokenizers (Iteratively) ---
# (이전과 동일)
tokenizers = {}
logger.info("--- Loading Tokenizers ---")
for name, path in TOKENIZER_PATHS.items():
    try:
        needs_trust = any(sub in name.lower() for sub in ["llama", "olmo"])
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=needs_trust)
        tokenizers[name] = tokenizer
        logger.info(f"Loaded tokenizer: {name} from {path} (Vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        logger.error(f"토크나이저 로딩 실패 ({name} from {path}): {e}")


# --- Check Words in Vocabularies with Progress Logging ---
def check_word_pairs_in_vocab(tokenizer, word_pairs_dict, log_interval=LOG_INTERVAL):
    """
    단어 쌍들이 어휘집에 있는지 확인하고 진행 상황을 로그로 남깁니다.
    결과 딕셔너리를 반환합니다.
    """
    tokenizer_name_or_path = getattr(tokenizer, 'name_or_path', 'Unknown')
    logger.info(f"\n--- Checking words from {JSON_FILE_PATH} in {tokenizer_name_or_path} vocab ---")
    korean_found_count = 0
    chinese_found_count = 0
    processed_count = 0 # 처리된 쌍 개수 카운터
    total_pairs = len(word_pairs_dict)
    start_time = time.time() # 시작 시간 기록

    if total_pairs == 0:
        logger.warning("단어 쌍 딕셔너리가 비어 있습니다. 확인을 건너뜁니다.")
        # ... (이전과 동일한 반환 로직) ...
        return { ... } # 간결성을 위해 생략

    vocab = tokenizer.get_vocab() # 어휘집 한 번 로드

    # --- 수정: 단어 쌍 순회하며 로그 출력 ---
    for korean_word, chinese_word in word_pairs_dict.items():
        # --- 기존 로직 ---
        is_kr_str = isinstance(korean_word, str)
        is_ch_str = isinstance(chinese_word, str)
        if is_kr_str and (korean_word in vocab or (" " + korean_word) in vocab):
            korean_found_count += 1
        if is_ch_str and (chinese_word in vocab or (" " + chinese_word) in vocab):
            chinese_found_count += 1
        # --- 로깅 추가 ---
        processed_count += 1
        # LOG_INTERVAL 배수마다 또는 마지막 항목일 때 로그 출력
        if processed_count % log_interval == 0 or processed_count == total_pairs:
            elapsed_time = time.time() - start_time
            # \t를 사용하여 탭으로 구분된 로그 출력
            logger.info(f"\tProcessed: {processed_count}/{total_pairs}\tElapsed: {elapsed_time:.2f}s")
            # print(f"\tProcessed: {processed_count}/{total_pairs}\tElapsed: {elapsed_time:.2f}s") # print 사용시

    # --- 나머지 결과 계산 및 반환 (이전과 동일) ---
    korean_percentage = (korean_found_count / total_pairs) if total_pairs > 0 else 0.0
    chinese_percentage = (chinese_found_count / total_pairs) if total_pairs > 0 else 0.0

    logger.info(f"Korean words found directly in vocab: {korean_found_count} / {total_pairs} ({korean_percentage:.2%})")
    logger.info(f"Chinese words found directly in vocab: {chinese_found_count} / {total_pairs} ({chinese_percentage:.2%})")
    if korean_found_count < total_pairs or chinese_found_count < total_pairs:
         logger.info("(Note: Words not found directly might be tokenized into subwords)")

    return {
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "total_pairs": total_pairs,
        "korean_found_count": korean_found_count,
        "korean_found_percentage": korean_percentage,
        "chinese_found_count": chinese_found_count,
        "chinese_found_percentage": chinese_percentage
    }

# --- 실행 및 결과 저장 ---
# (이전과 동일)
all_results = {}
for name, tokenizer in tokenizers.items():
    result = check_word_pairs_in_vocab(tokenizer, word_pairs) # LOG_INTERVAL 기본값 사용
    all_results[name] = result

try:
    logger.info(f"\n--- Saving results to {RESULTS_JSON_PATH} ---")
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved successfully.")
except Exception as e:
    logger.error(f"결과 파일 쓰기 오류 {RESULTS_JSON_PATH}: {e}")

print("\n--- Vocabulary check and result saving finished ---")