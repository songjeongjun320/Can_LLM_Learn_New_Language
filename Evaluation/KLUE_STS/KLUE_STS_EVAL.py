import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv # .env 파일 사용 시 필요
from huggingface_hub import login, HfApi # Hugging Face Hub 로그인 필요 시 사용

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 기본 설정 ---
ROOT_DIRECTORY = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/KLUE_STS/klue_sts_results"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_validation.json"
DATA_CACHE_DIR = "./klue_sts_eval_cache" # 평가용 캐시 디렉토리 (기존과 분리)
MAX_LENGTH = 1024
MAX_EVAL_SAMPLES = 200 # 전체 평가를 위해 주석 처리 또는 매우 큰 값으로 설정
INDIVIDUAL_LOG_FILENAME = "evaluation_log.json" # 각 모델별 로그 파일 이름
SUMMARY_FILENAME = "STS_evaluation_summary.json" # 최종 요약 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final_model" # 파인튜닝된 모델이 저장된 하위 디렉토리 이름 (이전 코드 기준)

# --- 메트릭 계산 함수 (사용자 제공 기반) ---
def calculate_metrics(true_scores, pred_scores):
    """주어진 실제 점수와 예측 점수 리스트로부터 다양한 메트릭을 계산합니다."""
    valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores)
                   if t is not None and p is not None and isinstance(t, (int, float)) and isinstance(p, (int, float))]

    if not valid_pairs:
        logger.warning("No valid (true_score, predicted_score) pairs found to calculate metrics.")
        return {
            "num_valid_samples": 0, "F1_score": None, "Pearson_r": None,
            "RMSE": None, "MAE": None, "MSE": None
        }

    true_clean, pred_clean = zip(*valid_pairs)
    num_valid_samples = len(true_clean)
    logger.info(f"Calculating metrics based on {num_valid_samples} valid samples.")

    f1, pearson_r_val, rmse, mae, mse = None, None, None, None, None
    threshold = 3.0
    true_binary = [1 if score >= threshold else 0 for score in true_clean]
    pred_binary = [1 if score >= threshold else 0 for score in pred_clean]
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary', zero_division=0)
        f1 = float(f1)
    except ValueError: pass # Handle cases with only one class if necessary

    if len(set(true_clean)) > 1 and len(set(pred_clean)) > 1:
        try:
            pearson_r_val, _ = pearsonr(true_clean, pred_clean)
            pearson_r_val = float(pearson_r_val)
        except Exception: pass
    else: pearson_r_val = None

    try:
        mse = mean_squared_error(true_clean, pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_clean, pred_clean)
        mse, rmse, mae = float(mse), float(rmse), float(mae)
    except Exception: pass

    return {
        "num_valid_samples": num_valid_samples,
        "F1_score": round(f1, 4) if f1 is not None else None,
        "Pearson_r": round(pearson_r_val, 4) if pearson_r_val is not None and not np.isnan(pearson_r_val) else None,
        "RMSE": round(rmse, 4) if rmse is not None else None,
        "MAE": round(mae, 4) if mae is not None else None,
        "MSE": round(mse, 4) if mse is not None else None
    }


# --- 단일 모델 평가 및 로그 저장 함수 ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_dataset, output_dir):
    """단일 파인튜닝된 모델을 평가하고 로그를 저장합니다."""
    logger.info(f"--- Evaluating model: {model_name} ---")
    logger.info(f"Model path: {model_path}")

    # 평가 로그 저장 경로
    log_file_path = os.path.join(output_dir, INDIVIDUAL_LOG_FILENAME)
    os.makedirs(output_dir, exist_ok=True) # output_dir이 없을 경우 생성

    model = None
    tokenizer = None

    try:
        # 모델 및 토크나이저 로드
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            logger.info("Tokenizer does not have pad_token, setting it to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, # 또는 모델 저장 시 사용된 dtype
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        model.eval() # 평가 모드 설정
        logger.info("Model and tokenizer loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer for {model_name}: {e}")
        # 실패 시 로그 파일에 에러 기록 (선택 사항)
        error_log = {"error": f"Failed to load model/tokenizer: {e}"}
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump([error_log], f, indent=2, ensure_ascii=False) # 리스트 형태로 저장
        except Exception as log_e:
            logger.error(f"Failed to write error log for {model_name}: {log_e}")
        return # 평가 진행 불가

    # 평가 루프
    true_scores = []
    pred_scores = []
    logs = []

    logger.info(f"Starting evaluation loop for {model_name} using {len(eval_dataset)} samples...")
    # eval_dataset 전체를 사용 (MAX_EVAL_SAMPLES 사용 시 아래 수정)
    # for item in tqdm(eval_dataset[:MAX_EVAL_SAMPLES], desc=f"Evaluating {model_name}"):
    for item in tqdm(eval_dataset, desc=f"Evaluating {model_name}"):
        prompt_text = item.get("input")
        output_text = item.get("output")

        if not prompt_text or not output_text:
            logger.warning("Skipping item due to missing 'input' or 'output'.")
            continue

        # 실제 점수 추출
        true_score = None
        true_score_match = re.search(r'The similarity score is\s*([0-5](?:\.\d+)?)', output_text)
        if true_score_match:
            try: true_score = float(true_score_match.group(1))
            except ValueError: pass
        if true_score is None:
            logger.warning(f"Could not parse true score from: {output_text}")
            # 로그에는 기록하되, 메트릭 계산에는 포함되지 않음
            # continue # 필요시 건너뛰기

        # 모델 입력 준비 및 생성
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
        predicted_score = None
        full_generated_text = "[GENERATION ERROR]"
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"], max_new_tokens=20, temperature=0.1,
                    num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # 생성된 텍스트에서 프롬프트 부분 제거 시도
            generated_response_part = full_generated_text
            if full_generated_text.strip().startswith(prompt_text.strip()):
                 generated_response_part = full_generated_text[len(prompt_text):].strip()

            # 예측 점수 추출
            pred_score_match = re.search(r'The similarity score is\s*([0-5](?:\.\d+)?)', generated_response_part)
            if pred_score_match:
                try:
                    extracted_val = float(pred_score_match.group(1))
                    if 0 <= extracted_val <= 5: predicted_score = extracted_val
                    else: logger.warning(f"Pred score out of range (0-5): {extracted_val}")
                except ValueError: pass
            else: logger.warning(f"Pred score pattern not found in: '{generated_response_part[:100]}...'")

        except Exception as e:
            logger.error(f"Error during generation for model {model_name}, input: {prompt_text[:50]}... E: {e}")

        # 결과 기록 (true_score는 None일 수 있지만 로그에는 포함)
        true_scores.append(true_score)
        pred_scores.append(predicted_score)
        logs.append({
            "input_prompt": prompt_text, "expected_output_text": output_text,
            "generated_text": full_generated_text, "true_score": true_score,
            "predicted_score": predicted_score
        })

    # 개별 로그 파일 저장
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        logger.info(f"Log saved successfully for {model_name}.")
    except Exception as e:
        logger.error(f"Failed to save log file for {model_name}: {e}")

    # 메모리 정리 (선택 사항, 특히 루프 내에서 유용)
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Finished evaluation for {model_name} and cleaned up resources.")
    print("-" * 30) # 모델 간 구분

# --- 로그 처리 및 요약 함수 (사용자 제공 기반) ---
def process_evaluation_logs(root_dir, log_filename=INDIVIDUAL_LOG_FILENAME):
    """루트 디렉토리 내의 각 모델 서브디렉토리에서 지정된 로그 파일을 찾아 메트릭을 계산하고 결과를 집계합니다."""
    results = {}
    logger.info(f"--- Starting Log Processing and Summary Generation ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Looking for log files named: {log_filename}")

    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}")
        return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            log_path = os.path.join(model_dir_path, log_filename)
            if os.path.exists(log_path):
                logger.info(f"Processing log file for model: {item_name}")
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if not isinstance(logs, list): continue # Skip if not a list

                    true_scores = [log.get("true_score") for log in logs if isinstance(log, dict)]
                    pred_scores = [log.get("predicted_score") for log in logs if isinstance(log, dict)]

                    metrics = calculate_metrics(true_scores, pred_scores)
                    results[item_name] = metrics
                except Exception as e:
                    logger.error(f"Error processing log {log_path}: {e}")
                    results[item_name] = {"error": str(e)}
            else:
                 # 모델 디렉토리는 있지만 로그 파일이 없는 경우 (평가 실패 등)
                 if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)): # 모델 파일이 있는지 확인
                     logger.warning(f"Log file '{log_filename}' not found in model directory: {item_name}")
                     results[item_name] = {"error": f"{log_filename} not found"}
                 # else: 모델 디렉토리가 아니거나 관련 없는 디렉토리일 수 있음

    # 집계된 결과 저장
    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving aggregated results summary to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Summary results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save summary results: {e}")

    return results

# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting Evaluation Script ---")

    # 1. 평가 데이터셋 로드 수정 (json.load 사용)
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        # 파일을 파이썬 리스트로 직접 로드
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            evaluation_data_list = json.load(f)

        # 파이썬 리스트를 datasets.Dataset 객체로 변환
        from datasets import Dataset # Dataset 클래스 import
        evaluation_dataset = Dataset.from_list(evaluation_data_list)

        logger.info(f"Loaded {len(evaluation_dataset)} samples for evaluation using json.load.")
        os.makedirs(DATA_CACHE_DIR, exist_ok=True) # 캐시 디렉토리 생성

    except json.JSONDecodeError as e:
         logger.error(f"CRITICAL: Failed to decode JSON from {JSON_VAL_DATASET_PATH}. Check file format. Error: {e}")
         exit()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load or convert the evaluation dataset. Cannot proceed. Error: {e}")
        exit() # 평가 데이터 없으면 종료

    # --- 이하 코드는 동일 ---
    # 2. 루트 디렉토리 내 모델들에 대해 순차적으로 평가 수행
    if not os.path.isdir(ROOT_DIRECTORY):
        logger.error(f"Evaluation root directory not found: {ROOT_DIRECTORY}")
        exit()

    evaluated_models = []
    # MAX_EVAL_SAMPLES 적용 위치 변경
    # evaluate_single_model에 전체 데이터셋을 넘기고, 그 안에서 슬라이싱 하거나
    # 여기서 부분집합을 만들어 넘김
    # 예시: 여기서 부분집합 생성
    eval_subset_to_run = evaluation_dataset.select(range(min(MAX_EVAL_SAMPLES, len(evaluation_dataset))))
    logger.info(f"Will evaluate on {len(eval_subset_to_run)} samples (up to MAX_EVAL_SAMPLES={MAX_EVAL_SAMPLES})")


    for item_name in os.listdir(ROOT_DIRECTORY):
        model_dir_path = os.path.join(ROOT_DIRECTORY, item_name)

        if os.path.isdir(model_dir_path):
            fine_tuned_model_path = os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)

            if os.path.isdir(fine_tuned_model_path):
                 evaluate_single_model(
                     model_name=item_name,
                     model_path=fine_tuned_model_path,
                     tokenizer_path=fine_tuned_model_path,
                     # eval_dataset=evaluation_dataset, # 전체 데이터셋 전달 대신
                     eval_dataset=eval_subset_to_run, # 부분 집합 전달
                     output_dir=model_dir_path
                 )
                 evaluated_models.append(item_name)
            else:
                logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found.")

    # 3. 모든 개별 평가 완료 후 로그 처리 및 요약 생성
    if evaluated_models: # 최소 하나라도 평가가 수행되었다면
        summary_results = process_evaluation_logs(ROOT_DIRECTORY, log_filename=INDIVIDUAL_LOG_FILENAME)

        # 요약 결과 출력 (선택 사항)
        if summary_results:
            print("\n--- Final Evaluation Summary ---")
            for model_name, metrics in summary_results.items():
                print(f"\nModel: {model_name}")
                if "error" in metrics:
                    print(f"  Error: {metrics['error']}")
                else:
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name}: {value}")
            print("-" * 30)
    else:
        logger.warning("No models were evaluated. Skipping summary generation.")

    logger.info("--- Evaluation Script Finished ---")