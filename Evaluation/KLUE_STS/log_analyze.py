import os
import re
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 기본 설정 ---
ROOT_DIRECTORY = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/KLUE_STS/klue_sts_results"
LOG_FILENAME_TO_PROCESS = "evaluation_log.json" # 처리할 로그 파일 이름 (이전 스크립트에서 생성된 이름)
SUMMARY_FILENAME = "STS_evaluation_summary_recalculated.json" # 새로 계산된 결과 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final_model"
# --- 메트릭 계산 함수 (기존과 동일) ---
def calculate_metrics(true_scores, pred_scores):
    valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores) if t is not None and p is not None and isinstance(t, (int, float)) and isinstance(p, (int, float))]
    if not valid_pairs: logger.warning("No valid pairs found."); return {"num_valid_samples": 0}
    true_clean, pred_clean = zip(*valid_pairs); num_valid_samples = len(true_clean)
    logger.info(f"Calculating metrics based on {num_valid_samples} valid samples.")
    f1, pearson_r_val, rmse, mae, mse = None, None, None, None, None
    threshold = 3.0; true_binary = [1 if score >= threshold else 0 for score in true_clean]; pred_binary = [1 if score >= threshold else 0 for score in pred_clean]
    try: _, _, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='binary', zero_division=0); f1 = float(f1)
    except ValueError: pass
    if len(set(true_clean)) > 1 and len(set(pred_clean)) > 1:
        try: pearson_r_val, _ = pearsonr(true_clean, pred_clean); pearson_r_val = float(pearson_r_val) if not np.isnan(pearson_r_val) else None
        except Exception as e: logger.warning(f"Pearson calc error: {e}"); pearson_r_val = None
    else: logger.warning("Cannot calc Pearson: data values may be constant."); pearson_r_val = None
    try: mse = mean_squared_error(true_clean, pred_clean); rmse = np.sqrt(mse); mae = mean_absolute_error(true_clean, pred_clean); mse, rmse, mae = float(mse), float(rmse), float(mae)
    except Exception as e: logger.error(f"Error calc regression metrics: {e}")
    return {"num_valid_samples": num_valid_samples, "F1_score": round(f1, 4) if f1 is not None else None, "Pearson_r": round(pearson_r_val, 4) if pearson_r_val is not None else None, "RMSE": round(rmse, 4) if rmse is not None else None, "MAE": round(mae, 4) if mae is not None else None, "MSE": round(mse, 4) if mse is not None else None}

# --- '정수.정수' 형태 점수 추출 함수 (eval 스크립트와 동일) ---
def extract_decimal_score(text):
    if not text or not isinstance(text, str): return None, []
    pattern = r'\b([0-5]\.\d+)\b' # 소수점 필수
    matches_str = re.findall(pattern, text)
    valid_scores = []
    if not matches_str:
        pattern_no_boundary = r'([0-5]\.\d+)'
        matches_str = re.findall(pattern_no_boundary, text)
        if not matches_str: return None, []
    for s_str in matches_str:
        try:
            score = float(s_str)
            if 0.0 <= score <= 5.0: valid_scores.append(score)
        except ValueError: continue
    if not valid_scores: return None, matches_str
    # 전략: 마지막 점수 사용 (이전 코드와 일관성 유지)
    final_score = valid_scores[-1]
    return final_score, valid_scores

# --- 로그 파일 처리 및 메트릭 재계산 함수 ---
def process_and_recalculate_logs(root_dir, log_filename=LOG_FILENAME_TO_PROCESS):
    """로그 파일을 읽어 generated_text에서 점수를 다시 추출하고 메트릭을 계산합니다."""
    results = {}
    logger.info(f"--- Starting Log Processing and Recalculation ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Processing log files named: {log_filename}")

    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}"); return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            log_path = os.path.join(model_dir_path, log_filename)
            if os.path.exists(log_path):
                logger.info(f"Processing log file for model: {item_name}")
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logger.warning(f"Log file {log_path} is not a list. Skipping."); continue

                    true_scores_from_log = []
                    newly_extracted_scores = []
                    processed_count = 0
                    extraction_fail_count = 0

                    for log_entry in tqdm(logs, desc=f"Re-extracting scores for {item_name}"):
                        if not isinstance(log_entry, dict): continue

                        true_score = log_entry.get("true_score") # 로그에 저장된 실제 점수 사용
                        generated_text = log_entry.get("generated_text")

                        # true_score 유효성 검사
                        if true_score is None or not isinstance(true_score, (int, float)):
                            # logger.debug("Skipping entry with invalid true_score.")
                            continue # 유효한 true_score 없으면 메트릭 계산 불가

                        # generated_text에서 점수 다시 추출
                        pred_score, _ = extract_decimal_score(generated_text) # 새로 추출

                        true_scores_from_log.append(true_score)
                        newly_extracted_scores.append(pred_score) # 추출된 점수 (None일 수 있음)
                        processed_count += 1
                        if pred_score is None:
                            extraction_fail_count += 1

                    logger.info(f"Finished re-extraction for {item_name}. Processed: {processed_count}, Extraction Fails: {extraction_fail_count}")

                    # 새로 추출된 점수로 메트릭 계산
                    metrics = calculate_metrics(true_scores_from_log, newly_extracted_scores)
                    # 모델 이름 추가
                    metrics["model"] = item_name
                    results[item_name] = metrics

                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {log_path}. Skipping.")
                    results[item_name] = {"model": item_name, "error": "JSON Decode Error"}
                except Exception as e:
                    logger.error(f"Unexpected error processing log {log_path}: {e}")
                    results[item_name] = {"model": item_name, "error": str(e)}
            else:
                # 로그 파일 없는 경우 처리 (기존 로직 유지)
                if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)):
                     logger.warning(f"Log file '{log_filename}' not found in model directory: {item_name}")
                     results[item_name] = {"model": item_name, "error": f"{log_filename} not found"}

    # 집계된 결과 저장
    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving recalculated results summary to: {output_path}")
    try:
        # JSON 직렬화를 위해 NumPy 타입을 Python 기본 타입으로 변환 (필요시)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
            return obj
        serializable_results = convert_numpy_types(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Recalculated summary results saved to {output_path}")
    except Exception as e: logger.error(f"Failed to save summary results: {e}")

    return results

# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting KLUE-STS Log Reprocessing Script (Decimal Only Extraction) ---")

    # 로그 처리 및 요약 생성 함수 호출
    summary_results = process_and_recalculate_logs(ROOT_DIRECTORY, log_filename=LOG_FILENAME_TO_PROCESS)

    # 요약 결과 출력
    if summary_results:
        print("\n--- Recalculated Evaluation Summary (STS - Decimal Only) ---")
        valid_results = {k: v for k, v in summary_results.items() if isinstance(v, dict) and "error" not in v and v.get("Pearson_r") is not None}
        sorted_models = sorted(valid_results.items(), key=lambda item: item[1].get('Pearson_r', -1), reverse=True)

        for model_name, metrics in sorted_models:
            print(f"\nModel: {model_name}")
            print(f"  Num Valid Samples: {metrics.get('num_valid_samples', 'N/A')}")
            print(f"  Pearson Correlation: {metrics.get('Pearson_r', 'N/A')}")
            print(f"  RMSE: {metrics.get('RMSE', 'N/A')}")
            print(f"  F1 Score (>=3.0): {metrics.get('F1_score', 'N/A')}")

        error_models = {k: v for k, v in summary_results.items() if not isinstance(v, dict) or "error" in v or v.get("Pearson_r") is None}
        if error_models:
            print("\n--- Models with Errors or No Valid Results ---")
            for model_name, metrics in error_models.items():
                 print(f"\nModel: {model_name}")
                 if isinstance(metrics, dict) and "error" in metrics: print(f"  Error: {metrics['error']}")
                 elif isinstance(metrics, dict): print(f"  Metrics: {metrics}")
                 else: print(f"  Invalid result format: {metrics}")
        print("-" * 30)
    else:
        logger.warning("Could not generate summary results.")

    logger.info("--- Log Reprocessing Script Finished ---")