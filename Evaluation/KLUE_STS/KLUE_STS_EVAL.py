import os
import json
import torch
import numpy as np
from datasets import load_dataset, Dataset # Dataset 클래스 추가
from transformers import (
    AutoModelForSequenceClassification, # Regression 용도로 사용
    AutoTokenizer,
    AutoModelForCausalLM,              # OLMo 로딩 시 필요
    AutoConfig                         # OLMo 로딩 시 필요
)
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import logging
# import re # 더 이상 필요 없음
from tqdm import tqdm
# from dotenv import load_dotenv # 불필요
# from huggingface_hub import login, HfApi # 불필요

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 기본 설정 ---
ROOT_DIRECTORY = "klue_sts_results_regression" # Regression 결과가 저장된 루트 디렉토리
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_sts_validation_numeric.json"
DATA_CACHE_DIR = "./klue_sts_regression_eval_cache"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 512
# INDIVIDUAL_LOG_FILENAME = "evaluation_log_regression.json" # 로그 파일 대신 결과 파일 사용
INDIVIDUAL_RESULT_FILENAME = "eval_results_regression.json" # 각 모델별 결과 파일 이름
SUMMARY_FILENAME = "STS_evaluation_summary_regression.json"
FINE_TUNED_MODEL_SUBDIR = "final" # 학습 스크립트 확인 필요

# --- Regression용 모델 로딩 함수 ---
def load_model_and_tokenizer_for_regression(model_config):
    num_labels = 1
    logger.info(f"Load model: {model_config.model_path} for Sequence Regression (STS Task)")
    is_local = os.path.isdir(model_config.model_path) # 경로 존재 여부로 로컬 판단
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, local_files_only=is_local, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        else: tokenizer.add_special_tokens({'pad_token': '[PAD]'}); logger.warning("Added [PAD] token.")
    elif tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])
    logger.info(f"Using pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    config = AutoConfig.from_pretrained(
        model_config.model_path, num_labels=num_labels, problem_type="regression",
        local_files_only=is_local, trust_remote_code=True)
    config.pad_token_id = tokenizer.pad_token_id
    logger.info("Model config loaded and modified for regression.")
    model = None
    try:
        logger.info(f"Attempting direct load with AutoModelForSequenceClassification for {model_config.name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_path, config=config, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=is_local, trust_remote_code=True,
            ignore_mismatched_sizes=True)
        logger.info(f"Successfully loaded {model.__class__.__name__} directly.")
    except ValueError as e:
        if "Unrecognized configuration class" in str(e) or "AutoModelForSequenceClassification" in str(e):
            logger.warning(f"Direct load failed for {model_config.name}. Attempting fallback: Load CausalLM -> Adapt.")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_path, config=config, torch_dtype=torch.bfloat16,
                local_files_only=is_local, trust_remote_code=True)
            logger.info("Base Causal LM model loaded for fallback.")
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, config=config, torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True)
            logger.info("Adapted base model to Sequence Classification model using fallback.")
            # Fallback 시 device_map="auto"가 적용 안 될 수 있으므로 수동 이동
            if not hasattr(model, 'hf_device_map'):
                 model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                 logger.info(f"Fallback model manually moved to device: {model.device}")
            else:
                 logger.info(f"Fallback model device map: {model.hf_device_map}")

        else: raise e
    except Exception as e: raise e
    model.config.use_cache = False
    if hasattr(model, 'config') and model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Model and tokenizer prepared for regression evaluation.")
    return model, tokenizer

# --- 메트릭 계산 함수 ---
def calculate_metrics(true_scores, pred_scores):
    valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores) if t is not None and p is not None and isinstance(t, (int, float)) and isinstance(p, (int, float))]
    if not valid_pairs:
        logger.warning("No valid pairs found for metrics calculation.")
        return {"num_valid_samples": 0, "F1_score": None, "Pearson_r": None, "RMSE": None, "MAE": None, "MSE": None}
    true_clean, pred_clean = zip(*valid_pairs)
    num_valid_samples = len(true_clean)
    logger.info(f"Calculating metrics based on {num_valid_samples} valid samples.")
    f1, pearson_r_val, rmse, mae, mse = None, None, None, None, None
    threshold = 3.0
    true_binary = [1 if score >= threshold else 0 for score in true_clean]
    pred_binary = [1 if score >= threshold else 0 for score in pred_clean]
    try:
        precision, recall, f1, support = precision_recall_fscore_support(true_binary, pred_binary, average='binary', zero_division=0)
        f1 = float(f1)
    except ValueError: pass
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
    return {"num_valid_samples": num_valid_samples, "F1_score": round(f1, 4) if f1 is not None else None, "Pearson_r": round(pearson_r_val, 4) if pearson_r_val is not None and not np.isnan(pearson_r_val) else None, "RMSE": round(rmse, 4) if rmse is not None else None, "MAE": round(mae, 4) if mae is not None else None, "MSE": round(mse, 4) if mse is not None else None}

# --- 단일 모델 평가 함수 (Regression) ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_dataset, output_dir):
    logger.info(f"--- Evaluating model: {model_name} (Regression) ---")
    logger.info(f"Model path: {model_path}")
    log_file_path = os.path.join(output_dir, "evaluation_log_regression.json")
    result_file_path = os.path.join(output_dir, INDIVIDUAL_RESULT_FILENAME) # 결과 파일 이름 사용
    os.makedirs(output_dir, exist_ok=True)
    model = None; tokenizer = None
    eval_results = {"model": model_name, "error": "Evaluation not completed"}
    try:
        # 임시 ModelConfig 객체 생성 (load 함수가 config 객체를 받으므로)
        temp_config = type('obj', (object,), {'name': model_name, 'model_path': model_path})()
        model, tokenizer = load_model_and_tokenizer_for_regression(temp_config)
        model.eval()
        logger.info("Model and tokenizer loaded for regression evaluation.")
    except Exception as e:
        logger.error(f"Load failed for {model_name}: {e}", exc_info=True) # 오류 상세 정보 로깅
        eval_results["error"] = f"Load failed: {e}"
        try:
            with open(result_file_path, 'w', encoding='utf-8') as f_res: json.dump(eval_results, f_res)
            # 로그 파일에도 오류 기록 (선택 사항)
            # with open(log_file_path, 'w', encoding='utf-8') as f_log: json.dump([{"error": str(e)}], f_log)
        except Exception as log_e: logger.error(f"Failed write error log/result: {log_e}")
        return
    true_scores = []; pred_scores = []; logs = []
    logger.info(f"Starting regression evaluation loop for {model_name} using {len(eval_dataset)} samples...")
    for item in tqdm(eval_dataset, desc=f"Evaluating {model_name}"):
        prompt_text = item.get("input")
        true_score = item.get("output") # 숫자 값
        log_entry = {"input_prompt": prompt_text, "true_score": true_score, "predicted_score": None, "error": None}
        if not prompt_text or true_score is None or not isinstance(true_score, (int, float)):
            logger.warning(f"Skipping item due to invalid input or true_score: {item}")
            log_entry["error"] = "Invalid input or true_score"
            true_scores.append(None); pred_scores.append(None); logs.append(log_entry)
            continue
        try:
            encoding = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length").to(model.device)
        except Exception as e_tok:
             logger.error(f"Tokenization error for item: {item}. Error: {e_tok}")
             log_entry["error"] = "Tokenization error"
             true_scores.append(true_score); pred_scores.append(None); logs.append(log_entry)
             continue
        predicted_score = None
        try:
            with torch.inference_mode():
                outputs = model(**encoding)
                predicted_score = outputs.logits.squeeze().item()
                predicted_score = max(0.0, min(5.0, predicted_score)) # 범위 조정
            log_entry["predicted_score"] = predicted_score
        except Exception as e_pred:
            logger.error(f"Prediction error for model {model_name} on item: {item}. Error: {e_pred}")
            log_entry["error"] = "Prediction error"
        true_scores.append(true_score); pred_scores.append(predicted_score); logs.append(log_entry)
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f: json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save log file: {e}")
    logger.info(f"Calculating metrics for {model_name}...")
    metrics = calculate_metrics(true_scores, pred_scores)
    eval_results = {**{"model": model_name, "error": None}, **metrics}
    logger.info(f"Saving individual evaluation results for {model_name} to: {result_file_path}")
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f: json.dump(eval_results, f, indent=4, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save result file: {e}")
    del model; del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info(f"Finished evaluation run for {model_name}.")
    print("-" * 30)

# --- *** 여기에 process_evaluation_results 함수 정의 추가 *** ---
def process_evaluation_results(root_dir, result_filename="eval_results_regression.json"):
    """루트 디렉토리 내의 각 모델 서브디렉토리에서 지정된 결과 파일을 찾아 결과를 집계합니다."""
    summary_results = {}
    logger.info(f"--- Starting Result Summary Generation (Regression) ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Looking for result files named: {result_filename}")

    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}"); return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            result_path = os.path.join(model_dir_path, result_filename)
            if os.path.exists(result_path):
                logger.info(f"Processing result file for model: {item_name} from {result_path}")
                try:
                    with open(result_path, 'r', encoding='utf-8') as f: metrics = json.load(f)
                    if isinstance(metrics, dict) and metrics.get("error") is None:
                        summary_results[item_name] = metrics
                    elif isinstance(metrics, dict):
                         summary_results[item_name] = {"error": metrics.get("error", "Unknown error in result file")}
                    else: summary_results[item_name] = {"error": "Invalid format in result file"}
                except Exception as e:
                    logger.error(f"Error processing result file {result_path}: {e}")
                    summary_results[item_name] = {"error": f"Failed to read/parse result file: {e}"}
            else:
                 if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)):
                     logger.warning(f"Result file '{result_filename}' not found in model directory: {item_name}")
                     summary_results[item_name] = {"error": f"{result_filename} not found"}

    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving aggregated regression results summary to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Summary results successfully saved to {output_path}")
    except Exception as e: logger.error(f"Failed to save summary results: {e}")

    return summary_results
# --- *** process_evaluation_results 함수 정의 끝 *** ---


# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting KLUE-STS Regression Evaluation Script ---")

    # 1. 평가 데이터셋 로드 (numeric 파일)
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        evaluation_dataset = load_dataset("json", data_files={"validation": JSON_VAL_DATASET_PATH}, cache_dir=DATA_CACHE_DIR, split="validation")
        logger.info(f"Loaded {len(evaluation_dataset)} samples for evaluation using load_dataset.")
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        # 'output' 컬럼 확인 및 float 변환
        if 'output' in evaluation_dataset.features:
             if evaluation_dataset.features['output'].dtype != 'float32' and evaluation_dataset.features['output'].dtype != 'float64':
                 logger.info("Converting 'output' column to float...")
                 evaluation_dataset = evaluation_dataset.map(lambda x: {'output': float(x['output']) if x['output'] is not None else None})
             logger.info(f"'output' column type: {evaluation_dataset.features['output'].dtype}")
        else:
             logger.error("CRITICAL: 'output' column not found in the evaluation dataset.")
             exit()

    except Exception as e:
        logger.error(f"CRITICAL: Failed to load or process the evaluation dataset from {JSON_VAL_DATASET_PATH}. Error: {e}", exc_info=True)
        exit()

    # 2. 루트 디렉토리 내 모델들에 대해 순차적으로 평가 수행
    if not os.path.isdir(ROOT_DIRECTORY):
        logger.error(f"Evaluation root directory not found: {ROOT_DIRECTORY}")
        exit()

    evaluated_models = []
    eval_subset_to_run = evaluation_dataset.select(range(min(MAX_EVAL_SAMPLES, len(evaluation_dataset))))
    logger.info(f"Will evaluate on {len(eval_subset_to_run)} samples (up to MAX_EVAL_SAMPLES={MAX_EVAL_SAMPLES})")

    # ModelConfig 클래스 정의 확인 (스크립트 상단에 있어야 함)
    if 'ModelConfig' not in globals():
        logger.error("CRITICAL: ModelConfig class is not defined.")
        exit()

    for item_name in os.listdir(ROOT_DIRECTORY):
        model_dir_path = os.path.join(ROOT_DIRECTORY, item_name)
        if os.path.isdir(model_dir_path):
            fine_tuned_model_path = os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)
            if os.path.isdir(fine_tuned_model_path):
                 # evaluate_single_model 호출 전 ModelConfig 객체를 사용하지 않으므로, 경로만 전달
                 evaluate_single_model(
                     model_name=item_name,
                     model_path=fine_tuned_model_path,
                     tokenizer_path=fine_tuned_model_path, # 토크나이저 경로도 동일
                     eval_dataset=eval_subset_to_run,
                     output_dir=model_dir_path
                 )
                 evaluated_models.append(item_name)
            else:
                logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found in {model_dir_path}.")

    # 3. 모든 개별 평가 완료 후 요약 생성
    if evaluated_models:
        # 여기서 process_evaluation_results 함수 호출
        summary_results = process_evaluation_results(ROOT_DIRECTORY, result_filename=INDIVIDUAL_RESULT_FILENAME)

        # 요약 결과 출력
        if summary_results:
            print("\n--- Final Evaluation Summary (Regression) ---")
            sorted_models = sorted(summary_results.items(),
                                   key=lambda item: item[1].get('Pearson_r', -1) if isinstance(item[1], dict) and item[1].get('Pearson_r') is not None else -1,
                                   reverse=True)
            for model_name, metrics in sorted_models:
                print(f"\nModel: {model_name}")
                if isinstance(metrics, dict):
                     if "error" in metrics and metrics["error"]: print(f"  Error: {metrics['error']}")
                     else:
                         print(f"  Num Valid Samples: {metrics.get('num_valid_samples', 'N/A')}")
                         print(f"  Pearson Corr (r): {metrics.get('Pearson_r', 'N/A')}")
                         print(f"  MAE: {metrics.get('MAE', 'N/A')}")
                         print(f"  RMSE: {metrics.get('RMSE', 'N/A')}")
                else: print("  Invalid metrics format.")
            print("-" * 30)
    else:
        logger.warning("No models were evaluated. Skipping summary generation.")

    logger.info("--- KLUE-STS Regression Evaluation Script Finished ---")