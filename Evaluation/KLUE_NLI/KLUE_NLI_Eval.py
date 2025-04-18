import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, # 학습 시 사용한 모델
    AutoTokenizer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from tqdm import tqdm
from datasets import Dataset # 리스트 변환용

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_nli_evaluation.log") # 로그 파일 이름 변경
    ]
)
logger = logging.getLogger(__name__)

# KLUE NLI Label Definitions (train_nli.py와 동일하게 유지)
NLI_LABELS = ["entailment", "neutral", "contradiction"]
NUM_LABELS = len(NLI_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(NLI_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(NLI_LABELS)} # 평가 시 필요
logger.info(f"Total number of KLUE-NLI labels: {NUM_LABELS}")


# Configuration parameters
ROOT_DIRECTORY = "klue_nli_results" # 학습 결과 저장 경로
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_nli_validation.json"
DATA_CACHE_DIR = "./klue_nli_eval_cache" # 평가 캐시
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 3000 # 평가 샘플 제한
INDIVIDUAL_LOG_FILENAME = "evaluation_log.json"
INDIVIDUAL_RESULT_FILENAME = "eval_results.json"
SUMMARY_FILENAME = "NLI_evaluation_summary.json" # 요약 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final" # 학습 시 저장된 모델 폴더


# --- 단일 모델 평가 함수 ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_data_list, output_dir):
    """단일 파인튜닝된 NLI 모델을 평가하고 로그 및 결과를 저장합니다."""
    logger.info(f"--- Evaluating model: {model_name} ---")
    logger.info(f"Model path: {model_path}")

    log_file_path = os.path.join(output_dir, INDIVIDUAL_LOG_FILENAME)
    result_file_path = os.path.join(output_dir, INDIVIDUAL_RESULT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)

    model = None
    tokenizer = None
    eval_results = { # 기본 결과 구조
        "model": model_name, "error": "Evaluation not completed",
        "accuracy": None, "precision_macro": None, "recall_macro": None,
        "f1_macro": None, "evaluated_samples": 0, "per_class_metrics": None
    }

    try:
        # 모델 및 토크나이저 로드
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
            else: raise ValueError("Cannot determine pad_token_id.")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=True, trust_remote_code=True)
        if model.config.pad_token_id is None: model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        device = model.device
        logger.info(f"Model and tokenizer loaded on device: {device}.")

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer for {model_name}: {e}")
        eval_results["error"] = f"Load failed: {e}"
        with open(log_file_path, 'w', encoding='utf-8') as f: json.dump([eval_results], f)
        with open(result_file_path, 'w', encoding='utf-8') as f: json.dump(eval_results, f)
        return eval_results

    # 평가 루프
    true_labels = []
    pred_labels = []
    logs = []
    num_samples_to_eval = min(MAX_EVAL_SAMPLES, len(eval_data_list))
    eval_subset = eval_data_list[:num_samples_to_eval]

    logger.info(f"Starting evaluation loop for {model_name} on {num_samples_to_eval} samples...")
    for item in tqdm(eval_subset, desc=f"Evaluating {model_name}"):
        premise = item.get("premise")
        hypothesis = item.get("hypothesis")
        gold_label_id = item.get("label") # <-- 정수 라벨 ID를 직접 가져옵니다.

        # premise, hypothesis, 또는 label ID가 없는 경우 건너뜁니다.
        if not premise or not hypothesis or gold_label_id is None:
            # logger.debug(f"Skipping sample due to missing data: {item}") # 디버깅 필요 시 주석 해제
            continue

        # 가져온 라벨 ID가 유효한 범위 내에 있는지 확인합니다 (0, 1, 2)
        if gold_label_id not in ID2LABEL:
            logger.warning(f"Skipping sample with invalid label ID: {gold_label_id} in item: {item}")
            continue

        # 이제 유효성이 확인된 정수 라벨 ID를 사용합니다.
        gold_label = gold_label_id # 변수 이름 일관성을 위해 유지하거나 gold_label_id 그대로 사용 가능

        # 토크나이징
        try: # 토크나이징 오류도 대비
            encoding = tokenizer(
                premise, hypothesis, truncation=True, max_length=MAX_LENGTH,
                padding="max_length", return_tensors="pt"
            ).to(device)
        except Exception as e_tok:
            logger.error(f"Tokenization error for item {item.get('guid', 'N/A')}: {e_tok}")
            continue # 토크나이징 실패 시 해당 샘플 스킵

        # 예측
        prediction = -1 # 초기화
        try:
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().item()

            # 예측된 ID가 유효한지 확인 (선택적이지만 안전)
            if prediction not in ID2LABEL:
                 logger.warning(f"Model produced an invalid prediction ID: {prediction}")
                 prediction = LABEL2ID.get("neutral", 1) # 잘못된 예측 시 neutral 처리

        except Exception as e_pred:
            logger.error(f"Prediction error for item {item.get('guid', 'N/A')}: {e_pred}")
            # 예측 실패 시 처리 (예: neutral 또는 특정 값 할당)
            prediction = LABEL2ID.get("neutral", 1) # 에러 시 neutral ID로

        # 로그 및 결과 리스트에 추가
        true_labels.append(gold_label) # 유효한 정수 gold_label 추가
        pred_labels.append(prediction) # 예측된 정수 prediction 추가

        logs.append({
            "guid": item.get("guid"), # guid 추가하면 추적 용이
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_label_id": gold_label, # 정수 ID
            "pred_label_id": prediction, # 정수 ID
            "gold_label_str": ID2LABEL.get(gold_label, "INVALID_GOLD"), # 문자열 변환 (로깅용)
            "pred_label_str": ID2LABEL.get(prediction, "INVALID_PRED")  # 문자열 변환 (로깅용)
        })

    # 개별 로그 저장
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f: json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save log file: {e}")

    # 메트릭 계산
    if not true_labels:
        logger.warning(f"No valid samples evaluated for {model_name}.")
        eval_results["error"] = "No valid samples evaluated"
    else:
        logger.info(f"Calculating metrics for {model_name}...")
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0)
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            true_labels, pred_labels, labels=list(range(NUM_LABELS)), average=None, zero_division=0)
        per_class_metrics = {
            ID2LABEL.get(i, f"ID_{i}"): {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": int(s)}
            for i, (p, r, f, s) in enumerate(zip(precision_per, recall_per, f1_per, support_per))}

        eval_results = {
            "model": model_name, "error": None,
            "accuracy": round(accuracy, 4), "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4), "f1_macro": round(f1_macro, 4),
            "evaluated_samples": len(true_labels), "per_class_metrics": per_class_metrics
        }
        logger.info(f"Metrics for {model_name}: Acc={eval_results['accuracy']}, F1={eval_results['f1_macro']}")

    # 개별 결과 저장
    logger.info(f"Saving individual evaluation results for {model_name} to: {result_file_path}")
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f: json.dump(eval_results, f, indent=4, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save result file: {e}")

    # 메모리 정리
    del model; del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info(f"Finished evaluation for {model_name} and cleaned up resources.")
    print("-" * 30)
    return eval_results


# --- 로그 처리 및 요약 함수 (RE 코드와 동일하게 사용 가능) ---
def process_evaluation_results(root_dir, result_filename=INDIVIDUAL_RESULT_FILENAME):
    summary_results = {}
    logger.info(f"--- Starting Result Summary Generation ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Looking for result files named: {result_filename}")
    if not os.path.isdir(root_dir): logger.error(f"Dir not found: {root_dir}"); return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            result_path = os.path.join(model_dir_path, result_filename)
            if os.path.exists(result_path):
                logger.info(f"Processing result file for model: {item_name}")
                try:
                    with open(result_path, 'r', encoding='utf-8') as f: metrics = json.load(f)
                    if isinstance(metrics, dict): summary_results[item_name] = {k: v for k, v in metrics.items() if k != 'error' and k != 'per_class_metrics'}
                    else: summary_results[item_name] = {"error": "Invalid format"}
                except Exception as e: summary_results[item_name] = {"error": str(e)}
            else:
                 if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)):
                     summary_results[item_name] = {"error": f"{result_filename} not found"}

    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving aggregated results summary to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Summary results saved to {output_path}")
    except Exception as e: logger.error(f"Failed to save summary: {e}")
    return summary_results


# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting KLUE-NLI Evaluation Script ---")
    # 1. 평가 데이터셋 로드
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            evaluation_data_list = json.load(f)
        logger.info(f"Loaded {len(evaluation_data_list)} samples for evaluation.")
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load evaluation dataset. Error: {e}"); exit()

    # 2. 루트 디렉토리 내 모델 평가
    if not os.path.isdir(ROOT_DIRECTORY):
        logger.error(f"Evaluation root directory not found: {ROOT_DIRECTORY}"); exit()

    all_model_results = {}
    for item_name in os.listdir(ROOT_DIRECTORY):
        model_dir_path = os.path.join(ROOT_DIRECTORY, item_name)
        if os.path.isdir(model_dir_path):
            fine_tuned_model_path = os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)
            if os.path.isdir(fine_tuned_model_path):
                 result = evaluate_single_model(
                     model_name=item_name, model_path=fine_tuned_model_path,
                     tokenizer_path=fine_tuned_model_path,
                     eval_data_list=evaluation_data_list, # 파이썬 리스트 전달
                     output_dir=model_dir_path)
                 if result: all_model_results[item_name] = result
            else: logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found.")

    logger.info(f"Finished evaluating all found models.")

    # 3. 최종 결과 요약 파일 생성
    final_summary = process_evaluation_results(ROOT_DIRECTORY, result_filename=INDIVIDUAL_RESULT_FILENAME)

    # 요약 결과 출력 (F1 기준 정렬)
    if final_summary:
        print("\n--- Final Evaluation Summary (NLI) ---")
        sorted_models = sorted(final_summary.items(),
                               key=lambda item: item[1].get('f1_macro', -1) if isinstance(item[1], dict) else -1,
                               reverse=True)
        for model_name, metrics in sorted_models:
            print(f"\nModel: {model_name}")
            if isinstance(metrics, dict):
                 if "error" in metrics: print(f"  Error: {metrics['error']}")
                 else:
                     # Accuracy와 Macro F1만 출력
                     print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
                     print(f"  Macro F1: {metrics.get('f1_macro', 'N/A')}")
            else: print("  Invalid metrics format.")
        print("-" * 30)

    logger.info("--- KLUE-NLI Evaluation Script Finished ---")