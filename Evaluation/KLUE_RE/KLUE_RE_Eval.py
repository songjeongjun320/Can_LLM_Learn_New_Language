import os
import json
import torch
import numpy as np
from datasets import load_dataset, Dataset # Dataset import 추가
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_re_evaluation.log") # 로그 파일 이름 변경
    ]
)
logger = logging.getLogger(__name__)

# KLUE RE Label Definitions (train_re.py와 동일하게 유지)
RE_LABELS = [
    "no_relation","org:dissolved","org:place_of_headquarters","org:alternate_names",
    "org:member_of","org:political/religious_affiliation","org:product",
    "org:founded_by","org:top_members/employees","org:number_of_employees/members",
    "per:date_of_birth","per:date_of_death","per:place_of_birth","per:place_of_death",
    "per:place_of_residence","per:origin","per:employee_of","per:schools_attended",
    "per:alternate_names","per:parents","per:children","per:siblings","per:spouse",
    "per:other_family","per:colleagues","per:product","per:religion","per:title"
]
NUM_LABELS = len(RE_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(RE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(RE_LABELS)}
logger.info(f"Total number of KLUE-RE labels: {NUM_LABELS}")

# Configuration parameters
ROOT_DIRECTORY = "klue_re_results" # 학습 결과가 저장된 루트 디렉토리
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_validation.json"
DATA_CACHE_DIR = "./klue_re_eval_cache" # 평가용 캐시
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200 # 평가 샘플 수 제한
INDIVIDUAL_LOG_FILENAME = "evaluation_log.json"
INDIVIDUAL_RESULT_FILENAME = "eval_results.json"
SUMMARY_FILENAME = "RE_evaluation_summary.json" # 요약 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final_model" # 학습 시 저장된 모델 폴더

# --- 단일 모델 평가 함수 ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_dataset, output_dir):
    """단일 파인튜닝된 모델을 평가하고 로그 및 결과를 저장합니다."""
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
        # 모델 및 토크나이저 로드 (Sequence Classification)
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        # 패딩 토큰 로드 확인 (학습 시 저장되었으므로 보통 문제 없음)
        if tokenizer.pad_token_id is None:
            logger.warning("Loaded tokenizer missing pad_token_id, attempting to set from eos.")
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
            else: raise ValueError("Cannot determine pad_token_id for loaded tokenizer.")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])


        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS, torch_dtype=torch.bfloat16,
            device_map="auto", local_files_only=True, trust_remote_code=True)

        # 모델 config의 pad_token_id도 확인 및 설정
        if model.config.pad_token_id is None:
             model.config.pad_token_id = tokenizer.pad_token_id

        model.eval()
        device = model.device
        logger.info(f"Model and tokenizer loaded successfully on device: {device}.")

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
    num_samples_to_eval = min(MAX_EVAL_SAMPLES, len(eval_dataset))
    eval_subset = eval_dataset.select(range(num_samples_to_eval))

    logger.info(f"Starting evaluation loop for {model_name} on {num_samples_to_eval} samples...")
    for item in tqdm(eval_subset, desc=f"Evaluating {model_name}"):
        sentence = item.get("sentence")
        subject_entity = item.get("subject_entity")
        object_entity = item.get("object_entity")
        label_id = item.get("label") # 숫자 레이블 ID

        if not all([sentence, subject_entity, object_entity, label_id is not None]): continue
        if not isinstance(label_id, int) or not (0 <= label_id < NUM_LABELS): continue

        gold_label = label_id # 실제 레이블 ID

        # 특수 마커 삽입 (학습 시와 동일 로직)
        sub_start = subject_entity.get('start_idx'); sub_end = subject_entity.get('end_idx')
        obj_start = object_entity.get('start_idx'); obj_end = object_entity.get('end_idx')
        if None in [sub_start, sub_end, obj_start, obj_end]: continue
        try:
             if sub_start < obj_start:
                 marked_sentence = (sentence[:sub_start] + "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                                    sentence[sub_end + 1:obj_start] + "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                                    sentence[obj_end + 1:])
             else:
                 marked_sentence = (sentence[:obj_start] + "[OBJ]" + sentence[obj_start:obj_end + 1] + "[/OBJ]" +
                                    sentence[obj_end + 1:sub_start] + "[SUBJ]" + sentence[sub_start:sub_end + 1] + "[/SUBJ]" +
                                    sentence[sub_end + 1:])
        except: continue

        # 토크나이징
        encoding = tokenizer(marked_sentence, truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt").to(device)

        # 예측
        prediction = -1 # 기본값
        try:
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().item()
        except Exception as e:
            logger.error(f"Error during prediction for model {model_name}: {e}")

        true_labels.append(gold_label)
        pred_labels.append(prediction if prediction != -1 else LABEL2ID.get("no_relation", 0)) # 에러 시 no_relation으로 처리

        logs.append({
            "sentence": sentence, "marked_sentence": marked_sentence,
            "subject_entity": subject_entity.get("word"), "object_entity": object_entity.get("word"),
            "gold_label": ID2LABEL.get(gold_label, "UNK"), "pred_label": ID2LABEL.get(prediction, "ERROR")
        })

    # 개별 로그 저장
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save log file for {model_name}: {e}")

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
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save result file for {model_name}: {e}")

    # 메모리 정리
    del model; del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info(f"Finished evaluation for {model_name} and cleaned up resources.")
    print("-" * 30)
    return eval_results

# --- 로그 처리 및 요약 함수 (결과 파일 기반) ---
def process_evaluation_results(root_dir, result_filename=INDIVIDUAL_RESULT_FILENAME):
    summary_results = {}
    logger.info(f"--- Starting Result Summary Generation ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Looking for result files named: {result_filename}")
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}"); return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            result_path = os.path.join(model_dir_path, result_filename)
            if os.path.exists(result_path):
                logger.info(f"Processing result file for model: {item_name}")
                try:
                    with open(result_path, 'r', encoding='utf-8') as f: metrics = json.load(f)
                    if isinstance(metrics, dict): summary_results[item_name] = {k: v for k, v in metrics.items() if k != 'error' and k != 'per_class_metrics'} # 요약에는 전체 메트릭만
                    else: summary_results[item_name] = {"error": "Invalid result format"}
                except Exception as e: summary_results[item_name] = {"error": str(e)}
            else:
                 if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)):
                     summary_results[item_name] = {"error": f"{result_filename} not found"}

    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving aggregated results summary to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Summary results successfully saved to {output_path}")
    except Exception as e: logger.error(f"Failed to save summary results: {e}")
    return summary_results

# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting KLUE-RE Evaluation Script ---")
    # 1. 평가 데이터셋 로드
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        # 파일 전체가 JSON 리스트 형태라고 가정
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            evaluation_data_list = json.load(f)
        evaluation_dataset = Dataset.from_list(evaluation_data_list)
        logger.info(f"Loaded {len(evaluation_dataset)} samples for evaluation.")
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
                     eval_dataset=evaluation_dataset, output_dir=model_dir_path)
                 if result: all_model_results[item_name] = result # 결과 저장
            else: logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found.")

    logger.info(f"Finished evaluating all found models.")

    # 3. 최종 결과 요약 파일 생성
    final_summary = process_evaluation_results(ROOT_DIRECTORY, result_filename=INDIVIDUAL_RESULT_FILENAME)

    # 요약 결과 출력
    if final_summary:
        print("\n--- Final Evaluation Summary ---")
        # F1 스코어 기준으로 정렬하여 출력 (값이 있는 경우)
        sorted_models = sorted(final_summary.items(),
                               key=lambda item: item[1].get('f1_macro', -1) if isinstance(item[1], dict) else -1,
                               reverse=True)
        for model_name, metrics in sorted_models:
            print(f"\nModel: {model_name}")
            if isinstance(metrics, dict):
                 if "error" in metrics: print(f"  Error: {metrics['error']}")
                 else:
                     for metric_name, value in metrics.items():
                         if metric_name != 'model': print(f"  {metric_name}: {value}")
            else: print("  Invalid metrics format.")
        print("-" * 30)

    logger.info("--- KLUE-RE Evaluation Script Finished ---")