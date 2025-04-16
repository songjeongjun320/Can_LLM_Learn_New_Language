import os
import re
import json
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, # 학습 시 사용한 모델
    AutoTokenizer
)
# from sklearn.metrics import ... # 직접 계산 함수 사용
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_mrc_evaluation.log") # 로그 파일 이름 변경
    ]
)
logger = logging.getLogger(__name__)

# --- 기본 설정 ---
ROOT_DIRECTORY = "klue_mrc_results" # 학습 결과 저장 경로
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_validation.json"
DATA_CACHE_DIR = "./klue_mrc_eval_cache" # 평가 캐시
MAX_LENGTH = 1024 # 학습 시 길이와 맞춤
MAX_EVAL_SAMPLES = 1000 # 평가 샘플 제한
INDIVIDUAL_LOG_FILENAME = "evaluation_log.json"
INDIVIDUAL_RESULT_FILENAME = "eval_results.json"
SUMMARY_FILENAME = "MRC_evaluation_summary.json"
FINE_TUNED_MODEL_SUBDIR = "final_model" # 학습 시 저장된 폴더

# --- MRC evaluation metrics (from original code) ---
def compute_exact_match(prediction, ground_truth):
    if not ground_truth: return 0 # 정답 없으면 0점
    prediction = prediction.strip().lower()
    ground_truths = [str(gt).strip().lower() for gt in ground_truth] # 문자열 변환 추가
    return max(int(prediction == gt) for gt in ground_truths)

def compute_f1_score(prediction, ground_truth):
    if not ground_truth: return 0.0 # 정답 없으면 0점
    def get_tokens(s): return str(s).strip().lower().split() # 문자열 변환 추가

    prediction_tokens = get_tokens(prediction)
    f1_scores = []
    for gt in ground_truth:
        gt_tokens = get_tokens(gt)
        common_tokens = set(prediction_tokens) & set(gt_tokens)
        if not common_tokens: f1_scores.append(0.0); continue
        precision = len(common_tokens) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
        if precision + recall == 0: f1_scores.append(0.0)
        else: f1_scores.append(2 * precision * recall / (precision + recall))
    return max(f1_scores) if f1_scores else 0.0

# --- 단일 모델 평가 함수 ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_data_list, output_dir):
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
        "exact_match": None, "f1_score": None, "total_examples": 0
    }

    try:
        # 모델 및 토크나이저 로드
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
            else: raise ValueError("Cannot determine pad_token_id.")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            local_files_only=True, trust_remote_code=True)

        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
             model.config.pad_token_id = tokenizer.pad_token_id

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
    exact_match_scores = []
    f1_scores = []
    logs = []
    num_samples_to_eval = min(MAX_EVAL_SAMPLES, len(eval_data_list))
    eval_subset = eval_data_list[:num_samples_to_eval]

    logger.info(f"Starting evaluation loop for {model_name} on {num_samples_to_eval} samples...")
    for item in tqdm(eval_subset, desc=f"Evaluating {model_name}"):
        title = item.get("title", "")
        context = item.get("context", "")
        question = item.get("question", "")
        if not context or not question: continue

        ground_truth_answers = item.get("answers", {}).get("text", [""])
        # ground_truth_answers가 비어있거나 None일 경우 빈 리스트로 처리
        if not ground_truth_answers: ground_truth_answers = [""]


        prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH - 64).to(device) # 답변 길이 고려

        answer = "[GENERATION ERROR]"
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"], max_new_tokens=64, temperature=0.1,
                    num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)

            # 생성된 전체 텍스트에서 답변 부분만 추출
            full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트가 생성 텍스트 시작 부분에 있는지 확인 후 제거
            if full_generated_text.startswith(prompt):
                 answer = full_generated_text[len(prompt):].strip()
            else:
                 # 프롬프트가 없다면, 생성된 텍스트 자체를 답변으로 간주 (모델 특성일 수 있음)
                 # 또는 특정 마커 이후를 답변으로 간주하는 로직 추가 가능
                 answer = full_generated_text # 임시로 전체 텍스트 사용, 모델 출력 확인 필요
                 logger.warning(f"Generated text does not start with prompt for question: {question[:30]}...")


        except Exception as e:
            logger.error(f"Error during generation for model {model_name}: {e}")

        em_score = compute_exact_match(answer, ground_truth_answers)
        f1 = compute_f1_score(answer, ground_truth_answers)
        exact_match_scores.append(em_score)
        f1_scores.append(f1)

        logs.append({
            "title": title, "context_preview": context[:100] + "...", "question": question,
            "ground_truth": ground_truth_answers, "prediction": answer,
            "em_score": em_score, "f1_score": f1
        })

    # 개별 로그 저장
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f: json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save log file: {e}")

    # 최종 메트릭 계산
    avg_em = np.mean(exact_match_scores) if exact_match_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    eval_results = {
        "model": model_name, "error": None,
        "exact_match": round(float(avg_em), 4),
        "f1_score": round(float(avg_f1), 4),
        "evaluated_examples": len(exact_match_scores) # 실제 평가된 샘플 수
    }
    logger.info(f"Metrics for {model_name}: EM={eval_results['exact_match']}, F1={eval_results['f1_score']}")


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


# --- 로그 처리 및 요약 함수 (결과 파일 기반) ---
def process_evaluation_results(root_dir, result_filename=INDIVIDUAL_RESULT_FILENAME):
    # ... (이 함수는 KLUE-RE 스크립트의 것과 거의 동일하게 사용 가능) ...
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
                    if isinstance(metrics, dict): summary_results[item_name] = {k: v for k, v in metrics.items() if k != 'error'}
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
    logger.info("--- Starting KLUE-MRC Evaluation Script ---")
    # 1. 평가 데이터셋 로드
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            evaluation_data_list = json.load(f)
        # Dataset 객체로 변환할 필요 없음 (eval_single_model이 리스트 직접 처리)
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
        print("\n--- Final Evaluation Summary (MRC) ---")
        sorted_models = sorted(final_summary.items(),
                               key=lambda item: item[1].get('f1_score', -1) if isinstance(item[1], dict) else -1,
                               reverse=True)
        for model_name, metrics in sorted_models:
            print(f"\nModel: {model_name}")
            if isinstance(metrics, dict):
                 if "error" in metrics: print(f"  Error: {metrics['error']}")
                 else:
                     # EM과 F1만 출력 (간결하게)
                     print(f"  Exact Match: {metrics.get('exact_match', 'N/A')}")
                     print(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
            else: print("  Invalid metrics format.")
        print("-" * 30)

    logger.info("--- KLUE-MRC Evaluation Script Finished ---")