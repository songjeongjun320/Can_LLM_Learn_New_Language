import os
import json
import torch
import numpy as np
from datasets import load_dataset, Dataset # Dataset import 추가
from transformers import (
    AutoModelForCausalLM, # 학습 시 사용한 모델과 동일하게 로드
    AutoTokenizer
)
# Sequence Classification 관련 메트릭 대신 DST 메트릭 사용
# from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
# from scipy.stats import pearsonr
import logging
import re
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 기본 설정 ---
ROOT_DIRECTORY = "klue_dst_results" # 학습 결과가 저장된 루트 디렉토리
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
DATA_CACHE_DIR = "./klue_dst_eval_cache"
MAX_LENGTH = 768 # 학습 시 사용한 길이와 맞춤
MAX_EVAL_SAMPLES = 200 # 평가 샘플 수 제한 (전체 사용 시 매우 큰 값 설정 또는 슬라이싱 제거)
INDIVIDUAL_LOG_FILENAME = "dst_evaluation_log.json" # 각 모델별 로그 파일 이름
INDIVIDUAL_RESULT_FILENAME = "dst_eval_results.json" # 각 모델별 결과 파일 이름
SUMMARY_FILENAME = "DST_evaluation_summary.json" # 최종 요약 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final_model" # 학습 시 저장된 모델 하위 폴더 이름

# --- DST 평가 관련 함수 ---
def normalize_state(state):
    """상태 표현을 정규화합니다 (리스트 형태로 반환)"""
    if not state or state == "없음": return []
    if isinstance(state, str):
        # 앞뒤 공백 제거 후 쉼표+공백으로 분리
        items = [item.strip() for item in state.strip().split(",")]
        # 빈 문자열 제거
        return sorted([item for item in items if item]) # 일관성을 위해 정렬
    # 이미 리스트인 경우 정렬하여 반환
    return sorted(state)

def calculate_joint_accuracy(true_states, pred_states):
    """조인트 정확도 계산"""
    correct = 0
    total = len(true_states)
    if total == 0: return 0
    for true_state, pred_state in zip(true_states, pred_states):
        true_set = set(normalize_state(true_state))
        pred_set = set(normalize_state(pred_state))
        if true_set == pred_set:
            correct += 1
    return correct / total

def calculate_slot_f1(true_states, pred_states):
    """슬롯 F1 점수 계산"""
    true_slots_all = []
    pred_slots_all = []
    for true_state, pred_state in zip(true_states, pred_states):
        true_slots_all.extend(normalize_state(true_state))
        pred_slots_all.extend(normalize_state(pred_state))

    # 각 슬롯의 등장 횟수를 세는 대신, 집합 연산 사용
    common_slots = set(true_slots_all) & set(pred_slots_all)
    tp = len([s for s in pred_slots_all if s in common_slots]) # 예측된 슬롯 중 실제 슬롯에도 있는 것
    # 주의: 위 방식은 멀티턴에서 동일 슬롯 반복 시 잘못 계산될 수 있음
    # 정확한 계산을 위해서는 각 턴별로 TP/FP/FN 계산 후 합산 필요

    # 간편 계산 방식 (턴 무시, 전체 슬롯 대상)
    tp_simple = len(set(true_slots_all) & set(pred_slots_all)) # 겹치는 고유 슬롯 수
    fp_simple = len(set(pred_slots_all) - set(true_slots_all)) # 예측에만 있는 고유 슬롯 수
    fn_simple = len(set(true_slots_all) - set(pred_slots_all)) # 실제에만 있는 고유 슬롯 수

    precision = tp_simple / (tp_simple + fp_simple) if (tp_simple + fp_simple) > 0 else 0
    recall = tp_simple / (tp_simple + fn_simple) if (tp_simple + fn_simple) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# --- 단일 모델 평가 및 로그 저장 함수 ---
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
        "joint_accuracy": None, "slot_f1": None, "slot_precision": None,
        "slot_recall": None, "num_samples": 0
    }

    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
            else: raise ValueError("Cannot set padding token for tokenizer.")
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])
        logger.info(f"Using pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")


        model = AutoModelForCausalLM.from_pretrained( # 학습 시 사용한 모델과 동일하게
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            local_files_only=True, trust_remote_code=True)
        model.eval()
        device = model.device # 할당된 디바이스 확인
        logger.info(f"Model and tokenizer loaded successfully on device: {device}.")

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer for {model_name}: {e}")
        eval_results["error"] = f"Load failed: {e}"
        # 실패 로그 저장 (선택적)
        with open(log_file_path, 'w', encoding='utf-8') as f: json.dump([eval_results], f)
        with open(result_file_path, 'w', encoding='utf-8') as f: json.dump(eval_results, f)
        return eval_results # 에러 결과 반환

    # 평가 루프
    true_states_list = [] # 정규화된 실제 상태 리스트
    pred_states_list = [] # 정규화된 예측 상태 리스트
    logs = []
    num_samples_to_eval = min(MAX_EVAL_SAMPLES, len(eval_dataset)) # 실제 평가할 샘플 수
    eval_subset = eval_dataset.select(range(num_samples_to_eval)) # 부분집합 선택

    logger.info(f"Starting evaluation loop for {model_name} on {num_samples_to_eval} samples...")
    for item in tqdm(eval_subset, desc=f"Evaluating {model_name}"):
        prompt_text = item.get("input")
        output_text = item.get("output") # 원본 output 텍스트

        if not prompt_text or not output_text: continue

        # 실제 상태 정규화 (리스트 형태)
        true_state_normalized = normalize_state(output_text)

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        pred_state_normalized = [] # 예측 상태 기본값
        full_generated_text = "[GENERATION ERROR]" # 생성 텍스트 기본값

        try:
            with torch.no_grad():
                # 모델 생성 시 pad_token_id와 eos_token_id 명시 중요
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100, # DST 상태 길이에 맞게 조정
                    temperature=0.1,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id # 종료 토큰 설정
                )
            # 생성된 부분만 디코딩 (input 제외)
            # generate 함수는 보통 input_ids 길이 이후의 토큰만 반환하지 않음
            # 따라서 전체 디코딩 후 프롬프트 제거 필요
            full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # 프롬프트 부분 제거 시도
            completion_text = full_generated_text
            if full_generated_text.strip().startswith(prompt_text.strip()):
                 completion_text = full_generated_text[len(prompt_text):].strip()
            else:
                 # 생성된 텍스트 시작 부분이 프롬프트와 다를 경우,
                 # 혹시 "대화 상태:" 이후 부분만 생성했는지 확인
                 state_marker = "대화 상태:"
                 marker_pos = full_generated_text.find(state_marker)
                 if marker_pos != -1:
                     completion_text = full_generated_text[marker_pos + len(state_marker):].strip()
                 # else: 그대로 사용 (프롬프트 제거 실패)


            # 예측 상태 정규화
            pred_state_normalized = normalize_state(completion_text)

        except Exception as e:
            logger.error(f"Error during generation for model {model_name}: {e}")

        true_states_list.append(true_state_normalized)
        pred_states_list.append(pred_state_normalized)
        logs.append({
            "prompt": prompt_text,
            "expected_output_text": output_text, # 원본 output
            "generated_text": full_generated_text, # 모델 생성 전체 텍스트
            "parsed_completion": completion_text, # 파싱된 completion
            "true_state_normalized": true_state_normalized,
            "pred_state_normalized": pred_state_normalized
        })

    # 개별 로그 저장
    logger.info(f"Saving individual evaluation log for {model_name} to: {log_file_path}")
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save log file for {model_name}: {e}")

    # 메트릭 계산
    logger.info(f"Calculating metrics for {model_name}...")
    joint_accuracy = calculate_joint_accuracy(true_states_list, pred_states_list)
    slot_metrics = calculate_slot_f1(true_states_list, pred_states_list)

    eval_results = {
        "model": model_name,
        "error": None, # 성공 시 None
        "joint_accuracy": round(float(joint_accuracy), 4),
        "slot_f1": round(float(slot_metrics["f1"]), 4),
        "slot_precision": round(float(slot_metrics["precision"]), 4),
        "slot_recall": round(float(slot_metrics["recall"]), 4),
        "num_samples": num_samples_to_eval
    }

    # 개별 결과 저장
    logger.info(f"Saving individual evaluation results for {model_name} to: {result_file_path}")
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save result file for {model_name}: {e}")


    # 메모리 정리
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Finished evaluation for {model_name} and cleaned up resources.")
    print("-" * 30)

    return eval_results # 요약을 위해 결과 반환

# --- 로그 처리 및 요약 함수 (DST 메트릭 사용) ---
def process_evaluation_results(root_dir, result_filename=INDIVIDUAL_RESULT_FILENAME):
    """루트 디렉토리 내의 각 모델 결과 파일을 읽어 요약합니다."""
    summary_results = {}
    logger.info(f"--- Starting Result Summary Generation ---")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Looking for result files named: {result_filename}")

    if not os.path.isdir(root_dir):
        logger.error(f"Root directory not found: {root_dir}")
        return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            result_path = os.path.join(model_dir_path, result_filename)
            if os.path.exists(result_path):
                logger.info(f"Processing result file for model: {item_name}")
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        # 결과 파일은 딕셔너리 형태라고 가정
                        metrics = json.load(f)
                    if isinstance(metrics, dict):
                         # 에러 필드 제외하고 메트릭만 요약에 포함 (선택적)
                         summary_results[item_name] = {k: v for k, v in metrics.items() if k != 'error'}
                    else:
                         logger.warning(f"Result file {result_path} is not a dictionary. Skipping.")
                         summary_results[item_name] = {"error": "Invalid result file format"}

                except Exception as e:
                    logger.error(f"Error processing result file {result_path}: {e}")
                    summary_results[item_name] = {"error": str(e)}
            else:
                 # 결과 파일이 없는 경우 (평가 실패 등)
                 if os.path.exists(os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)):
                     logger.warning(f"Result file '{result_filename}' not found in model directory: {item_name}")
                     summary_results[item_name] = {"error": f"{result_filename} not found"}

    # 요약 결과 저장
    output_path = os.path.join(root_dir, SUMMARY_FILENAME)
    logger.info(f"Saving aggregated results summary to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Summary results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save summary results: {e}")

    return summary_results

# --- 메인 실행 함수 ---
if __name__ == "__main__":
    logger.info("--- Starting KLUE-DST Evaluation Script ---")

    # 1. 평가 데이터셋 로드 (json.load 사용)
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            evaluation_data_list = json.load(f)
        evaluation_dataset = Dataset.from_list(evaluation_data_list)
        logger.info(f"Loaded {len(evaluation_dataset)} samples for evaluation.")
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load evaluation dataset. Error: {e}")
        exit()

    # 2. 루트 디렉토리 내 모델 평가
    if not os.path.isdir(ROOT_DIRECTORY):
        logger.error(f"Evaluation root directory not found: {ROOT_DIRECTORY}")
        exit()

    all_model_results = {} # 모든 모델의 결과 저장용
    for item_name in os.listdir(ROOT_DIRECTORY):
        model_dir_path = os.path.join(ROOT_DIRECTORY, item_name)
        if os.path.isdir(model_dir_path):
            fine_tuned_model_path = os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)
            if os.path.isdir(fine_tuned_model_path):
                 # 평가 함수 호출하고 결과 저장
                 result = evaluate_single_model(
                     model_name=item_name,
                     model_path=fine_tuned_model_path,
                     tokenizer_path=fine_tuned_model_path,
                     eval_dataset=evaluation_dataset, # 전체 데이터셋 전달
                     output_dir=model_dir_path
                 )
                 if result: # 평가가 성공적으로 완료되면 결과 저장
                     all_model_results[item_name] = result
            else:
                logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found.")

    logger.info(f"Finished evaluating all found models.")

    # 3. 최종 결과 요약 파일 생성 (선택적, process_evaluation_results 함수 사용)
    # 위 루프에서 이미 각 모델 결과를 저장했으므로, 요약은 단순히 이 파일들을 읽어오는 것
    logger.info("Generating final summary file from individual results...")
    final_summary = process_evaluation_results(ROOT_DIRECTORY, result_filename=INDIVIDUAL_RESULT_FILENAME)

    # 요약 결과 출력 (선택 사항)
    if final_summary:
        print("\n--- Final Evaluation Summary ---")
        for model_name, metrics in final_summary.items():
            print(f"\nModel: {model_name}")
            if "error" in metrics:
                print(f"  Error: {metrics['error']}")
            elif metrics: # metrics가 None이 아닐 경우
                 for metric_name, value in metrics.items():
                    # 모델 이름은 이미 출력했으므로 제외
                    if metric_name != 'model':
                         print(f"  {metric_name}: {value}")
            else:
                 print("  No metrics available.")
        print("-" * 30)

    logger.info("--- KLUE-DST Evaluation Script Finished ---")