import os
import json
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification, # 학습 시 사용한 모델
    AutoTokenizer,
    DataCollatorForTokenClassification # 평가 시에도 필요
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_fscore_support
from seqeval.scheme import IOB2 # 또는 BIOES 등 데이터 형식에 맞는 스킴 지정 가능 (기본값 CONLL)
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset # Subset, DataLoader 추가
from datasets import load_dataset # 필요한 경우 사용

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_ner_evaluation.log") # 로그 파일 이름 변경
    ]
)
logger = logging.getLogger(__name__)

# KLUE NER Label Definitions (train_ner.py와 동일하게 유지)
# KLUE NER Label Definitions
NER_TAGS = [
    "B-LC",     # 0
    "I-LC",     # 1
    "B-DT",     # 2
    "I-DT",     # 3
    "B-OG",     # 4
    "I-OG",     # 5
    "B-PS",     # 6
    "I-PS",     # 7
    "B-QT",     # 8
    "I-QT",     # 9
    "B-TI",     # 10
    "I-TI",     # 11
    "O"         # 12
]
NUM_LABELS = len(NER_TAGS)
LABEL2ID = {label: idx for idx, label in enumerate(NER_TAGS)}
ID2LABEL = {idx: label for idx, label in enumerate(NER_TAGS)} # 평가 시 필요
logger.info(f"Total number of KLUE-NER labels: {NUM_LABELS}")

# Configuration parameters
ROOT_DIRECTORY = "klue_ner_results" # 학습 결과 저장 경로
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_ner_validation.json" # 평가용 JSON
DATA_CACHE_DIR = "./klue_ner_eval_cache" # 평가 캐시
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200 # 평가 샘플 제한
INDIVIDUAL_LOG_FILENAME = "evaluation_log.json"
INDIVIDUAL_RESULT_FILENAME = "eval_results.json"
SUMMARY_FILENAME = "NER_evaluation_summary.json" # 요약 파일 이름
FINE_TUNED_MODEL_SUBDIR = "final_merged" # 학습 시 저장된 폴더

# Custom Dataset for KLUE-NER (train_ner.py와 동일하게 사용)
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        logger.info(f"Loading NER dataset from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.data)} samples for NER")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item.get("tokens", [])
        ner_tags = item.get("ner_tags", [])
        if not tokens or not ner_tags or len(tokens) != len(ner_tags):
             if idx > 0: return self.__getitem__(0)
             else: raise ValueError(f"Invalid sample at index {idx}")

        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                                  max_length=self.max_length, padding="max_length", return_tensors="pt")
        word_ids = encoding.word_ids(batch_index=0)
        labels = [-100] * len(encoding["input_ids"][0])
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None: continue
            if word_idx != previous_word_idx:
                if word_idx < len(ner_tags): labels[i] = ner_tags[word_idx]
            previous_word_idx = word_idx
        return {"input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(labels, dtype=torch.long)}

# --- 단일 모델 평가 함수 ---
def evaluate_single_model(model_name, model_path, tokenizer_path, eval_dataset, output_dir):
    """단일 파인튜닝된 NER 모델을 평가하고 로그 및 결과를 저장합니다."""
    logger.info(f"--- Evaluating model: {model_name} ---")
    logger.info(f"Model path: {model_path}")

    log_file_path = os.path.join(output_dir, INDIVIDUAL_LOG_FILENAME)
    result_file_path = os.path.join(output_dir, INDIVIDUAL_RESULT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)

    model = None
    tokenizer = None
    eval_results = { # 기본 결과 구조
        "model": model_name, "error": "Evaluation not completed",
        "entity_macro_f1": None, "entity_macro_precision": None, "entity_macro_recall": None, # Entity Macro
        "token_micro_f1": None, # Token Micro (참고용)
        "evaluated_samples": 0, "classification_report": None
    }
    try:
        # 모델 및 토크나이저 로드
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True, add_prefix_space=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
            else: raise ValueError("Cannot determine pad_token_id.")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.decode([tokenizer.pad_token_id])

        model = AutoModelForTokenClassification.from_pretrained(
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

    # 평가 데이터셋 준비 (Subset 적용)
    num_samples_to_eval = min(MAX_EVAL_SAMPLES, len(full_eval_dataset))
    eval_subset = Subset(full_eval_dataset, range(num_samples_to_eval))
    logger.info(f"Created evaluation subset with {len(eval_subset)} samples.")

    # 평가 데이터 로더 생성
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_subset, batch_size=16, collate_fn=data_collator)

    # --- seqeval을 위한 리스트 초기화 ---
    all_true_sequences = [] # 예: [['O', 'B-PS', 'I-PS'], ['B-LC', 'O']]
    all_pred_sequences = [] # 위와 동일 형식

    logger.info(f"Starting evaluation loop for {model_name} on {len(eval_dataset)} samples...")
    for batch in tqdm(eval_dataloader, desc=f"Predicting {model_name}"):
        labels = batch.pop("labels").to(device)
        input_ids = batch.pop("input_ids").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # --- 배치 내 각 시퀀스 처리 (seqeval 형식으로 변환) ---
        for i in range(labels_np.shape[0]):
            true_sequence = []
            pred_sequence = []
            for j in range(labels_np.shape[1]):
                if labels_np[i, j] != -100: # 패딩/서브워드 제외
                    true_label_id = labels_np[i, j]
                    pred_label_id = predictions_np[i, j]
                    # ID를 문자열 태그로 변환 (O 태그 포함)
                    true_sequence.append(ID2LABEL.get(true_label_id, "O"))
                    pred_sequence.append(ID2LABEL.get(pred_label_id, "O"))
            # 유효한 시퀀스만 추가
            if true_sequence: # 빈 시퀀스 방지
                all_true_sequences.append(true_sequence)
                all_pred_sequences.append(pred_sequence)
    # --- 예측 루프 종료 ---

    # --- 메트릭 계산 (seqeval 사용) ---
    if not all_true_sequences:
        logger.warning(f"No valid sequences found for evaluation for {model_name}.")
        eval_results["error"] = "No valid sequences for metric calculation"
    else:
        logger.info(f"Calculating entity-level metrics using seqeval for {model_name}...")
        # Entity-Level Macro F1, Precision, Recall
        entity_macro_f1 = seqeval_f1_score(all_true_sequences, all_pred_sequences, average="macro", mode='strict', scheme=IOB2, zero_division=0)
        entity_macro_precision = seqeval_precision_score(all_true_sequences, all_pred_sequences, average="macro", mode='strict', scheme=IOB2, zero_division=0)
        entity_macro_recall = seqeval_recall_score(all_true_sequences, all_pred_sequences, average="macro", mode='strict', scheme=IOB2, zero_division=0)

        # Token-Level Micro F1 (참고용)
        # seqeval은 토큰 레벨 정확도와 유사한 micro F1도 제공
        token_micro_f1 = seqeval_f1_score(all_true_sequences, all_pred_sequences, average="micro", mode='strict', scheme=IOB2, zero_division=0)

        # Classification Report (텍스트)
        report_str = classification_report(all_true_sequences, all_pred_sequences, mode='strict', scheme=IOB2, zero_division=0, output_dict=False)
        logger.info("Classification Report:\n" + report_str)
        # Classification Report (딕셔너리, 저장용)
        report_dict = classification_report(all_true_sequences, all_pred_sequences, mode='strict', scheme=IOB2, zero_division=0, output_dict=True)


        eval_results = {
            "model": model_name, "error": None,
            "entity_macro_f1": round(entity_macro_f1, 4),
            "entity_macro_precision": round(entity_macro_precision, 4),
            "entity_macro_recall": round(entity_macro_recall, 4),
            "token_micro_f1": round(token_micro_f1, 4), # Token micro f1
            "evaluated_samples": len(all_true_sequences), # 평가된 샘플(문장) 수
            "classification_report": report_dict # 상세 리포트 저장
        }
        logger.info(f"Metrics for {model_name}: Entity Macro F1={eval_results['entity_macro_f1']}, Token Micro F1={eval_results['token_micro_f1']}")

    # --- 개별 결과 저장 ---
    logger.info(f"Saving individual evaluation results for {model_name} to: {result_file_path}")
    try:
        # JSON 직렬화를 위해 NumPy 타입을 Python 기본 타입으로 변환 (필요시)
        def convert_numpy_types(obj):
             if isinstance(obj, np.integer): return int(obj)
             elif isinstance(obj, np.floating): return float(obj)
             elif isinstance(obj, np.ndarray): return obj.tolist()
             elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
             elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
             return obj
        serializable_results = convert_numpy_types(eval_results)

        with open(result_file_path, 'w', encoding='utf-8') as f:
             json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    except Exception as e: logger.error(f"Failed to save result file: {e}")

    del model; del tokenizer; del eval_dataloader; del batch
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
    if not os.path.isdir(root_dir): logger.error(f"Dir not found: {root_dir}"); return None

    for item_name in os.listdir(root_dir):
        model_dir_path = os.path.join(root_dir, item_name)
        if os.path.isdir(model_dir_path):
            result_path = os.path.join(model_dir_path, result_filename)
            if os.path.exists(result_path):
                logger.info(f"Processing result file for model: {item_name}")
                try:
                    with open(result_path, 'r', encoding='utf-8') as f: metrics = json.load(f)
                    if isinstance(metrics, dict):
                        # 요약에는 주요 메트릭만 포함
                        summary_results[item_name] = {
                            k: v for k, v in metrics.items()
                            if k != 'error' and k != 'classification_report'
                        }
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
    logger.info("--- Starting KLUE-NER Evaluation Script ---")
    # 1. 평가 데이터셋 로드 (NERDataset 사용)
    logger.info(f"Loading evaluation dataset from: {JSON_VAL_DATASET_PATH}")
    try:
        # 평가 시에도 NERDataset을 사용하여 토크나이징 및 레이블 정렬 로직 재사용
        # 단, 평가용 데이터셋 로드 시 tokenizer가 필요하므로 루프 안에서 처리해야 함
        # 여기서는 일단 파일 경로만 확인
        if not os.path.exists(JSON_VAL_DATASET_PATH):
             raise FileNotFoundError(f"Validation file not found: {JSON_VAL_DATASET_PATH}")
        logger.info("Validation dataset file exists.")
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to check evaluation dataset file. Error: {e}"); exit()

    # 2. 루트 디렉토리 내 모델 평가
    if not os.path.isdir(ROOT_DIRECTORY):
        logger.error(f"Evaluation root directory not found: {ROOT_DIRECTORY}"); exit()

    all_model_results = {}
    for item_name in os.listdir(ROOT_DIRECTORY):
        model_dir_path = os.path.join(ROOT_DIRECTORY, item_name)
        if os.path.isdir(model_dir_path):
            fine_tuned_model_path = os.path.join(model_dir_path, FINE_TUNED_MODEL_SUBDIR)
            if os.path.isdir(fine_tuned_model_path):
                 try:
                     # 평가 시 사용할 임시 토크나이저 로드 (NERDataset 초기화 위해)
                     temp_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, local_files_only=True, trust_remote_code=True, add_prefix_space=True)
                     # 평가 데이터셋 로드 (NERDataset 사용)
                     evaluation_dataset = NERDataset(JSON_VAL_DATASET_PATH, temp_tokenizer, max_length=MAX_LENGTH)
                     # 평가 샘플 제한 (Subset 사용)
                     num_samples = min(MAX_EVAL_SAMPLES, len(evaluation_dataset))
                     eval_subset = Subset(evaluation_dataset, range(num_samples))

                     result = evaluate_single_model(
                         model_name=item_name, model_path=fine_tuned_model_path,
                         tokenizer_path=fine_tuned_model_path, # 실제 평가는 이 경로 사용
                         eval_dataset=eval_subset, # Subset 전달
                         output_dir=model_dir_path)
                     if result: all_model_results[item_name] = result
                     del temp_tokenizer # 임시 토크나이저 삭제
                 except Exception as eval_e:
                     logger.error(f"Error during evaluation setup or execution for {item_name}: {eval_e}")
                     all_model_results[item_name] = {"error": str(eval_e)}
            else: logger.warning(f"Skipping '{item_name}': Subdir '{FINE_TUNED_MODEL_SUBDIR}' not found.")

    logger.info(f"Finished evaluating all found models.")

    # 3. 최종 결과 요약 파일 생성
    final_summary = process_evaluation_results(ROOT_DIRECTORY, result_filename=INDIVIDUAL_RESULT_FILENAME)

    # 요약 결과 출력 (F1 기준 정렬)
    if final_summary:
        print("\n--- Final Evaluation Summary (NER - Micro F1) ---")
        sorted_models = sorted(final_summary.items(),
                               key=lambda item: item[1].get('f1_micro', -1) if isinstance(item[1], dict) else -1,
                               reverse=True)
        for model_name, metrics in sorted_models:
            print(f"\nModel: {model_name}")
            if isinstance(metrics, dict):
                 if "error" in metrics: print(f"  Error: {metrics['error']}")
                 else:
                     print(f"  F1 (Micro): {metrics.get('f1_micro', 'N/A')}")
                     # print(f"  Precision (Micro): {metrics.get('precision_micro', 'N/A')}")
                     # print(f"  Recall (Micro): {metrics.get('recall_micro', 'N/A')}")
            else: print("  Invalid metrics format.")
        print("-" * 30)

    logger.info("--- KLUE-NER Evaluation Script Finished ---")