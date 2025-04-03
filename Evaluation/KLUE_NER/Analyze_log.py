# Entity-level Macro F1
# Character-level Macro F1
# 를 도출해내는 코드

import os
import json
import logging
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import f1_score as entity_f1_score
from seqeval.scheme import IOB2  # KLUE NER은 IOB2 스킴을 사용합니다.

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# --- 설정 ---
BASE_RESULTS_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/KLUE_NER/klue_ner_results"
OUTPUT_FILE = os.path.join(BASE_RESULTS_PATH, "analyzed_log.json")
EVAL_LOG_FILENAME = "evaluation_log.json"
# --- 설정 끝 ---

def calculate_metrics(gold_labels_list, pred_labels_list):
    """
    주어진 골드 라벨과 예측 라벨 리스트로부터 Entity-level 및 Character-level Macro F1을 계산합니다.

    Args:
        gold_labels_list (list[list[str]]): 실제 라벨 시퀀스들의 리스트. 예: [['B-PS', 'I-PS', 'O'], ['B-LC', 'O']]
        pred_labels_list (list[list[str]]): 예측된 라벨 시퀀스들의 리스트.

    Returns:
        tuple: (entity_macro_f1, char_macro_f1)
               계산 중 오류 발생 시 (None, None) 반환
    """
    entity_macro_f1 = None
    char_macro_f1 = None

    # 1. Entity-level Macro F1 (seqeval 사용)
    try:
        # strict=True와 mode='strict'는 동일하며, 스킴(IOB2 등)을 엄격하게 검사합니다.
        # mode=None이면 스킴을 자동으로 감지하려고 시도합니다.
        # 여기서는 KLUE NER이 IOB2임을 알고 있으므로 명시적으로 지정합니다.
        entity_macro_f1 = entity_f1_score(
            gold_labels_list, pred_labels_list, average="macro", mode='strict', scheme=IOB2, zero_division=0
        )
        logger.debug("Entity-level Macro F1 calculation successful.")
    except Exception as e:
        logger.error(f"Error calculating entity-level macro F1: {e}")
        # seqeval은 라벨 형식이 맞지 않으면 오류를 발생시킬 수 있습니다. (예: 'I-PS' 앞에 'B-PS'나 'I-PS'가 없는 경우)
        # 로그 파일 포맷에 따라서는 불완전한 예측이 있을 수 있으므로 예외 처리
        pass # 오류 발생 시 None으로 유지

    # 2. Character-level Macro F1 (sklearn 사용)
    try:
        # sklearn은 평탄화된 리스트를 요구합니다.
        flat_gold_labels = [label for seq in gold_labels_list for label in seq]
        flat_pred_labels = [label for seq in pred_labels_list for label in seq]

        if not flat_gold_labels or not flat_pred_labels:
             logger.warning("Empty label lists found for character-level calculation, skipping.")
             return entity_macro_f1, None # 문자 레벨 계산 불가

        # 모든 가능한 라벨 집합 구하기 (sklearn에 필요)
        all_labels = sorted(list(set(flat_gold_labels + flat_pred_labels)))

        precision, recall, f1, support = precision_recall_fscore_support(
            flat_gold_labels,
            flat_pred_labels,
            average="macro",
            labels=all_labels, # 가능한 모든 라벨 명시
            zero_division=0 # F1 계산 시 분모가 0이 되는 경우 0으로 처리
        )
        char_macro_f1 = f1
        logger.debug("Character-level Macro F1 calculation successful.")
    except Exception as e:
        logger.error(f"Error calculating character-level macro F1: {e}")
        pass # 오류 발생 시 None으로 유지

    return entity_macro_f1, char_macro_f1

def analyze_evaluation_logs(base_path, output_file):
    """
    지정된 경로의 하위 폴더에서 evaluation_log.json 파일을 찾아 분석하고 결과를 저장합니다.
    """
    logger.info(f"Starting analysis of evaluation logs in: {base_path}")
    all_results = {}

    # base_path 아래의 모든 항목(파일 및 디렉토리)을 가져옵니다.
    try:
        sub_items = os.listdir(base_path)
    except FileNotFoundError:
        logger.error(f"Base path not found: {base_path}")
        return

    for item_name in sub_items:
        item_path = os.path.join(base_path, item_name)

        # 디렉토리인지 확인합니다.
        if os.path.isdir(item_path):
            model_name = item_name # 디렉토리 이름을 모델 이름으로 사용
            log_file_path = os.path.join(item_path, EVAL_LOG_FILENAME)

            if os.path.exists(log_file_path):
                logger.info(f"Processing log file for model: {model_name} at {log_file_path}")
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)

                    if not isinstance(log_data, list) or not log_data:
                        logger.warning(f"Log file {log_file_path} is empty or not a list. Skipping.")
                        continue

                    # 모든 문장의 gold와 pred 라벨을 모읍니다.
                    all_gold_labels = []
                    all_pred_labels = []
                    valid_entry_count = 0
                    for entry in log_data:
                        if isinstance(entry, dict) and 'gold_labels' in entry and 'pred_labels' in entry:
                            # seqeval은 리스트의 리스트 형태를 기대합니다.
                            all_gold_labels.append(entry['gold_labels'])
                            all_pred_labels.append(entry['pred_labels'])
                            valid_entry_count += 1
                        else:
                             logger.warning(f"Skipping invalid entry in {log_file_path}: {entry}")

                    if valid_entry_count > 0:
                        logger.info(f"Calculating metrics for {model_name} based on {valid_entry_count} entries.")
                        entity_f1, char_f1 = calculate_metrics(all_gold_labels, all_pred_labels)

                        all_results[model_name] = {
                            "entity_level_macro_f1": entity_f1,
                            "character_level_macro_f1": char_f1
                        }
                        logger.info(f"Metrics for {model_name}: Entity Macro F1={entity_f1:.4f if entity_f1 is not None else 'N/A'}, Char Macro F1={char_f1:.4f if char_f1 is not None else 'N/A'}")
                    else:
                         logger.warning(f"No valid entries found in {log_file_path} to calculate metrics.")


                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from {log_file_path}. Skipping.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing {log_file_path}: {e}")

            else:
                logger.warning(f"Evaluation log file not found for model {model_name} at: {log_file_path}")
        else:
             logger.debug(f"Skipping non-directory item: {item_name}")


    # 최종 결과를 JSON 파일로 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Analysis complete. Results saved to: {output_file}")
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving results: {e}")


if __name__ == "__main__":
    analyze_evaluation_logs(BASE_RESULTS_PATH, OUTPUT_FILE)