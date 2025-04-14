import json
import re
import logging
import os

# --- Configuration ---
INPUT_FILE_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/ch_kr.json"
OUTPUT_FILE_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/ch_kr_filtered.json"

# 영어 알파벳 문자를 찾기 위한 정규 표현식 패턴 (대소문자 구분 없음)
# ASCII 범위의 영어 알파벳만 찾습니다.
ENGLISH_LETTER_PATTERN = re.compile(r'[a-zA-Z]')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Processing Logic ---
def filter_en_ch_json(input_path, output_path, pattern):
    """
    JSON 딕셔너리 파일을 로드하여 값(중국어)에 영어가 포함된
    키-값 쌍(entry)을 제거하고 결과를 새 파일에 저장합니다.
    """
    try:
        # 입력 파일 읽기 (UTF-8 인코딩 명시)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"JSON 파일 디코딩 오류: {input_path}")
        return
    except Exception as e:
        logger.error(f"입력 파일 읽기 오류 {input_path}: {e}")
        return

    # 데이터 형식이 딕셔너리인지 확인
    if not isinstance(data, dict):
        logger.error(f"{input_path} 파일에 JSON 딕셔너리가 필요하지만, {type(data)} 타입이 발견되었습니다. 처리를 중단합니다.")
        return

    filtered_data = {}
    removed_count = 0
    total_count = len(data)

    logger.info(f"{input_path}에서 {total_count}개의 키-값 쌍 처리를 시작합니다...")

    # 각 키-값 쌍(entry)을 순회
    for key, value in data.items():
        # 키(영어)와 값(중국어)이 모두 문자열인지 확인 (안전 장치)
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning(f"키 또는 값이 문자열이 아니므로 쌍 건너뛰기: '{key}': '{value}'")
            removed_count += 1
            continue

        # 정규 표현식을 사용하여 값(중국어 위치)에 영어 문자 포함 여부 확인
        contains_english_value = pattern.search(value)

        # 값(중국어 위치)에 영어가 포함되지 *않는* 경우에만 유지
        if not contains_english_value:
            filtered_data[key] = value
        else:
            # 제거된 쌍 로깅 (필요시 주석 해제)
            logger.debug(f"값(중국어)에 영어 포함으로 쌍 제거: '{key}': '{value}'")
            removed_count += 1

    kept_count = len(filtered_data)
    logger.info(f"처리 완료. {kept_count}개 쌍 유지, {removed_count}개 쌍 제거됨.")

    # 필터링된 데이터를 새 출력 파일에 저장
    try:
        # 출력 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # UTF-8 인코딩과 ensure_ascii=False로 중국어 문자 유지, 보기 좋게 들여쓰기
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        logger.info(f"필터링된 데이터가 성공적으로 저장되었습니다: {output_path}")
    except Exception as e:
        logger.error(f"출력 파일 쓰기 오류 {output_path}: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 정의된 경로와 패턴을 사용하여 필터링 함수 실행
    filter_en_ch_json(INPUT_FILE_PATH, OUTPUT_FILE_PATH, ENGLISH_LETTER_PATTERN)