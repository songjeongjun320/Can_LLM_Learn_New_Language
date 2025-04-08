import json
import time
from deep_translator import GoogleTranslator # 또는 다른 번역기 (MyMemoryTranslator 등)
import os

# --- 설정 ---
# 입력 파일 경로 (영어:한글 데이터가 있는 파일)
# 이전 단계에서 생성된 'swapped_filtered_translated_words.json' 파일을 사용합니다.
input_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/swapped_filtered_translated_words.json'

# 출력 파일 경로 (영어:중국어 번역 결과를 저장할 파일)
output_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/english_to_chinese_translations.json'

# 번역 언어 설정
source_language = 'en'       # 원본 언어 (영어)
target_language = 'zh-CN'    # 대상 언어 (중국어 간체)

# API 요청 간 딜레이 (초 단위) - 너무 짧으면 IP 차단 위험
delay_between_requests = 0.6
# --- ---- ---

english_words_to_translate = []
english_chinese_data = {}
translation_errors = {} # 번역 실패한 단어 기록

# --- 1. 입력 JSON 파일에서 영어 단어 목록 읽기 ---
print(f"입력 파일 로드 중: {input_file_path}")
try:
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        # JSON 파일 로드 (영어:한글 형태)
        data = json.load(infile)
        # 딕셔너리의 키(영어 단어)들을 리스트로 추출
        english_words_to_translate = list(data.keys())
        print(f"성공: {len(english_words_to_translate)}개의 영어 단어를 로드했습니다.")

except FileNotFoundError:
    print(f"오류: 입력 파일 '{input_file_path}'을(를) 찾을 수 없습니다.")
    exit() # 파일 없으면 종료
except json.JSONDecodeError:
    print(f"오류: 입력 파일 '{input_file_path}'이(가) 유효한 JSON 형식이 아닙니다.")
    exit() # 잘못된 형식이면 종료
except Exception as e:
    print(f"입력 파일을 읽는 중 예상치 못한 오류 발생: {e}")
    exit() # 기타 오류 발생 시 종료

# --- 2. 번역기 초기화 ---
print("번역기를 초기화합니다...")
try:
    translator = GoogleTranslator(source=source_language, target=target_language)
    print("번역기 초기화 완료.")
except Exception as e:
    print(f"번역기 초기화 중 오류 발생: {e}")
    exit() # 초기화 실패 시 종료

# --- 3. 각 영어 단어를 중국어로 번역 ---
total_words = len(english_words_to_translate)
print(f"총 {total_words}개 단어 번역 시작 (대상 언어: {target_language})...")
print("-" * 30)

for i, word in enumerate(english_words_to_translate):
    try:
        # 번역 실행
        translated_word = translator.translate(word)

        # 번역 결과 유효성 검사 (None 이거나 비어있는 경우 처리)
        if translated_word and translated_word.strip():
            english_chinese_data[word] = translated_word.strip() # 앞뒤 공백 제거
            print(f"({i+1}/{total_words}) 번역 성공: '{word}' -> '{translated_word.strip()}'")
        else:
            # 번역 결과가 없거나 비어있는 경우
            translation_errors[word] = "Empty or None result"
            print(f"({i+1}/{total_words}) 번역 경고: '{word}' -> 번역 결과가 비어있습니다.")

    except Exception as e:
        # 개별 단어 번역 중 오류 발생 시 기록하고 계속 진행
        error_message = str(e)
        translation_errors[word] = error_message
        print(f"({i+1}/{total_words}) 번역 오류: '{word}' -> {error_message}")

    finally:
        # API 요청 사이에 딜레이 추가
        time.sleep(delay_between_requests)

print("-" * 30)
print("번역 작업 완료.")

# --- 4. 결과 JSON 파일 저장 ---
print(f"번역 결과를 '{output_file_path}' 파일에 저장합니다...")
try:
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(english_chinese_data, outfile, ensure_ascii=False, indent=4)
    print("성공: 파일 저장이 완료되었습니다.")

except Exception as e:
    print(f"출력 파일을 저장하는 중 오류 발생: {e}")

# --- 5. 결과 요약 ---
successful_translations = len(english_chinese_data)
failed_translations = len(translation_errors)
print("\n--- 번역 결과 요약 ---")
print(f"총 시도 단어 수: {total_words}")
print(f"성공적으로 번역된 단어 수: {successful_translations}")
print(f"번역 실패 또는 경고 발생 단어 수: {failed_translations}")

if failed_translations > 0:
    print("\n--- 실패/경고 상세 ---")
    for word, reason in translation_errors.items():
        print(f" - '{word}': {reason}")
print("-" * 25)