import json
import os

# --- 파일 경로 설정 ---
# 입력 파일 경로 (영어:중국어) - 중국어 값(value) 추출용
en_ch_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_ch.json'
# 입력 파일 경로 (영어:한국어) - 한국어 값(value) 추출용
en_kr_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/en_kr.json'
# 출력 파일 경로 (한국어:중국어)
kr_ch_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/kr_ch.json' # 파일 이름 변경 kr_ch

# --- 데이터 로드 함수 (순서 유지) ---
def load_json_ordered_dict(file_path):
    """주어진 경로의 JSON 파일을 순서가 보장되는 딕셔너리로 로드합니다."""
    print(f"파일 로드 시도: {file_path}")
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return None
    try:
        # 파이썬 3.7+ 에서는 기본 dict가 순서를 유지합니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  성공: {len(data)}개의 항목을 로드했습니다.")
            return data
    except json.JSONDecodeError:
        print(f"오류: 파일이 유효한 JSON 형식이 아닙니다 - {file_path}")
        return None
    except Exception as e:
        print(f"파일 로드 중 오류 발생 ({file_path}): {e}")
        return None

# --- 데이터 로드 ---
print("--- 입력 데이터 로드 시작 ---")
en_ch_data = load_json_ordered_dict(en_ch_file_path)
en_kr_data = load_json_ordered_dict(en_kr_file_path)
print("--- 입력 데이터 로드 완료 ---")

# 파일 로드 실패 시 종료
if en_ch_data is None or en_kr_data is None:
    print("\n오류: 필수 입력 파일 로드에 실패하여 프로그램을 종료합니다.")
    exit()

# --- 값(Value)들을 순서대로 리스트로 추출 ---
print("\n--- 값 리스트 추출 시작 ---")
korean_values = list(en_kr_data.values())
chinese_values = list(en_ch_data.values())
print(f"한국어 값 {len(korean_values)}개 추출 완료.")
print(f"중국어 값 {len(chinese_values)}개 추출 완료.")

# 두 리스트의 길이가 같은지 확인 (중요 가정)
if len(korean_values) != len(chinese_values):
    print("\n경고: 두 파일에서 추출된 값 리스트의 길이가 다릅니다!")
    print(f"  - 한국어 값 개수: {len(korean_values)}")
    print(f"  - 중국어 값 개수: {len(chinese_values)}")
    print("  결과가 정확하지 않을 수 있습니다. 짧은 쪽 길이에 맞춰 진행합니다.")
    # 길이가 다를 경우 짧은 쪽 길이에 맞춰서 진행
    num_items_to_process = min(len(korean_values), len(chinese_values))
    # 리스트 슬라이싱
    korean_values = korean_values[:num_items_to_process]
    chinese_values = chinese_values[:num_items_to_process]
else:
    num_items_to_process = len(korean_values)
    print(f"두 리스트 모두 {num_items_to_process}개의 값을 가지고 있습니다.")

# --- 한국어:중국어 딕셔너리 생성 ---
print("\n--- 한국어:중국어 딕셔너리 생성 시작 ---")
kr_ch_data = {}
overwritten_keys_count = 0
original_keys_leading_to_overwrite = {} # 덮어쓰기 정보 추적용 (선택적)

for i in range(num_items_to_process):
    korean_word = korean_values[i]
    chinese_word = chinese_values[i]

    # 한국어 단어가 유효한 경우에만 (None이거나 빈 문자열 제외, 키로 사용되므로 중요)
    if korean_word and korean_word.strip():
        clean_korean_word = korean_word.strip()

        # 덮어쓰기 발생 여부 확인
        if clean_korean_word in kr_ch_data:
            overwritten_keys_count += 1
            # 어떤 값이 덮어쓰는지 추적 (선택 사항)
            # original_value = kr_ch_data[clean_korean_word]
            # original_keys_leading_to_overwrite.setdefault(clean_korean_word, {'original': original_value, 'overwritten_by': []})['overwritten_by'].append(chinese_word)

        # 딕셔너리에 추가 (중복된 한국어 키는 덮어쓰기됨)
        kr_ch_data[clean_korean_word] = chinese_word
    else:
         print(f"  경고: {i+1}번째 항목의 한국어 값이 비어있어 키로 사용할 수 없습니다. 건너<0xEB><0x87><0x8D>니다.")


final_item_count = len(kr_ch_data)
print("--- 한국어:중국어 딕셔너리 생성 완료 ---")


# --- 결과 JSON 파일 저장 ---
print(f"\n--- 결과 저장 시작 ({kr_ch_file_path}) ---")
try:
    with open(kr_ch_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(kr_ch_data, outfile, ensure_ascii=False, indent=4)
    print(f"성공: 한국어-중국어 매핑 결과를 '{kr_ch_file_path}'에 저장했습니다.")
except Exception as e:
    print(f"오류: 결과 파일을 저장하는 중 오류 발생: {e}")

# --- 최종 결과 요약 ---
print("\n--- 최종 결과 요약 ---")
print(f"처리된 쌍의 수 (두 값 리스트 중 짧은 길이 기준): {num_items_to_process}")
print(f"최종 생성된 한국어-중국어 딕셔너리 항목 수: {final_item_count}")
if overwritten_keys_count > 0:
     print(f"  - !!! 중요: 한국어 키 중복으로 인해 덮어쓰기 발생: {overwritten_keys_count} 건 !!!")
     print(f"     (최종 파일에는 중복된 한국어 키당 마지막 중국어 값만 포함됩니다)")
elif final_item_count < num_items_to_process:
     print(f"  (참고: 빈 한국어 값으로 인해 {num_items_to_process - final_item_count}개의 항목이 제외되었습니다.)")

print("-" * 25)
print("참고: 이 코드는 두 입력 파일의 값(value) 순서가 정확히 일치한다고 가정하고 작동합니다.")