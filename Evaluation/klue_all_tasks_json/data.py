import json
import os

# --- 설정 ---
input_file_path = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
# 출력 파일 이름 (원본과 구분되도록 변경)
output_file_path = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json_filtered.json"
max_input_length = 500 # 최대 허용 입력 길이 (문자 기준)

# --- 함수 정의: JSON 필터링 ---
def filter_json_by_input_length(input_path, output_path, max_len):
    """JSON 파일을 읽어 input 필드 길이가 max_len 이하인 항목만 필터링하여 새 파일에 저장"""
    print(f"처리 시작: {input_path}")
    filtered_data = []
    original_count = 0
    filtered_count = 0
    skipped_count = 0

    try:
        # 입력 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as infile:
            original_data = json.load(infile)

        if not isinstance(original_data, list):
            print(f"오류: {input_path} 파일의 최상위 구조가 리스트가 아닙니다.")
            return

        original_count = len(original_data)
        print(f"원본 데이터 항목 수: {original_count}")

        # 각 항목 순회 및 필터링
        for i, item in enumerate(original_data):
            if isinstance(item, dict) and "input" in item:
                input_value = item.get("input")
                # input 값이 문자열인지 확인
                if isinstance(input_value, str):
                    # 문자열 길이 확인
                    if len(input_value) <= max_len:
                        filtered_data.append(item)
                        filtered_count += 1
                    else:
                        # 길이가 너무 긴 경우 건너뜀
                        # print(f"정보: 항목 {i+1} 건너뜀 (입력 길이 {len(input_value)} > {max_len})")
                        skipped_count += 1
                else:
                    # input 값이 문자열이 아닌 경우 (일단 건너뜀)
                    print(f"경고: 항목 {i+1} 건너뜀 ('input' 값이 문자열이 아님)")
                    skipped_count += 1
            else:
                # 항목이 딕셔너리가 아니거나 'input' 키가 없는 경우 (일단 건너뜀)
                print(f"경고: 항목 {i+1} 건너뜀 (형식 오류 또는 'input' 키 부재)")
                skipped_count += 1

        # 필터링된 데이터를 새 JSON 파일에 저장
        output_dir = os.path.dirname(output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True) # 출력 디렉토리 생성

        with open(output_path, 'w', encoding='utf-8') as outfile:
            # ensure_ascii=False: 한글 유지, indent=2: 가독성 높임
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=2)

        print(f"처리 완료: {output_path}")
        print(f"  필터링 후 항목 수: {filtered_count}")
        print(f"  제외된 항목 수 (길이 초과 또는 오류): {skipped_count}")
        print(f"  (확인용) 원본 항목 수: {original_count} = 필터링 후({filtered_count}) + 제외({skipped_count})")


    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"오류: 입력 파일 '{input_path}' 파싱 중 오류 발생 (유효한 JSON 아님).")
    except Exception as e:
        print(f"파일 처리 중 예기치 않은 오류 발생 ({input_path}): {e}")

# --- 메인 실행 ---
if __name__ == "__main__":
    filter_json_by_input_length(input_file_path, output_file_path, max_input_length)
    print("\n모든 작업 완료.")