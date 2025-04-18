import json
import re
import os

# 입력 파일 경로와 출력 파일 경로 설정
input_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/KR_ENG_Dataset_Refined/lemon-mint_finetome_100k_step2_no_role.jsonl'
# 출력 파일 이름 및 확장자 변경 (.json)
output_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/KR_ENG_Dataset_Refined/lemon-mint_finetome_100k_step2_no_role_transformed_final.json' # 파일 이름 변경

# 출력 디렉토리가 없으면 생성
output_dir = os.path.dirname(output_file_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"생성된 출력 디렉토리: {output_dir}")

# 정규 표현식 패턴 정의: content와 content_en의 '값' 부분을 캡처
# "key": "value" 형태를 찾습니다. value 부분은 큰따옴표 안의 모든 문자(이스케이프된 따옴표 포함)를 non-greedy하게 찾습니다.
content_pattern = re.compile(r'"content":\s*"((?:\\.|[^"\\])*)"')
content_en_pattern = re.compile(r'"content_en":\s*"((?:\\.|[^"\\])*)"')

# 결과를 저장할 리스트 초기화
all_transformed_data = []

processed_lines_count = 0
extracted_pairs_count = 0
error_lines_count = 0
warning_lines_count = 0

try:
    # 원본 파일 인코딩을 'utf-8'로 가정하고 읽기
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        print(f"입력 파일 처리 시작: {input_file_path}")

        for i, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue # 빈 줄 건너뛰기

            processed_lines_count += 1
            pairs_found_in_line = 0
            handled = False # 해당 라인이 처리되었는지 여부 플래그

            try:
                # 1. 정규 표현식으로 모든 content/content_en 값 추출 시도
                contents = content_pattern.findall(line)
                content_ens = content_en_pattern.findall(line)

                # 1.1. 찾은 쌍의 개수가 일치하고 0보다 큰 경우
                if len(contents) > 0 and len(contents) == len(content_ens):
                    handled = True # 정규식으로 처리 시도
                    for c_val, ce_val in zip(contents, content_ens):
                        try:
                            # 추출된 값(c_val, ce_val)을 유효한 JSON 문자열로 만들어 파싱 시도
                            # 이렇게 하면 JSON 표준 이스케이프(\n, \", \\ 등)가 올바르게 처리됨
                            parsed_c = json.loads(f'"{c_val}"')
                            parsed_ce = json.loads(f'"{ce_val}"')

                            # 키 이름 변경하여 딕셔너리 생성
                            output_dict = {
                                "input": parsed_ce,  # content_en -> input
                                "output": parsed_c   # content -> output
                            }
                            all_transformed_data.append(output_dict)
                            extracted_pairs_count += 1
                            pairs_found_in_line += 1

                        except json.JSONDecodeError as e_decode:
                            # 추출된 값이 유효한 JSON 문자열 값이 아닌 경우 (드뭄)
                            print(f"경고: 라인 {i+1}의 값 파싱 실패 (JSONDecodeError: {e_decode}). 원본 값 사용 시도. content='{c_val[:50]}...', content_en='{ce_val[:50]}...'")
                            # 원본 값 그대로 사용 (이스케이프 처리 안 될 수 있음)
                            output_dict = {
                                "input": ce_val,
                                "output": c_val
                            }
                            all_transformed_data.append(output_dict)
                            extracted_pairs_count += 1
                            pairs_found_in_line += 1
                            warning_lines_count += 1 # 경고 카운트 증가

                        except Exception as e_inner:
                            print(f"오류: 라인 {i+1}의 쌍 처리 중 내부 오류 발생: {e_inner}. 값: content='{c_val[:50]}...', content_en='{ce_val[:50]}...'")
                            error_lines_count += 1 # 오류 카운트 증가 (쌍 처리 실패)

                # 1.2. 찾은 쌍의 개수가 불일치하는 경우
                elif len(contents) != len(content_ens):
                    handled = True # 처리 시도는 했으나 실패
                    print(f"오류: 라인 {i+1}에서 'content'({len(contents)}개)와 'content_en'({len(content_ens)}개)의 수가 불일치: {line[:100]}...")
                    error_lines_count += 1

                # 2. 정규식으로 쌍을 찾지 못한 경우 (handled == False), 표준 JSON으로 파싱 시도
                if not handled:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and 'content' in data and 'content_en' in data:
                            # 표준 JSON 객체에서 값 추출 및 변환
                            output_dict = {
                                "input": data['content_en'],
                                "output": data['content']
                            }
                            all_transformed_data.append(output_dict)
                            extracted_pairs_count += 1
                            pairs_found_in_line += 1
                            handled = True # 표준 JSON으로 처리 성공
                        # else:
                            # 유효한 JSON이지만 필요한 키가 없는 경우는 무시하거나 로깅할 수 있음
                            # print(f"정보: 라인 {i+1}은 유효한 JSON이지만 필요한 키가 없음: {line[:100]}...")

                    except json.JSONDecodeError:
                        # 정규식도 실패하고, 표준 JSON 파싱도 실패한 경우
                         if not handled: # 위에서 이미 오류로 처리되지 않았다면
                             print(f"경고: 라인 {i+1}에서 쌍/키를 찾지 못했고 유효한 JSON도 아님: {line[:100]}...")
                             warning_lines_count += 1 # 경고로 처리 (데이터 추출 불가)


            except Exception as e:
                # 라인 처리 중 예기치 않은 오류 (예: 정규식 오류 등)
                print(f"오류: 라인 {i+1} 처리 중 예외 발생: {e}. 라인 내용: {line[:100]}...")
                error_lines_count += 1

            # 진행 상황 표시 (선택 사항)
            if (processed_lines_count % 10000 == 0):
                print(f"  {processed_lines_count} 라인 처리 완료 (현재 추출된 쌍: {extracted_pairs_count})...")

except FileNotFoundError:
    print(f"오류: 입력 파일 '{input_file_path}'을(를) 찾을 수 없습니다.")
    all_transformed_data = None # 처리 중단됨을 명시
except Exception as e:
    print(f"파일 읽기/처리 중 예기치 않은 오류 발생: {e}")
    all_transformed_data = None # 처리 중단됨을 명시

# --- 결과 저장 ---
if all_transformed_data is not None:
    if not all_transformed_data:
         print("\n경고: 추출된 데이터가 없습니다. 빈 파일을 생성합니다.")

    try:
        print(f"\n총 {extracted_pairs_count}개의 쌍을 추출했습니다. 파일에 저장합니다...")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # 전체 리스트를 JSON 배열 형식으로 저장
            # ensure_ascii=False : 한글 등 비ASCII 문자를 유니코드 이스케이프 없이 저장
            # indent=2 : 사람이 읽기 좋게 2칸 들여쓰기 적용
            json.dump(all_transformed_data, outfile, ensure_ascii=False, indent=2)

        print("\n--- 처리 결과 요약 ---")
        print(f"총 처리된 입력 라인 수: {processed_lines_count}")
        print(f"총 추출 및 변환된 쌍(레코드) 수: {extracted_pairs_count}")
        print(f"오류가 발생한 라인 수 (데이터 추출 실패 가능성 높음): {error_lines_count}")
        print(f"경고가 발생한 라인 수 (데이터 파싱/형식 문제 가능성): {warning_lines_count}")
        print(f"결과가 저장된 파일: {output_file_path}")

    except Exception as e:
        print(f"\n오류: 최종 결과를 파일 '{output_file_path}'에 저장하는 중 문제 발생: {e}")
else:
    print("\n오류로 인해 데이터 처리가 중단되었거나 유효한 데이터가 없어 결과를 저장하지 않았습니다.")