import os
import re
import json

def clean_vtt(vtt_data):
    # 타임스탬프 뒤의 불필요한 position 정보 제거
    cleaned_data = re.sub(r'(\d{2}:\d{2}:\d{2})\.\d{3} --> (\d{2}:\d{2}:\d{2})\.\d{3} .*', r'\1 --> \2', vtt_data)
    
    # 대괄호나 소괄호 안의 내용 제거
    cleaned_data = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned_data)

    # WEBVTT, NOTE 라인과 빈 줄 모두 제거
    cleaned_data = re.sub(r'^(WEBVTT|NOTE.*\n)', '', cleaned_data, flags=re.MULTILINE)
    cleaned_data = re.sub(r'^\s*\n', '', cleaned_data, flags=re.MULTILINE)
    cleaned_data = re.sub(r'lrm', '', cleaned_data, flags=re.MULTILINE)
    cleaned_data = re.sub(r' ', '', cleaned_data, flags=re.MULTILINE)

    # 특정 태그 제거
    cleaned_data = re.sub(r'<c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'<c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'NETFLIX오리지널시리즈', '', cleaned_data)
    cleaned_data = re.sub(r'넷플릭스시리즈', '', cleaned_data)

    return cleaned_data

def process_vtt_file(file_path, output_file_path):
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        vtt_data = file.read()

    # VTT 데이터 정리
    cleaned_vtt = clean_vtt(vtt_data)

    # 줄 단위로 분리
    lines = cleaned_vtt.strip().split('\n')
    entries = []
    current_index = 1
    current_timestamp = None
    current_context = []

    for line in lines:
        if '-->' in line:  # 타임스탬프 줄인 경우
            # 이전 블록 처리: 타임스탬프와 모아진 텍스트가 있다면
            if current_timestamp is not None and current_context:
                context_text = "".join(current_context).strip()
                # 특수문자는 제거하고 띄어쓰기는 유지 (한글, 영문, 숫자, 띄어쓰기)
                context_text = re.sub(r'[^\w\s]+', '', context_text)
                if context_text:
                    entry = {
                        "index": current_index,
                        "timestamp": current_timestamp,
                        "context": context_text
                    }
                    entries.append(entry)
                    current_index += 1
            # 새 타임스탬프 갱신 및 텍스트 버퍼 초기화
            current_timestamp = line.strip()
            current_context = []
        else:
            # 만약 줄이 단순 숫자(인덱스 번호)라면 무시
            if re.match(r'^\d+$', line.strip()):
                continue
            # 타임스탬프가 없는 상태에서는 해당 텍스트 무시
            if current_timestamp is not None:
                current_context.append(line.strip())
            else:
                continue

    # 마지막 블록 처리
    if current_timestamp is not None and current_context:
        context_text = " ".join(current_context).strip()
        context_text = re.sub(r'[^\w\s]+', '', context_text)
        if context_text:
            entry = {
                "index": current_index,
                "timestamp": current_timestamp,
                "context": context_text
            }
            entries.append(entry)

    # JSON 파일로 저장 (한글 깨짐 방지를 위해 ensure_ascii=False)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(entries, output_file, ensure_ascii=False, indent=4)

    print(f"Cleaned VTT has been saved to {output_file_path}")
    print(f"Total {current_index} sentences.")

    return current_index  # 처리된 문장 수 반환

def process_all_vtt_files(root_dir, result_dir):
    # 결과를 저장할 최상위 폴더가 없으면 생성
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    total_sentences_all_files = 0  # 전체 문장 수를 추적하는 변수

    # root_dir 내의 각 폴더(드라마별) 순회
    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            # 드라마별 결과 폴더 생성
            drama_result_dir = os.path.join(result_dir, foldername)
            if not os.path.exists(drama_result_dir):
                os.makedirs(drama_result_dir)
            # 해당 폴더 내의 모든 .vtt 파일 처리
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.vtt'):
                    file_path = os.path.join(folder_path, file_name)
                    # JSON 파일은 파일명(확장자 제외)으로 저장
                    output_file_path = os.path.join(drama_result_dir, f"{os.path.splitext(file_name)[0]}.json")
                    total_sentences_all_files += process_vtt_file(file_path, output_file_path)  # 처리된 문장 수 더하기

    print(f"Total sentences processed across all files: {total_sentences_all_files}")

# 사용 예시
root_directory = r'C:\Users\songj\OneDrive\Desktop\Can-LLM-learn-new-Language\Samples'
result_directory = r'C:\Users\songj\OneDrive\Desktop\Can-LLM-learn-new-Language\Samples_Result'

process_all_vtt_files(root_directory, result_directory)
