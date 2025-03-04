import re
import json

def clean_vtt(vtt_data):
    # WEBVTT, NOTE 라인과 빈 줄 모두 제거
    cleaned_data = re.sub(r'^(WEBVTT|NOTE.*\n)', '', vtt_data, flags=re.MULTILINE)
    cleaned_data = re.sub(r'^\s*\n', '', cleaned_data, flags=re.MULTILINE)

    # 특정 태그 제거
    cleaned_data = re.sub(r'<c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'<c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'NETFLIX오리지널시리즈', '', cleaned_data)
    # 타임스탬프 뒤의 불필요한 position 정보 제거
    cleaned_data = re.sub(r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}) .*', r'\1', cleaned_data)
    # 대괄호나 소괄호 안의 내용 제거
    cleaned_data = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned_data)

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
    total_sentences = 0

    for line in lines:
        if '-->' in line:  # 타임스탬프 줄인 경우
            # 이전 블록 처리: 타임스탬프와 모아진 텍스트가 있다면
            if current_timestamp is not None and current_context:
                context_text = " ".join(current_context).strip()
                # 모든 띄어쓰기와 특수문자 제거 (한글, 영문, 숫자만 남김)
                context_text = re.sub(r'[\s\W]+', '', context_text)
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
                total_sentences += 1
            else:
                continue

    # 마지막 블록 처리
    if current_timestamp is not None and current_context:
        context_text = " ".join(current_context).strip()
        context_text = re.sub(r'[\s\W]+', '', context_text)
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
    print(f"Total sentences: {total_sentences}")

# 예시 사용 (파일 경로와 출력 경로를 알맞게 수정)
file_path = r'C:\Users\songj\OneDrive\Desktop\Can-LLM-learn-new-Language\Samples\나의.아저씨\나의.아저씨.S01E01.WEBRip.Netflix.ko[cc].vtt'
output_file_path = r'output.json'
process_vtt_file(file_path, output_file_path)
