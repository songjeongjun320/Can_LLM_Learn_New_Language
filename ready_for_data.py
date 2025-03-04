import json

# result_origin_v4.json 파일 로드
with open("result_origin_v4.json", "r", encoding="utf-8") as f:
    result_data = json.load(f)

# subtitle.json 파일 로드
with open("subtitle.json", "r", encoding="utf-8") as f:
    subtitle_data = json.load(f)

# dataset.json으로 저장할 데이터 리스트 초기화
dataset = []

# 두 파일의 데이터를 매칭하여 dataset 생성
for result_item, subtitle_item in zip(result_data, subtitle_data):
    input_text = subtitle_item.get("context", "")  # subtitle.json의 "context"를 input으로
    output_text = result_item.get("response", "")  # result_origin_v4.json의 "response"를 output으로
    
    # input과 output이 모두 비어있지 않은 경우만 추가
    if input_text and output_text:
        dataset.append({"input": input_text, "output": output_text})

# dataset.json 파일로 저장
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("dataset.json 파일이 생성되었습니다.")