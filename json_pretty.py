# Make json file readible.

import json
import textwrap

# JSON 파일 로드
with open('result_origin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# response 값을 50자씩 나누어 배열로 분할
for item in data:
    if 'response' in item:
        # 50자씩 나누기
        wrapped_text = textwrap.wrap(item['response'], width=100)
        item['response'] = wrapped_text

# 수정된 JSON 파일 저장
with open('result_pretty.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("JSON 파일이 수정되어 저장되었습니다.")
