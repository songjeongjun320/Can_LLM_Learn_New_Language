import json

# 파일 경로 설정
input_file = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/translated_words.json'
output_file = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/removed_translated_words.json'

# JSON 파일 읽기
with open(input_file, 'r', encoding='utf-8') as file:
    translated_words = json.load(file)

# 초기 데이터 길이 출력
initial_length = len(translated_words)

# 중복된 한글 단어를 제거하는 함수
unique_translated_words = {}
removed_entries = []

for korean_word, english_word in translated_words.items():
    # 소문자로 변환
    korean_word = korean_word.lower()
    english_word = english_word.lower()

    if korean_word not in unique_translated_words:
        unique_translated_words[korean_word] = english_word
    else:
        removed_entries.append({korean_word: english_word})

# 업데이트된 데이터 길이 출력
updated_length = len(unique_translated_words)

# 제거된 항목의 개수
removed_count = len(removed_entries)

# 수정된 단어 쌍을 새로운 JSON 파일에 저장
output_data = {
    "updated_translations": unique_translated_words,
    "removed_entries": removed_entries
}

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

# 출력 결과
print(f"Initial data length: {initial_length}")
print(f"Updated data length: {updated_length}")
print(f"Number of removed entries: {removed_count}")
print(f"Updated translations and removed entries have been saved to {output_file}")
