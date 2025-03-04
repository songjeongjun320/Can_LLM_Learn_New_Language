import json

# JSON 파일 경로 설정
file_path = 'C:\\Users\\songj\\OneDrive\\Desktop\\Can LLM Conquer Human Language\\KorQuAD_2.1_train_12\\korquad2.1_train_36.json'

# JSON 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터 구조 확인 및 출력
def explore_data(data, num_examples=10):  # 예: 10개까지 데이터 출력
    print("데이터셋 정보:")
    print(f"데이터 키: {list(data.keys())}")
    
    print("\n샘플 데이터:")
    for i, article in enumerate(data.get('data', [])[:num_examples]):
        print(f"\n[Article {i+1}] Title: {article.get('title', 'N/A')}")
        for paragraph in article.get('paragraphs', []):
            context = paragraph.get('context', 'N/A')
            if context != 'N/A':
                print("Paragraph:", context[:100], "...")  # 문단의 일부 출력
            for qa in paragraph.get('qas', []):  # 각 문단의 Q&A 접근
                question = qa.get('question', 'N/A')
                answers = [ans.get('text', 'N/A') for ans in qa.get('answers', [])]  # 모든 답 추출
                print(f"  Q: {question}")
                print(f"  Answers: {', '.join(answers)}")  # 모든 답을 콤마로 연결해서 출력
            break  # 첫 번째 문단만 출력

# 데이터 출력
explore_data(data)
