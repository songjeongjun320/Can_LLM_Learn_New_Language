# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-base")

# 질문 리스트
questions = [
    "Can you speak English?",
    "오늘 점심 뭐먹을까?",
    "Jīntiān wǔfàn chī shénme?"
]

# 질문에 대한 답변 생성 및 출력
for question in questions:
    response = pipe(question, max_length=50, num_return_sequences=1)
    print(f"Question: {question}")
    print(f"Answer: {response[0]['generated_text']}")
    print("=" * 30)
