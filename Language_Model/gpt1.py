# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/openai-gpt")

# 질문 리스트
questions = [
    "Can you speak English?",
    "한국말 할 줄 아니?",
    "Nǐ dǒng zhōngwén ma?"
]

# 질문에 대한 답변 생성 및 출력
for question in questions:
    response = pipe(question, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    print(f"Question: {question}")
    print(f"Answer: {response[0]['generated_text']}")
    print("=" * 30)
