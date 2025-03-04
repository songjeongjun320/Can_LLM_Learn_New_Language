# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="FacebookAI/roberta-base")
question = "This word 안녕 is hello in Korean. If I say 안녕 to you. What should you say? <mask>"
result = pipe(question)

print("===========================================")
print("Question: ", question)
print("Answer: ", result[0]['token_str'])  # 'generated_text' 대신 'token_str' 사용
print("===========================================")
