from transformers import pipeline

pipe = pipeline("conversational", model="google-bert/bert-base-uncased")
question = "Can you learn new language?"
result = pipe(question)

print("===========================================")
print("Question: ", question)
print("Answer: ", result[0]['token_str'])  # 'generated_text' 대신 'token_str' 사용
print("===========================================")
