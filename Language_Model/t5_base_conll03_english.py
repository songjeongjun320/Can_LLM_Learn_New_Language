# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="dbmdz/t5-base-conll03-english")
question = "Can you speak English?"
result = pipe(question)


print("===========================================")
print("Question: ", question)
print("Answer: ", result[0]['generated_text'])
print("===========================================")
