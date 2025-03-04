from transformers import pipeline

# pipe = pipeline("text2text-generation", model="google/flan-t5-base")  # Korean included
# pipe = pipeline("text2text-generation", model="google/flan-t5-large")  # Korean included

question = "Can you speak English?"
result = pipe(question)

print("===========================================")
print("Question: ", question)
print("Answer: ", result[0]['generated_text'])
print("===========================================")

