from transformers import pipeline

pipe = pipeline("text2text-generation", model="google-t5/t5-small") # It is not including Korean   

question = "Hi I am Jun."
result = pipe(question)

print("===========================================")
print("Question: ", question)
print("Answer: ", result[0]['generated_text'])
print("===========================================")
