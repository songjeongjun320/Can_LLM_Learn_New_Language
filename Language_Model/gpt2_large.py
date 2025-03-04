from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2-large')
set_seed(42)

question = "점심 뭐먹을까?"
result = generator(question, max_length=30, num_return_sequences=5, truncation=True)

# Print the result
for idx, text in enumerate(result):
    print("==========================================")
    print(f"Generated Text {idx+1}: {text['generated_text']}")
print("==========================================")