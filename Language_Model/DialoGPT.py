from transformers import pipeline

messages = [
    {"role": "user", "content": "Can you learn new language?"},
]
# Initialize the pipeline with the model
pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium", pad_token_id = 50256)
response = pipe(messages)

# Output the generated response
print(response)