# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Can you speak English?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
pipe(messages)