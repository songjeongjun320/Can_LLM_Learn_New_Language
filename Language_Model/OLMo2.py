# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="allenai/OLMo-2-1124-13B-Instruct")
pipe(messages)