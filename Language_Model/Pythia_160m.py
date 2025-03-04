# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="EleutherAI/pythia-160m")

question = "내가 무슨 말 하는지 이해해?"
result = pipe(question)

print(result)