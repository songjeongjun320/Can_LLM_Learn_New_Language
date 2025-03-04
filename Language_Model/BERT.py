from transformers import pipeline

# Use the pipeline for question answering
# qa_pipeline = pipeline("question-answering", model="bert-base-uncased", tokenizer="bert-base-uncased")
unmasker = pipeline('fill-mask', model='bert-base-uncased')
result = unmasker("Hello I'm a [MASK] model.")

for tmp in result:
    print(tmp)

# Define the question and context
question = "대한민국의 수도는 어디야?"
context = "서울특별시는 대한민국의 수도이자 최대도시이며, 대한민국 유일의 특별시이다. ... 역사적으로도 백제, 조선, 대한제국의 수도이자 현재 대한민국의 수도로서 중요성이 ..."

# Use the pipeline to get the answer
# result = qa_pipeline(question=question, context=context)
# print(f"Answer: {result['answer']}")
