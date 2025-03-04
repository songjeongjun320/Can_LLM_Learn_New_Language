from transformers import BertTokenizer, BertModel
import torch

# 사전 학습된 BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# 입력 문장
sentence = "나는 오늘 영화를 봤다"

# 토큰화 및 텐서 변환
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# BERT 모델을 사용하여 임베딩 생성
with torch.no_grad():
    outputs = model(**inputs)

# [CLS] 토큰의 임베딩 출력
sentence_embedding = outputs.last_hidden_state[:, 0, :]
print(sentence_embedding.shape)  # (1, 768) -> 768차원 벡터
