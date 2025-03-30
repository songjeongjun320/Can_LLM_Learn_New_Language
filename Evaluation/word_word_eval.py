from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# multilingual fastText 모델 로드 (미리 학습된 모델을 사용)
model = FastText.load("path_to_multilingual_fasttext_model")

# 영어 단어와 한국어 번역된 단어 임베딩
en_word = model.wv["apple"]  # 예: 'apple'에 대한 임베딩
ko_word = model.wv["사과"]   # 예: '사과'에 대한 임베딩

# 두 벡터의 코사인 유사도 계산
similarity = cosine_similarity([en_word], [ko_word])[0][0]

print(f"Cosine similarity between 'apple' and '사과': {similarity:.4f}")
