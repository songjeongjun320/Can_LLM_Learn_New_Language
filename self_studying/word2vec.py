from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 예제 문장 데이터
sentences = [
    "나는 오늘 영화를 봤다",
    "영화는 정말 재미있었다",
    "오늘 날씨가 좋아서 기분이 좋다",
    "나는 책을 읽는 것을 좋아한다"
]

# 문장을 토큰화
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Word2Vec 모델 학습
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 특정 단어의 벡터 출력
word_vector = model.wv['영화']
print(word_vector)

# 가장 유사한 단어 찾기
similar_words = model.wv.most_similar('영화', topn=5)
print(similar_words)
