import json
import asyncio
import logging
from googletrans import Translator

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 파일 경로
input_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/words.json'
output_file_path = '/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/translated_words.json'

# 번역기 초기화
translator = Translator()

# 비동기 함수로 번역
async def translate_words():
    # JSON 파일을 읽어들임
    logger.info("입력 파일을 읽고 있습니다: %s", input_file_path)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("입력 파일 읽기 완료")
    except Exception as e:
        logger.error("입력 파일 읽기 실패: %s", e)
        return

    # 번역된 단어들을 저장할 딕셔너리
    translated_dict = {}

    # 각 한국어 단어를 영어로 번역
    logger.info("단어 번역을 시작합니다...")
    for word in data['kor']:
        logger.info("단어 번역 중: %s", word)
        try:
            # 비동기 방식으로 번역
            translation = await translator.translate(word, src='ko', dest='en')
            translated_dict[word] = translation.text
            logger.info("단어 번역 완료: %s -> %s", word, translation.text)
        except Exception as e:
            logger.error("단어 번역 실패: %s - %s", word, e)

    # 번역된 데이터를 새로운 JSON 형식으로 저장
    logger.info("번역된 데이터를 파일에 저장 중: %s", output_file_path)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(translated_dict, f, ensure_ascii=False, indent=4)
        logger.info("번역된 데이터 저장 완료")
    except Exception as e:
        logger.error("번역된 데이터 저장 실패: %s", e)

# asyncio를 사용해 비동기 함수 실행
if __name__ == "__main__":
    logger.info("번역 작업 시작")
    asyncio.run(translate_words())
    logger.info("번역 작업 완료")
