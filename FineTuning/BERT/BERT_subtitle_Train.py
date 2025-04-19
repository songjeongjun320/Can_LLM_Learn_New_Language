import os
import glob
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM, # BERT를 MLM 작업으로 파인튜닝
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import logging
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 ---
# DATA_DIR = "/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/KR_ENG_Dataset_Refined" # 번역 데이터셋
VERSION = "v1"
DATA_DIR = "/scratch/jsong132/Can_LLM_Learn_New_Language/DB/Refined_Datas/v2" # 자막:영어 데이터셋
MODEL_NAME = "bert-base-uncased" # 사용할 BERT 모델
OUTPUT_DIR = f"./Tuned_Results/bert-uncased-finetuned-subtitle_dt_{VERSION}" # 파인튜닝된 모델 저장 경로
CACHE_DIR = "./cache_subtitle" # 데이터셋 캐시 저장 경로 (선택 사항)
os.makedirs(CACHE_DIR, exist_ok=True)

# 학습 파라미터
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 32 # GPU 메모리에 맞게 조절
PER_DEVICE_EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_STEPS = 200 # 모델 체크포인트 저장 간격
MAX_SEQ_LENGTH = 512 # 입력 시퀀스 최대 길이 (BERT는 보통 512)
MLM_PROBABILITY = 0.15 # MLM에서 마스킹할 토큰 비율
TRAIN_TEST_SPLIT_RATIO = 0.1 # 검증 세트 비율

# --- 데이터 로딩 함수 ---
def load_data_from_json_files(data_dir):
    """지정된 디렉토리의 모든 JSON 파일을 로드하여 단일 리스트로 반환"""
    all_data = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        logging.error(f"오류: '{data_dir}' 디렉토리에서 JSON 파일을 찾을 수 없습니다.")
        sys.exit(1) # 프로그램 종료

    logging.info(f"'{data_dir}'에서 다음 JSON 파일들을 로드합니다: {json_files}")
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    logging.warning(f"경고: '{file_path}' 파일의 형식이 리스트가 아닙니다. 건너<0xEB><0x9B><0x84>니다.")
        except json.JSONDecodeError:
            logging.error(f"오류: '{file_path}' 파일 파싱 중 오류 발생.")
        except Exception as e:
            logging.error(f"오류: '{file_path}' 파일 처리 중 예외 발생: {e}")

    logging.info(f"총 {len(all_data)}개의 데이터 포인트를 로드했습니다.")
    if not all_data:
        logging.error("오류: 유효한 데이터가 로드되지 않았습니다. 프로그램을 종료합니다.")
        sys.exit(1)
    return all_data

# --- 데이터 전처리 함수 ---
def preprocess_data(examples, tokenizer):
    """입력과 출력을 합쳐서 토크나이징 (MLM용)"""
    texts = []
    for inp, outp in zip(examples["input"], examples["output"]):
        try:
            if inp and outp:  # 입력 또는 출력이 비어있지 않은 경우만 처리
                texts.append(inp + " [SEP] " + outp)
        except Exception as e:
            print(f"Error processing: {inp}, {outp}, Error: {e}")
            continue  # 문제가 있는 예제는 건너뜁니다
            
    # 주의: max_length 초과 시 잘라냅니다.
    tokenized_output = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_special_tokens_mask=True,  # MLM Collator에 필요할 수 있음
    )
    return tokenized_output


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    raw_data = load_data_from_json_files(DATA_DIR)

    # 2. 모델 및 토크나이저 로드
    logging.info(f"'{MODEL_NAME}' 모델과 토크나이저를 로드합니다...")
    try:
        # 빠른 토크나이저 사용
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        # MLM 작업을 위한 모델 로드
        model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    except OSError as e:
        logging.error(f"오류: 모델 또는 토크나이저 로드 실패 ({MODEL_NAME}). 모델 이름이 올바른지, 인터넷 연결이 가능한지 확인하세요. 에러: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"모델/토크나이저 로드 중 예기치 않은 오류: {e}")
        sys.exit(1)


    logging.info("모델 및 토크나이저 로드 완료.")
    logging.warning(f"주의: '{MODEL_NAME}' 모델은 영어 uncased 모델입니다. 한국어('output' 필드) 처리 성능이 제한적일 수 있습니다.")
    logging.warning("주의: BERT는 인코더 모델로, 직접적인 번역 생성에는 적합하지 않습니다. 이 코드는 Masked Language Modeling(MLM)으로 파인튜닝합니다.")

    # 3. 데이터를 Hugging Face Dataset 형식으로 변환
    # 필요한 필드만 추출하여 딕셔너리 리스트 생성
    hf_data = [{"input": item["input"], "output": item["output"]} for item in raw_data if "input" in item and "output" in item]
    if not hf_data:
        logging.error("오류: 처리할 유효한 'input'/'output' 쌍이 데이터에 없습니다.")
        sys.exit(1)

    dataset = Dataset.from_list(hf_data)
    logging.info(f"Hugging Face Dataset 생성 완료 (총 {len(dataset)}개 항목)")

    # 4. 데이터셋 분할 (Train / Validation)
    if TRAIN_TEST_SPLIT_RATIO > 0:
        logging.info(f"데이터셋을 학습 세트와 검증 세트로 분할합니다 (검증 비율: {TRAIN_TEST_SPLIT_RATIO}).")
        split_dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT_RATIO)
        tokenized_datasets = split_dataset.map(
            preprocess_data,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=split_dataset["train"].column_names,
            cache_file_names={
                "train": os.path.join(CACHE_DIR, "train_cache.arrow"),
                "test": os.path.join(CACHE_DIR, "test_cache.arrow"),
            }
        )
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]
        logging.info(f"학습 세트: {len(train_dataset)}개, 검증 세트: {len(eval_dataset)}개")
    else:
        logging.info("데이터셋 전체를 학습에 사용합니다 (검증 세트 없음).")
        tokenized_datasets = dataset.map(
            preprocess_data,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count() // 2,
            cache_file_names={
                "train": os.path.join(CACHE_DIR, "full_train_cache.arrow"),
            }
        )
        train_dataset = tokenized_datasets
        eval_dataset = None # 검증 세트 없음


    # 5. Data Collator 설정 (MLM용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    logging.info(f"MLM Data Collator 설정 완료 (마스킹 확률: {MLM_PROBABILITY}).")

    # 6. Training Arguments 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs', # TensorBoard 로그 저장 경로
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps" if eval_dataset else "no", # 검증 세트가 있을 경우 steps마다 평가
        eval_steps=SAVE_STEPS if eval_dataset else None, # 저장할 때 같이 평가
        save_total_limit=3, # 최대 저장할 체크포인트 수
        load_best_model_at_end=True if eval_dataset else False, # 학습 종료 후 최적 모델 로드 (검증 필요)
        metric_for_best_model="loss" if eval_dataset else None, # 최적 모델 기준 (낮을수록 좋음)
        greater_is_better=False if eval_dataset else None,
        fp16=torch.cuda.is_available(), # GPU 사용 가능 시 FP16 사용 (속도 향상, 메모리 절약)
        report_to="tensorboard", # 혹은 "wandb", "none" 등
        dataloader_num_workers=1 # 데이터 로딩 워커 수 (환경에 맞게 조절)
    )
    logging.info(f"Training Arguments 설정 완료. 출력 디렉토리: {OUTPUT_DIR}")

    # 7. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # 검증 세트 전달
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. 학습 시작
    logging.info("*** 학습 시작 ***")
    try:
        train_result = trainer.train()
        # train_result = trainer.train(resume_from_checkpoint=True) # 가장 최근 checkpoint부터 재시작

        logging.info("*** 학습 완료 ***")

        # 학습 결과 메트릭 저장 및 로깅
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # 최종 모델 저장
        logging.info(f"최종 모델 및 토크나이저를 '{OUTPUT_DIR}'에 저장합니다...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logging.info("모델 및 토크나이저 저장 완료.")

    except Exception as e:
        logging.error(f"학습 중 오류 발생: {e}")
        # 오류 발생 시 현재 상태 저장 시도 (선택 사항)
        # try:
        #     logging.warning("오류 발생으로 현재 모델 상태를 저장합니다...")
        #     trainer.save_model(os.path.join(OUTPUT_DIR, "error_checkpoint"))
        #     tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "error_checkpoint"))
        # except Exception as save_e:
        #     logging.error(f"오류 발생 후 모델 저장 실패: {save_e}")
        sys.exit(1)

    # (선택 사항) 최종 평가 수행
    if eval_dataset:
        logging.info("*** 최종 평가 시작 ***")
        eval_metrics = trainer.evaluate()
        logging.info(f"최종 평가 결과: {eval_metrics}")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)