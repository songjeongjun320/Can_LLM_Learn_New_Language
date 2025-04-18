import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split  # 여기가 핵심!
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
import re
from tqdm import tqdm
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from functools import partial

from torch.utils.data import Dataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정 클래스
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# 모델 설정들 (기본 OLMo 1B, OLMo 7B)
MODEL_CONFIGS = [
    # ModelConfig(
    #     name="full-OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_dst_results/full-olmo1B-org-klue-dst",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-1b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
    #     output_dir="klue_dst_results/full-olmo1B-v12-klue-dst",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_dst_results/full-olmo7B-org-klue-dst",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="full-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_dst_results/full-olmo7B-v13-klue-dst",
    #     is_local=True
    # ),
    #     ModelConfig(
    #     name="full-Llama-3.2:3B", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
    #     output_dir="klue_dst_results/full-llama3.2-3b-klue-dst",
    #     is_local=True
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased",
    #     model_path="bert-base-uncased",
    #     is_local=False,
    #     output_dir="klue_dst_results/BERT-base-uncased-klue-dst",
    # ),
    # ModelConfig(
    #     name="BERT-base-uncased-Tuned",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_BERT-base-uncased",
    #     is_local=True, # Assuming this is local based on path pattern
    #     output_dir="klue_dst_results/BERT-base-uncased-Tuned-klue-dst",
    # ),
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        output_dir="klue_dst_results/lora-llama3.2-3b-it-klue-dst",
        is_local=True,
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        output_dir="klue_dst_results/lora-llama3.1-8b-it-klue-dst",
        is_local=True
    ),
    # ModelConfig(
    #     name="BERT-uncased-kr-eng-translation",
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/bert-uncased-finetuned-kr-eng",
    #     output_dir="klue_dst_results/llama3.1-8b-it-klue-dst",
    #     is_local=True
    # ),
]

# 기본 설정
DATA_CACHE_DIR = "./klue_dst_origin_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_train_filtered.json"
MAX_LENGTH = 712  # DST는 대화 컨텍스트가 더 길 수 있으므로 길이 증가

# 데이터셋 준비 함수 - JSON 파일 생성
def prepare_dataset_json():
    """KLUE DST 데이터셋을 불러와서 JSON 파일로 변환합니다.
    이미 분할된 JSON_TRAIN_DATASET_PATH 파일을 사용합니다.
    """
    try:
        logger.info("Loading KLUE DST training dataset from JSON file...")
        with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)

        # 데이터 분할 (90% train, 10% validation)
        # dataset_size = len(full_dataset)
        dataset_size = 6000
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size

        train_raw = full_dataset[:train_size]
        val_raw = full_dataset[train_size:]

        logger.info(f"Dataset split complete - Train: {len(train_raw)} samples, Val: {len(val_raw)} samples")

        train_samples = []
        val_samples = []

        # 프롬프트와 완성 텍스트 생성 함수
        def create_prompt(dialogue_history, current_utterance):
            history_text = ""
            for turn in dialogue_history:
                if turn["role"] == "user":
                    history_text += f"사용자: {turn['text']}\n"
                else:
                    history_text += f"시스템: {turn['text']}\n"
            return (
                "다음은 사용자와 시스템 간의 대화입니다. 마지막 사용자 발화에 대한 대화 상태를 추적하세요.\n\n"
                f"{history_text}사용자: {current_utterance}\n\n대화 상태:"
            )

        def create_completion(state):
            if not state:
                return " 없음"
            else:
                return " " + ", ".join(state)

        # 학습 데이터 전처리
        logger.info("Creating training samples...")
        for dialogue in tqdm(train_raw):
            dialogue_history = []
            for turn in dialogue["dialogue"]:
                # 사용자 발화만 처리
                if turn["role"] == "user":
                    current_utterance = turn["text"]
                    state = turn["state"]
                    prompt = create_prompt(dialogue_history, current_utterance)
                    completion = create_completion(state)
                    sample = {
                        "input": prompt,
                        "output": completion
                    }
                    train_samples.append(sample)
                dialogue_history.append(turn)

        # 검증 데이터 전처리
        logger.info("Creating validation samples...")
        for dialogue in tqdm(val_raw):
            dialogue_history = []
            for turn in dialogue["dialogue"]:
                # 사용자 발화만 처리
                if turn["role"] == "user":
                    current_utterance = turn["text"]
                    state = turn["state"]
                    prompt = create_prompt(dialogue_history, current_utterance)
                    completion = create_completion(state)
                    sample = {
                        "input": prompt,
                        "output": completion
                    }
                    val_samples.append(sample)
                dialogue_history.append(turn)

        # 전처리된 학습 데이터 저장
        logger.info(f"Saving processed training dataset... (samples: {len(train_samples)})")
        with open(JSON_TRAIN_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed KLUE DST datasets saved to: {JSON_TRAIN_DATASET_PATH}")

    except Exception as e:
        logger.error(f"Error creating DST datasets: {e}")
        raise

# 모델 및 토크나이저 로드 함수
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model: {model_config.model_path}")

    is_local=False
    if (model_config.is_local):
        is_local = True

    # 일반적인 HuggingFace 모델 로드 (OLMo 1B, OLMo 7B)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, 
        local_files_only=is_local,
        trust_remote_code=True
        )
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # bfloat16 정밀도로 모델 로드 (메모리 효율성 증가)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 자동으로 GPU에 모델 분산
        local_files_only=is_local,
        trust_remote_code=True  # OLMo 모델에 필요
    )
    
    return model, tokenizer

# 메인 학습 함수
def train_model(model_config):
    """
    모델을 학습시킵니다.
    JSON_TRAIN_DATASET_PATH를 90% 학습 / 10% 검증으로 분할하여 사용합니다.
    """
    # --- 데이터 전처리 함수 정의 ---
    def preprocess_function_for_training(examples, tokenizer, max_length=MAX_LENGTH):
        """학습용 데이터 전처리: input과 output을 결합하여 Causal LM 형식으로 만듭니다."""
        texts = []
        for inp, outp in zip(examples["input"], examples["output"]):
            inp = inp or "" # None 방지
            outp = outp or "" # None 방지
            eos_tok = tokenizer.eos_token or tokenizer.sep_token or "" # 특수 토큰 처리
            texts.append(inp + eos_tok + outp + eos_tok)

        model_inputs = tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    # --- 전처리 함수 정의 끝 ---
    logger.info(f"Starting training for {model_config.name}")
    logger.info(f"Using {JSON_TRAIN_DATASET_PATH} and splitting into 90% train / 10% validation.")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # --- 데이터 로딩 및 분할 (datasets 라이브러리 사용) ---
    logger.info(f"Loading dataset from: {JSON_TRAIN_DATASET_PATH}")
    try:
        # JSON 파일 직접 로드 (리스트 형태의 JSON 가정)
        # 주의: load_dataset("json", ...)은 각 줄이 JSON 객체일 때 더 적합합니다.
        # 만약 파일 전체가 하나의 큰 JSON 리스트라면 직접 로드 후 Dataset 객체로 변환 필요.
        # 여기서는 파일이 datasets 라이브러리가 인식하는 형태라고 가정합니다.
        # (만약 오류 발생 시, 아래 json.load 후 Dataset.from_list 사용하는 방법으로 변경)
        full_dataset = load_dataset(
             "json",
             data_files={"train": JSON_TRAIN_DATASET_PATH},
             cache_dir=DATA_CACHE_DIR
        )["train"] # 'train' 스플릿 가져오기

        # --- 데이터셋 분할 (datasets 라이브러리 기능 사용) ---
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True) # 90/10 분할
        raw_train_dataset = split_dataset["train"]
        raw_val_dataset = split_dataset["test"] # 10%를 검증용으로 사용

        logger.info(f"Loaded and split dataset: train={len(raw_train_dataset)}, validation={len(raw_val_dataset)}")

    except Exception as e:
        logger.error(f"Failed to load or split dataset from {JSON_TRAIN_DATASET_PATH}: {e}")
        raise e

    # --- 전처리 (datasets.map 사용) ---
    logger.info("Preprocessing datasets...")
    preprocess_with_tokenizer = partial(preprocess_function_for_training, tokenizer=tokenizer, max_length=MAX_LENGTH)

    tokenized_train_dataset = raw_train_dataset.map(
        preprocess_with_tokenizer, batched=True,
        remove_columns=raw_train_dataset.column_names # 원본 컬럼 제거
    )
    tokenized_val_dataset = raw_val_dataset.map(
        preprocess_with_tokenizer, batched=True,
        remove_columns=raw_val_dataset.column_names # 원본 컬럼 제거
    )
    logger.info(f"Tokenized datasets prepared: train={len(tokenized_train_dataset)}, validation={len(tokenized_val_dataset)}")

    
    # TrainingArguments (이전 설정 사용)
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps", # eval_strategy 오타 수정
        eval_steps=400,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # --- 데이터 콜레이터 정의 (이 부분을 추가!) ---
    logger.info("Initializing Data Collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM이므로 Masked LM은 False
    )
    # --- 데이터 콜레이터 정의 끝 ---

    # 트레이너 초기화 (토크나이즈된 데이터셋 전달)
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset, # <-- 토크나이즈된 데이터 전달
        eval_dataset=tokenized_val_dataset,   # <-- 토크나이즈된 데이터 전달
        data_collator=data_collator,
    )
    
    # 학습 실행
    trainer.train(resume_from_checkpoint=False)
    
    # 최종 모델 및 토크나이저 저장
    final_model_path = os.path.join(model_config.output_dir, "final_model")
    logger.info(f"Final Model: {final_model_path}")
    
    return model, tokenizer

# DST 평가 관련 함수
def normalize_state(state):
    """상태 표현을 정규화합니다"""
    # 빈 상태 처리
    if not state or state == "없음":
        return []
    
    # 문자열인 경우 리스트로 분할
    if isinstance(state, str):
        # 쉼표와 공백으로 구분된 항목들을 분할
        items = [item.strip() for item in state.split(",")]
        return [item for item in items if item]
    
    # 이미 리스트인 경우 그대로 반환
    return state

def calculate_joint_accuracy(true_states, pred_states):
    """조인트 정확도 계산 - 모든 슬롯이 정확히 일치해야 함"""
    correct = 0
    total = len(true_states)
    
    for true_state, pred_state in zip(true_states, pred_states):
        # 상태 정규화
        true_set = set(normalize_state(true_state))
        pred_set = set(normalize_state(pred_state))
        
        # 완전히 일치하는 경우에만 정답으로 간주
        if true_set == pred_set:
            correct += 1
    
    return correct / total if total > 0 else 0

def calculate_slot_f1(true_states, pred_states):
    """슬롯 F1 점수 계산 - 개별 슬롯 단위로 정확도 측정"""
    true_slots_all = []
    pred_slots_all = []
    
    for true_state, pred_state in zip(true_states, pred_states):
        true_slots = set(normalize_state(true_state))
        pred_slots = set(normalize_state(pred_state))
        
        true_slots_all.extend(list(true_slots))
        pred_slots_all.extend(list(pred_slots))
    
    # 정밀도, 재현율, F1 계산을 위한 TP, FP, FN 계산
    tp = len(set(true_slots_all) & set(pred_slots_all))
    fp = len(set(pred_slots_all) - set(true_slots_all))
    fn = len(set(true_slots_all) - set(pred_slots_all))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 메인 실행 함수
if __name__ == "__main__":
    # 각 모델별로 학습 및 평가 실행
    for model_config in MODEL_CONFIGS:
        # 출력 디렉토리 생성
        os.makedirs(model_config.output_dir, exist_ok=True)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        
        logger.info(f"Starting training for {model_config.name}")
        
        try:
            # 모델 학습
            model, tokenizer = train_model(model_config)
            
            logger.info(f"Completed training and evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")