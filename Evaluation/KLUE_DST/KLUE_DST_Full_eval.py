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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import logging
import re
from tqdm import tqdm
import evaluate
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정 클래스
class ModelConfig:
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_ollama = False  # 기본값으로 False 설정

# 모델 설정들 (기본 OLMo 1B, OLMo 7B)
MODEL_CONFIGS = [
    ModelConfig(
        name="full-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_dst_results/full-olmo1B-org-klue-dst"
    ),
    ModelConfig(
        name="full-OLMo-1b-v12", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo1B-v12", 
        output_dir="klue_dst_results/full-olmo1B-v12-klue-dst"
    ),
    ModelConfig(
        name="full-OLMo-7b-org", 
        model_path="allenai/OLMo-7B", 
        output_dir="klue_dst_results/full-olmo7B-org-klue-dst"
    ),
    ModelConfig(
        name="full-OLMo-7b-v13", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/olmo7B-v13", 
        output_dir="klue_dst_results/full-olmo7B-v13-klue-dst"
    ),
    ModelConfig(
        name="full-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_dst_results/full-llama3.2-3b-klue-dst"
    )
]

# 기본 설정
DATA_CACHE_DIR = "./klue_dst_origin_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
MAX_LENGTH = 768  # DST는 대화 컨텍스트가 더 길 수 있으므로 길이 증가
MAX_EVAL_SAMPLES = 200

# 데이터셋 준비 함수 - JSON 파일 생성
def prepare_dataset_json():
    """KLUE DST 데이터셋을 불러와서 JSON 파일로 변환합니다."""
    if os.path.exists(JSON_TRAIN_DATASET_PATH) and os.path.exists(JSON_VAL_DATASET_PATH):
        logger.info(f"Datasets already exist: {JSON_TRAIN_DATASET_PATH} and {JSON_VAL_DATASET_PATH}")
        return
    
    logger.info("Loading KLUE DST data...")
    
    # KLUE DST 데이터셋 로드 (사용자가 제공한 JSON 파일 형식 가정)
    try:
        raw_data_path = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_raw.json"
        with open(raw_data_path, "r", encoding="utf-8") as f:
            dst_data = json.load(f)
        
        # 학습 및 검증 데이터를 위한 리스트
        train_samples = []
        val_samples = []
        
        # 데이터 분할 (80% 학습, 20% 검증)
        split_idx = int(len(dst_data) * 0.8)
        train_raw = dst_data[:split_idx]
        val_raw = dst_data[split_idx:]
        
        # 함수: 프롬프트와 완성 텍스트 생성
        def create_prompt(dialogue_history, current_utterance):
            history_text = ""
            for turn in dialogue_history:
                if turn["role"] == "user":
                    history_text += f"사용자: {turn['text']}\n"
                else:
                    history_text += f"시스템: {turn['text']}\n"
            
            return f"다음은 사용자와 시스템 간의 대화입니다. 마지막 사용자 발화에 대한 대화 상태를 추적하세요.\n\n{history_text}사용자: {current_utterance}\n\n대화 상태:"
        
        def create_completion(state):
            if not state:
                return " 없음"
            else:
                return " " + ", ".join(state)
        
        # 학습 데이터 준비
        logger.info("Creating training samples...")
        for dialogue in tqdm(train_raw):
            dialogue_history = []
            
            for i, turn in enumerate(dialogue["dialogue"]):
                # 사용자 발화만 처리
                if turn["role"] == "user":
                    current_utterance = turn["text"]
                    state = turn["state"]
                    
                    # 현재까지의 대화 기록과 현재 발화를 바탕으로 프롬프트 생성
                    prompt = create_prompt(dialogue_history, current_utterance)
                    completion = create_completion(state)
                    
                    sample = {
                        "input": prompt,
                        "output": completion
                    }
                    train_samples.append(sample)
                
                # 대화 기록에 현재 턴 추가
                dialogue_history.append(turn)
        
        # 검증 데이터 준비
        logger.info("Creating validation samples...")
        for dialogue in tqdm(val_raw):
            dialogue_history = []
            
            for i, turn in enumerate(dialogue["dialogue"]):
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
                
                # 대화 기록에 현재 턴 추가
                dialogue_history.append(turn)
        
        # 학습 데이터 JSON 파일로 저장
        logger.info(f"Saving training dataset... (samples: {len(train_samples)})")
        with open(JSON_TRAIN_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        
        # 검증 데이터 JSON 파일로 저장
        logger.info(f"Saving validation dataset... (samples: {len(val_samples)})")
        with open(JSON_VAL_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created KLUE DST datasets: {JSON_TRAIN_DATASET_PATH} and {JSON_VAL_DATASET_PATH}")
    
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise

# 모델 및 토크나이저 로드 함수
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model: {model_config.model_path}")

    # 일반적인 HuggingFace 모델 로드 (OLMo 1B, OLMo 7B)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path, trust_remote_code=True)
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # bfloat16 정밀도로 모델 로드 (메모리 효율성 증가)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 자동으로 GPU에 모델 분산
        trust_remote_code=True  # OLMo 모델에 필요
    )
    
    return model, tokenizer

# 메인 학습 함수
def train_model(model_config):
    # 데이터셋 준비
    prepare_dataset_json()
    
    # 데이터셋 로드
    logger.info("JSON loading...")
    with open(JSON_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    train_data = data["train"]
    val_data = data["validation"]
    
    logger.info(f"train data: {len(train_data)}, valid data: {len(val_data)}")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 학습용 데이터셋 생성
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            prompt = item["input"]
            completion = item["output"]
            
            # 프롬프트와 완성 결합
            full_text = prompt + completion
            
            # 토큰화
            encoded = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 프롬프트 부분 토큰화 (라벨 마스킹용)
            prompt_encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 라벨 생성: 프롬프트 부분은 -100으로 마스킹
            labels = encoded["input_ids"].clone().squeeze(0)
            prompt_length = prompt_encoded["input_ids"].shape[1]
            labels[:prompt_length] = -100
            
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": labels
            }
    
    # 데이터셋 생성
    train_dataset = SimpleDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = SimpleDataset(val_data, tokenizer, MAX_LENGTH)
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 학습 하이퍼파라미터 설정
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # 대화 데이터가 더 길기 때문에 배치 크기 감소
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 누적 증가로 효과적인 배치 크기 유지
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,  # bfloat16 정밀도 사용
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,  # 메모리 절약을 위한 그래디언트 체크포인팅
        optim="adamw_torch",  # PyTorch 구현 사용
    )
    
    # 얼리 스토핑 콜백
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5
    )
    
    # 트레이너 초기화 및 학습
    logger.info("Reset Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
    )
    
    # 학습 실행
    trainer.train()
    
    # 최종 모델 및 토크나이저 저장
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Final Model: {final_model_path}")
    
    # Ollama 모델이 아닌 경우에만 저장 (로컬 모델)
    if not model_config.is_ollama:
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
    else:
        # Ollama 모델의 경우 토크나이저만 저장
        tokenizer.save_pretrained(final_model_path)
        # Ollama 모델 정보 저장
        with open(os.path.join(final_model_path, "ollama_config.json"), "w") as f:
            json.dump({
                "model": model_config.model_path,
                "host": model_config.ollama_host
            }, f)
    
    logger.info("Tuned!")
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

# 모델 평가 함수 (DST 태스크에 맞게 수정)
def evaluate_model(model, tokenizer, model_config):
    logger.info(f"Evaluating the model: {model_config.name}")
    
    # 데이터셋 로드
    with open(JSON_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    val_data = data["validation"][:MAX_EVAL_SAMPLES]  # 평가 샘플 제한
    
    model.eval()
    
    true_states = []
    pred_states = []
    
    log_file_path = os.path.join(model_config.output_dir, "dst_log.json")
    logs = []
    
    for item in tqdm(val_data):
        prompt = item["input"]
        true_state = item["output"].strip()
        
        # 빈 상태 확인
        if true_state == "없음":
            true_state = []
        else:
            # 문자열에서 첫 번째 공백 제거 (출력 포맷에 따라 조정 가능)
            if true_state.startswith(" "):
                true_state = true_state[1:]
            # 쉼표로 분할
            true_state = true_state.split(", ")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 추론
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,  # DST 상태가 더 길 수 있으므로 증가
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 결과 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분을 제거하고 완성 부분만 추출
        completion_text = generated_text[len(prompt):].strip()
        
        # 빈 상태 처리
        if completion_text == "없음" or not completion_text:
            pred_state = []
        else:
            # 첫 번째 공백 제거 (있는 경우)
            if completion_text.startswith(" "):
                completion_text = completion_text[1:]
            # 쉼표로 분할하여 상태 목록 생성
            pred_state = completion_text.split(", ")
        
        # 예측과 실제 상태 저장
        true_states.append(true_state)
        pred_states.append(pred_state)
        
        # 로그 저장을 위한 데이터
        log_data = {
            "prompt": prompt,
            "generated_text": generated_text,
            "true_state": true_state,
            "pred_state": pred_state
        }
        
        logs.append(log_data)
    
    # 로그를 파일에 저장
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    # DST 메트릭 계산
    joint_accuracy = calculate_joint_accuracy(true_states, pred_states)
    slot_metrics = calculate_slot_f1(true_states, pred_states)
    
    logger.info(f"Eval result:")
    logger.info(f"Eval Model: {model_config.model_path}")
    logger.info(f"Evaluated samples: {len(true_states)}")
    logger.info(f"Joint Accuracy: {joint_accuracy:.4f}")
    logger.info(f"Slot F1: {slot_metrics['f1']:.4f}")
    logger.info(f"Slot Precision: {slot_metrics['precision']:.4f}")
    logger.info(f"Slot Recall: {slot_metrics['recall']:.4f}")
    
    # 평가 결과를 파일에 저장
    eval_results = {
        "model": model_config.name,
        "joint_accuracy": float(joint_accuracy),
        "slot_f1": float(slot_metrics["f1"]),
        "slot_precision": float(slot_metrics["precision"]),
        "slot_recall": float(slot_metrics["recall"]),
        "num_samples": len(true_states)
    }
    
    # 결과를 JSON 파일로 저장
    eval_file_path = os.path.join(model_config.output_dir, "dst_eval_results.json")
    with open(eval_file_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    return eval_results

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
            
            # 모델 평가
            evaluate_model(model, tokenizer, model_config)
            
            logger.info(f"Completed training and evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")