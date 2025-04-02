# Standard library imports
import json
import logging
import os
import re
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from peft.utils.other import fsdp_auto_wrap_policy

from sklearn.model_selection import train_test_split  # 여기가 핵심!
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig
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
    #     name="lora-OLMo-1b-org", 
    #     model_path="allenai/OLMo-1B", 
    #     output_dir="klue_dst_results/lora-olmo1B-org-klue-dst",
    #     is_local=False
    # ),
    ModelConfig(
        name="lora-OLMo-1b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
        output_dir="klue_dst_results/lora-olmo1B-v12-klue-dst",
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_dst_results/lora-olmo7B-org-klue-dst",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_dst_results/lora-olmo7B-v13-klue-dst",
    #     is_local=True
    # ),
        ModelConfig(
        name="lora-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_dst_results/lora-llama3.2-3b-klue-dst",
        is_local=True
    )
]

# 기본 설정
DATA_CACHE_DIR = "./klue_dst_origin_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
MAX_LENGTH = 768  # DST는 대화 컨텍스트가 더 길 수 있으므로 길이 증가
MAX_TRAIN_SAMPLES = 300000
MAX_EVAL_SAMPLES = 500

# 데이터셋 준비 함수 - JSON 파일 생성
def prepare_dataset_json():
    """KLUE DST 데이터셋을 불러와서 JSON 파일로 변환합니다.
    이미 분할된 JSON_TRAIN_DATASET_PATH와 JSON_VAL_DATASET_PATH 파일을 사용합니다.
    """
    try:
        # 학습 데이터 로드
        logger.info("Loading KLUE DST training dataset from JSON file...")
        with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
            train_raw = json.load(f)

        # 검증 데이터 로드
        logger.info("Loading KLUE DST validation dataset from JSON file...")
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_raw = json.load(f)

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

        # 전처리된 검증 데이터 저장
        logger.info(f"Saving processed validation dataset... (samples: {len(val_samples)})")
        with open(JSON_VAL_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed KLUE DST datasets saved to: {JSON_TRAIN_DATASET_PATH} and {JSON_VAL_DATASET_PATH}")

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
    # 1. 원본 데이터 로드
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        full_train_data = json.load(f)  # 전체 훈련 데이터

    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.2,  # Val 20%
        random_state=42,  # 재현성 보장
        shuffle=True
    )
    logger.info(f"Loaded data - train: {len(train_data)} examples, validation: {len(val_data)} examples")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_config)

    # 데이터 샘플 확인
    logger.info(f"Sample train data: {train_dataset[0]}")
    
    # LoRA 설정 추가 (OLMo에 맞게 수정 필요)
    peft_params = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=4,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"]  # OLMo 확인 후 조정
    )
    if model_config.name == "lora-Llama-3.2:3B":
        peft_params = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=4,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj"]
        )

    # 모델에 LoRA 적용
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        evaluation_strategy="steps",
        eval_steps=300,
        learning_rate=3e-5,
        per_device_train_batch_size=4,  # 배치 크기 증가
        per_device_eval_batch_size=4,  # 배치 크기 증가
        gradient_accumulation_steps=4,  # 축적 단계 감소
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=600,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=50,
        fp16=True,  # FP16으로 전환
        bf16=False,  # BF16 비활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,  # Warmup 비율 감소
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,  # 체크포인팅 비활성화
        optim="adamw_torch",  # 필요 시 "adamw_8bit"로 변경
    )

    # SFTTrainer 초기화
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_params,
    )

    # 학습 실행
    logger.info("Starting training...")
    trainer.train()
    
    # 최종 모델 저장
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Fine-tuning completed!")
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
    logger.info("====================================")
    logger.info(f"Evaluating the model: {model_config.name}")
    logger.info("====================================")

    # 검증 데이터 로드
    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)[:MAX_EVAL_SAMPLES]  # 평가 샘플 제한
    
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
            # model, tokenizer = train_model(model_config)
            
            ##########
            # 1. 기본 모델과 토크나이저 로드
            base_model, tokenizer = load_model_and_tokenizer(model_config)
            
            # 2. PEFT 모델 경로 설정
            peft_model_path = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/KLUE_DST/klue_dst_results/lora-olmo1B-org-klue-dst/final"
            
            # 3. PEFT 모델 로드 (어댑터만 추가)
            model = PeftModel.from_pretrained(
                base_model,
                peft_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            ###########
            # 모델 평가
            evaluate_model(model, tokenizer, model_config)
            
            logger.info(f"Completed training and evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")