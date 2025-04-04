# Standard library imports
import json
import logging
import os
import re
from tqdm import tqdm
from datasets import Dataset


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
    Trainer,
    TrainingArguments
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
    ModelConfig(
        name="lora-OLMo-1b-org", 
        model_path="allenai/OLMo-1B", 
        output_dir="klue_dst_results/lora-olmo1B-org-klue-dst",
        is_local=False
    ),
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
        name="lora-Llama-3.2-3b", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_dst_results/lora-llama3.2-3b-klue-dst",
        is_local=True
    )
]

# 기본 설정
DATA_CACHE_DIR = "./klue_dst_origin_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_dst_validation.json"
MAX_LENGTH = 512  # DST는 대화 컨텍스트가 더 길 수 있으므로 길이 증가

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

# Modify preprocess function to accept the tokenizer
def preprocess(example, tokenizer):
    # 입력과 출력을 결합하여 토큰화
    combined_text = example["input"] + example["output"]
    
    encoding = tokenizer(
        combined_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 입력 부분만 있는 인코딩도 생성
    input_encoding = tokenizer(
        example["input"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # 입력 길이를 구함
    input_length = input_encoding["input_ids"].shape[1]
    
    # labels 배열 생성: 입력 부분은 -100으로 마스킹(loss 계산에서 제외)
    labels = encoding["input_ids"].clone()
    labels[0, :input_length] = -100
    
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0)
    }

# Then, in the `train_model` function, pass the tokenizer to preprocess
def train_model(model_config):
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Load the training data
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        full_train_data = json.load(f)  # 전체 훈련 데이터

    train_data, val_data = train_test_split(
        full_train_data, 
        test_size=0.2,  # Val 20%
        random_state=42,  # 재현성 보장
        shuffle=True
    )

    # Convert the train and validation data into Dataset objects
    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)

    # Apply preprocessing by passing tokenizer as an argument
    train_data = train_data.map(lambda x: preprocess(x, tokenizer))
    val_data = val_data.map(lambda x: preprocess(x, tokenizer))

    logger.info(f"Loaded data - train: {len(train_data)} examples, validation: {len(val_data)} examples")

    # 데이터 샘플 확인
    # logger.info(f"Sample train data: {train_data[0]}")
    
    # LoRA 설정 추가 (OLMo에 맞게 수정 필요)
    peft_params = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=4,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"]  # OLMo 확인 후 조정
    )
    if (model_config.name == "lora-Llama-3.2-3b"):
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
        gradient_checkpointing=False,  # 체크포인팅 비활성화
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
    """상태 표현(문자열 또는 리스트)을 정규화된 슬롯-값 문자열 리스트로 변환"""
    if not state:
        return set() # 빈 set 반환

    normalized_slots = set()
    if isinstance(state, str):
        # "슬롯1 - 값1, 슬롯2 - 값2" 형태 가정
        items = [item.strip() for item in state.split(',')]
    elif isinstance(state, list):
        items = state
    else:
        logger.warning(f"Unexpected state type: {type(state)}. Returning empty set.")
        return set()

    for item in items:
        if item: # 빈 문자열 제외
            normalized_slots.add(item)
    return normalized_slots

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
# 모델 평가 함수 (DST 태스크에 맞게 수정, 배치 처리 추가)
def evaluate_model(model, tokenizer, model_config, eval_batch_size=8): # 배치 크기 추가 (GPU 메모리에 맞게 조절)
    logger.info("====================================")
    logger.info(f"Evaluating the model: {model_config.name} with batch size {eval_batch_size}")
    logger.info("====================================")

    os.makedirs(model_config.output_dir, exist_ok=True)

    logger.info(f"Loading validation data from {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load validation data for evaluation: {e}")
        return None

    model.eval()

    all_true_states = [] # 실제 정답 상태 리스트
    all_pred_states = [] # 모델 예측 상태 리스트
    evaluation_logs = [] # 상세 로그 기록용

    # --- 배치 처리를 위한 설정 ---
    # Causal LM 배치 생성에는 Left Padding이 권장됨
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # OLMo 등 일부 모델은 pad 토큰이 없을 수 있음. eos 토큰으로 설정
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token for batch generation.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id # 모델 설정에도 반영

    # MAX_LENGTH는 입력+출력 길이를 고려해야 함. 입력 길이를 제한.
    # 너무 길면 생성이 안될 수 있으므로 입력 최대 길이를 설정.
    max_input_length = MAX_LENGTH - 150 # 생성할 토큰(100) + 여유분(50)을 제외한 최대 입력 길이

    # --- 배치 단위 평가 루프 ---
    for i in tqdm(range(0, len(val_data), eval_batch_size), desc="Evaluating DST Batches"):
        batch_items = val_data[i:min(i + eval_batch_size, len(val_data))] # 현재 배치 가져오기
        batch_prompts = [item["input"] for item in batch_items]
        batch_true_states = [item["output"].strip() for item in batch_items]

        # 배치 토크나이징 (Left Padding 적용)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,       # 배치 내에서 가장 긴 시퀀스 기준으로 패딩
            truncation=True,
            max_length=max_input_length, # 입력 최대 길이 제한
        ).to(model.device)

        # 배치 텍스트 생성
        with torch.no_grad():
            # Generation parameters
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'], # <-- 이렇게 수정
                max_new_tokens=100,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, # 패딩 토큰 ID 명시
                do_sample=False,
                num_beams=1,
            )

        # 배치 결과 디코딩
        # 생성된 부분만 추출 (입력 프롬프트 제외)
        # outputs 텐서에는 입력 프롬프트 + 생성된 텍스트가 포함됨
        # inputs['input_ids'].shape[1]은 패딩된 입력의 길이
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        batch_pred_states_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 정답과 예측 저장 및 로그 기록
        for idx, pred_state in enumerate(batch_pred_states_str):
            true_state = batch_true_states[idx]
            pred_state_cleaned = pred_state.strip() # 후처리

            all_true_states.append(true_state)
            all_pred_states.append(pred_state_cleaned)

            # 로그 기록 (필요 시 전체 생성 텍스트도 포함 가능)
            # full_generated_text = tokenizer.decode(outputs[idx], skip_special_tokens=True) # 디버깅용
            evaluation_logs.append({
                "prompt": batch_prompts[idx],
                "true_state_str": true_state,
                "predicted_state_str": pred_state_cleaned,
                # "full_generated_text": full_generated_text # 필요하면 주석 해제
            })

    # --- 평가 완료 후 처리 ---
    # 토크나이저 패딩 설정 복원
    tokenizer.padding_side = original_padding_side
    if tokenizer.pad_token == tokenizer.eos_token and original_padding_side == "right": # 임시로 변경했었다면
         tokenizer.pad_token = None # 원래 None이었다면 다시 None으로 (주의: 다른 곳에서 pad_token 필요시 문제될 수 있음)
         # 혹은 원래 pad_token 값을 저장했다가 복원하는 것이 더 안전

    # DST 메트릭 계산
    joint_accuracy = calculate_joint_accuracy(all_true_states, all_pred_states)
    slot_metrics = calculate_slot_f1(all_true_states, all_pred_states)

    logger.info(f"--- Evaluation Results for {model_config.name} (Batch Size: {eval_batch_size}) ---")
    logger.info(f"Evaluated samples: {len(all_true_states)}")
    logger.info(f"Joint Goal Accuracy (JGA): {joint_accuracy:.4f}")
    logger.info(f"Slot Micro F1: {slot_metrics['f1']:.4f}")
    logger.info(f"Slot Micro Precision: {slot_metrics['precision']:.4f}")
    logger.info(f"Slot Micro Recall: {slot_metrics['recall']:.4f}")
    logger.info("---------------------------------------------")

    # 평가 결과 저장 (이하 동일)
    eval_results = {
        "model": model_config.name,
        "eval_batch_size": eval_batch_size, # 배치 크기 정보 추가
        "joint_accuracy": float(joint_accuracy),
        "slot_f1": float(slot_metrics["f1"]),
        "slot_precision": float(slot_metrics["precision"]),
        "slot_recall": float(slot_metrics["recall"]),
        "num_samples": len(all_true_states)
    }

    # 로그 및 결과 파일 저장 (이하 동일)
    log_file_path = os.path.join(model_config.output_dir, f"dst_evaluation_log_bs{eval_batch_size}.json") # 파일명에 배치 크기 포함
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation logs saved to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation logs: {e}")

    results_file_path = os.path.join(model_config.output_dir, f"dst_eval_results_bs{eval_batch_size}.json") # 파일명에 배치 크기 포함
    try:
        with open(results_file_path, "w", encoding="utf-8") as f:
            serializable_results = {k: float(v) if isinstance(v, np.generic) else v for k, v in eval_results.items()}
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to: {results_file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

    return eval_results
    
# 메인 실행 함수
if __name__ == "__main__":
    logger.info("Starting KLUE-DST processing")

    # <<< CONTROL FLAG >>>
    # Set to True to skip training and ONLY run evaluation using pre-trained adapters.
    # Assumes adapters are saved in 'model_config.output_dir / final'.
    # Set to False to run training first, then evaluation (original behavior).
    EVAL_ONLY = True # <-- 이 값을 True 또는 False로 변경하세요

    if EVAL_ONLY:
        logger.info(">>> Running in EVALUATION-ONLY mode <<<")
        logger.info(">>> Training step will be SKIPPED. Attempting to load adapters... <<<")
    else:
        logger.info(">>> Running in TRAINING and EVALUATION mode <<<")

    all_results = {}

    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")

        # 변수 초기화
        model_for_eval = None
        tokenizer_for_eval = None
        eval_results = None
        # 평가 전용 모드에서 사용할 기본 모델 변수
        base_model = None
        base_tokenizer = None

        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            os.makedirs(DATA_CACHE_DIR, exist_ok=True)

            if not EVAL_ONLY:
                # --- Training and Evaluation Mode ---
                logger.info(f"Starting training for {model_config.name}...")
                # train_model은 학습된 PEFT 모델 객체와 토크나이저를 반환
                trained_peft_model, trained_tokenizer = train_model(model_config)

                if trained_peft_model and trained_tokenizer:
                    logger.info(f"Training successful for {model_config.name}.")
                    model_for_eval = trained_peft_model
                    tokenizer_for_eval = trained_tokenizer
                else:
                    logger.warning(f"Training failed or returned None for {model_config.name}. Cannot proceed to evaluation in this mode.")

            else:
                # --- Evaluation-Only Mode ---
                logger.info(f"Attempting to load model and adapter for evaluation-only: {model_config.name}")

                # 1. Load the base model and tokenizer
                logger.info("Loading base model and tokenizer...")
                base_model, base_tokenizer = load_model_and_tokenizer(model_config)
                tokenizer_for_eval = base_tokenizer # 토크나이저는 기본 토크나이저 사용

                # 2. Define the path to the pre-trained adapter ('final' 디렉토리)
                adapter_path = os.path.join(model_config.output_dir, "final") # 저장 경로 'final'
                logger.info(f"Looking for pre-trained adapter at: {adapter_path}")

                # 3. Check if the adapter directory exists and load the PEFT model
                if os.path.isdir(adapter_path): # 디렉토리 존재 확인
                    try:
                        logger.info(f"Loading PEFT adapter from {adapter_path}...")
                        # 기본 모델 위에 어댑터 로드
                        peft_model = PeftModel.from_pretrained(
                            base_model,
                            adapter_path,
                            torch_dtype=torch.bfloat16, # 기본 모델과 동일 타입 지정 권장
                            # device_map="auto" # 기본 모델 로드 시 적용되었으므로 보통 불필요
                        )
                        # PEFT 모델 병합 (선택 사항, 추론 속도 향상 가능하나 메모리 더 사용)
                        # logger.info("Merging PEFT adapter into the base model...")
                        # peft_model = peft_model.merge_and_unload()
                        # logger.info("PEFT model merged.")

                        logger.info("PEFT adapter loaded successfully onto the base model.")
                        model_for_eval = peft_model # 평가에 사용할 모델은 어댑터 적용된 모델

                    except Exception as e:
                        logger.error(f"Failed to load PEFT adapter from {adapter_path}: {e}")
                        logger.exception("Adapter loading failed. Skipping evaluation for this model.")
                        # 실패 시 base_model 메모리 해제
                        if base_model: del base_model
                        if base_tokenizer: del base_tokenizer
                        base_model, base_tokenizer = None, None
                else:
                    logger.warning(f"Adapter directory not found at {adapter_path}. Skipping evaluation for {model_config.name}.")
                    # 어댑터 없으면 base_model 메모리 해제
                    if base_model: del base_model
                    if base_tokenizer: del base_tokenizer
                    base_model, base_tokenizer = None, None


            # --- Perform Evaluation (if model and tokenizer are ready) ---
            if model_for_eval and tokenizer_for_eval:
                logger.info(f"Starting evaluation for {model_config.name}...")
                eval_results = evaluate_model(model_for_eval, tokenizer_for_eval, model_config, eval_batch_size=16)

                if eval_results:
                    # 결과를 JSON으로 저장하기 전에 numpy 타입을 float으로 변환
                    serializable_results = {k: float(v) if isinstance(v, (np.generic, np.ndarray)) else v for k, v in eval_results.items()}
                    all_results[model_config.name] = serializable_results
                    logger.info(f"Stored evaluation results for {model_config.name}")
                else:
                    logger.warning(f"Evaluation failed or produced no results for {model_config.name}")
            else:
                logger.warning(f"Model or tokenizer not available for evaluation for {model_config.name}. Skipping evaluation step.")

            logger.info(f"Completed processing for {model_config.name}")

        except Exception as e:
            logger.error(f"Unhandled error processing {model_config.name}: {str(e)}")
            logger.exception("Exception details:")
        finally:
            # --- Memory Cleanup ---
            logger.info(f"Cleaning up resources for {model_config.name}...")
            if model_for_eval is not None:
                del model_for_eval
            if tokenizer_for_eval is not None:
                del tokenizer_for_eval
            if EVAL_ONLY and base_model is not None:
                 del base_model
            if EVAL_ONLY and base_tokenizer is not None:
                 del base_tokenizer
            if 'trained_peft_model' in locals(): del trained_peft_model
            if 'trained_tokenizer' in locals(): del trained_tokenizer
            if 'trainer' in locals(): del trainer
            if eval_results is not None: del eval_results

            torch.cuda.empty_cache()
            logger.info(f"Finished cleaning up resources for {model_config.name}")


    # Save combined results
    combined_results_path = "klue_dst_results/combined_dst_eval_results.json" # 결과 파일명 변경
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    try:
        with open(combined_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"All evaluation results saved to: {combined_results_path}")
    except Exception as e:
        logger.error(f"Failed to save combined results to {combined_results_path}: {e}")

    logger.info("KLUE-DST processing completed")