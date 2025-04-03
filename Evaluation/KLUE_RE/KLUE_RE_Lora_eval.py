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
# from peft.utils.other import fsdp_auto_wrap_policy # 주석 처리 (현재 사용 안 함)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    # AutoModelForSequenceClassification, # 주석 처리 (Causal LM 사용)
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig # SFTConfig 임포트 추가 (필요시)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_re_training.log")
    ]
)
logger = logging.getLogger(__name__)

# KLUE RE Label Definitions
RE_LABELS = [
    "no_relation", "org:dissolved", "org:founded", "org:place_of_headquarters",
    "org:alternate_names", "org:member_of", "org:members",
    "org:political/religious_affiliation", "org:product", "org:founded_by",
    "org:top_members/employees", "org:number_of_employees/members",
    "per:date_of_birth", "per:date_of_death", "per:place_of_birth",
    "per:place_of_death", "per:place_of_residence", "per:origin",
    "per:employee_of", "per:schools_attended", "per:alternate_names",
    "per:parents", "per:children", "per:siblings", "per:spouse",
    "per:other_family", "per:colleagues", "per:product", "per:religion",
    "per:title"
]
NUM_LABELS = len(RE_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(RE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(RE_LABELS)}
LABEL_NAME_TO_ID = {v: k for k, v in ID2LABEL.items()} # 레이블 이름 -> ID 맵 추가
logger.info(f"Total number of KLUE-RE labels: {NUM_LABELS}")


# Model configuration class
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations (기존과 동일)
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-OLMo-1b-org",
        model_path="allenai/OLMo-1B",
        output_dir="klue_re_results/lora-olmo1B-org-klue-re",
        is_local=False
    ),
    ModelConfig(
        name="lora-OLMo-1b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B",
        output_dir="klue_re_results/lora-olmo1B-v12-klue-re",
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_tc_results/lora-olmo7B-org-klue-tc",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_tc_results/lora-olmo7B-v13-klue-tc",
    #     is_local=True
    # ),
    ModelConfig(
        name="lora-Llama-3.2-3b",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b",
        output_dir="klue_re_results/lora-llama3.2-3b-klue-re",
        is_local=True
    )
]

# Configuration parameters
DATA_CACHE_DIR = "./klue_re_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_re_validation.json"
MAX_SEQ_LENGTH = 128 # 적절히 조절 (기존 100에서 늘림)
MAX_NEW_TOKENS = 15 # 관계 레이블 생성 시 최대 토큰 수

# Model and tokenizer loading function (기존과 거의 동일)
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model: {model_config.model_path}")
    is_local = model_config.is_local

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path,
        local_files_only=is_local,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad token. Setting pad_token=eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # 모델 설정에도 반영 (필수는 아닐 수 있으나 명시적)
        # config.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True
    )

    return model, tokenizer

# --- train_model 수정 ---
def train_model(model_config):
    # compute_metrics 함수 제거 (SFTTrainer에서 직접 사용 X)

    logger.info("====================================")
    logger.info(f"Starting training for {model_config.name}")
    logger.info("====================================")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # ===================== Load model/tokenizer =====================
    model, tokenizer = load_model_and_tokenizer(model_config)

    # ===================== Load and format JSON =====================
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 데이터 전처리 함수 정의 (기존과 동일)
    def preprocess_data(examples):
        texts = []
        for sentence, subj, obj, label in zip(examples["sentence"], examples["subject_entity"], examples["object_entity"], examples["label"]):
            subj_start, subj_end = subj["start_idx"], subj["end_idx"]
            obj_start, obj_end = obj["start_idx"], obj["end_idx"]

            # 엔티티 위치에 특수 마커 추가 [S], [/S], [O], [/O] 사용
            if subj_start < obj_start:
                marked_sentence = (
                    sentence[:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                    sentence[subj_end+1:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                    sentence[obj_end+1:]
                )
            else: # obj가 먼저 나올 경우
                marked_sentence = (
                    sentence[:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                    sentence[obj_end+1:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                    sentence[subj_end+1:]
                )

            label_name = ID2LABEL[label] # ID2LABEL 사용 일관성
            instruction = f"문장에서 주어진 두 개체 간의 관계를 분류하세요.\n\n문장: {marked_sentence}\n\n관계: {label_name}"
            texts.append(instruction)

        return {"text": texts}

    # 특수 토큰 추가 ([S], [/S], [O], [/O])
    special_tokens = ["[S]", "[/S]", "[O]", "[/O]"]
    num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True) # special_tokens=True 권장
    logger.info(f"Added {num_added_toks} special tokens.")
    if num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized model token embeddings to {len(tokenizer)}")

    # 데이터셋 변환 및 전처리 (기존과 동일)
    formatted_data = {
        "sentence": [d["sentence"] for d in raw_data],
        "subject_entity": [d["subject_entity"] for d in raw_data],
        "object_entity": [d["object_entity"] for d in raw_data],
        "label": [d["label"] for d in raw_data],
    }
    dataset = Dataset.from_dict(formatted_data)
    processed_dataset = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=["sentence", "subject_entity", "object_entity", "label"],
        desc="Processing dataset"
    )

    # 학습/검증 데이터셋 분할 (기존과 동일)
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42) # seed 추가 권장
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    logger.info(f"Train: {len(train_data)} samples | Val: {len(val_data)} samples")
    logger.info(f"Example processed data point: {train_data[0]['text']}") # 예시 데이터 확인

    # LoRA 설정 (기존과 동일)
    peft_params = LoraConfig(
        lora_alpha=8, lora_dropout=0.05, r=4, bias="none", task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"] # OLMo 용
    )
    if "llama" in model_config.name.lower(): # 모델 이름 기반으로 target_modules 설정
        peft_params = LoraConfig(
            lora_alpha=8, lora_dropout=0.05, r=4, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Llama 3.2에 더 일반적인 설정
        )
        logger.info("Using LoRA target modules for Llama.")
    else:
        logger.info("Using LoRA target modules for OLMo.")


    # PEFT 모델 준비
    # model = prepare_model_for_kbit_training(model) # QLoRA 사용할 경우 주석 해제
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=4, # 배치 크기 조정 (메모리 부족 시 줄이기)
        per_device_eval_batch_size=4, # 배치 크기 조정
        gradient_accumulation_steps=4, # 배치 크기 줄인 만큼 늘리기 (Effective batch size = 4 * 4 = 16)
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2, # 저장 제한
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        # fp16=True, # BF16 사용 시 주석 처리 또는 False
        bf16=True, # Ampere 이상 GPU에서 권장
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # SFTTrainer는 loss 기반으로 최적 모델 선택
        report_to="none", # wandb 등 사용 시 변경
        gradient_checkpointing=True, # 메모리 절약을 위해 사용 (학습 속도 느려짐)
        optim="adamw_torch",
        # ddp_find_unused_parameters=False # Multi-GPU 사용 시 필요할 수 있음
    )

    # SFTTrainer 초기화 수정
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

    # 최종 모델 저장 (PEFT 모델로)
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")

    # 트레이너를 통해 저장하는 것이 더 안전 (상태 저장 포함)
    trainer.save_model(final_model_path)
    # trainer.model.save_pretrained(final_model_path) # 이것도 가능
    tokenizer.save_pretrained(final_model_path)

    logger.info("Fine-tuning completed!")
    # 학습 완료 후 모델과 토크나이저 반환
    # return trainer.model, tokenizer # trainer.model은 PeftModel
    return model, tokenizer # get_peft_model로 받은 model 반환 (동일 객체)


# --- evaluate_model 수정 ---
def evaluate_model(model, tokenizer, model_config):
    """Evaluate the instruction-tuned model using text generation."""
    logger.info("=============================")
    logger.info(f"Evaluating model: {model_config.name} using generation")
    logger.info("=============================")

    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # 평가 데이터셋 크기 조절 (테스트용)
    # val_subset = val_data[:100] # 전체 데이터로 평가 시 주석 처리
    val_subset = val_data

    model.eval()
    # device = model.device # device_map="auto" 사용 시 device 직접 지정 불필요
    device = next(model.parameters()).device # 모델의 첫 파라미터가 있는 디바이스 사용

    true_label_ids = []
    pred_label_ids = []
    logs = []

    # 모델 생성 파라미터 설정
    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "do_sample": False, # 평가 시에는 greedy search 사용
        "early_stopping": True,
    }

    for item in tqdm(val_subset, desc="Evaluating"):
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]
        gold_label_id = item["label"]
        gold_relation_name = ID2LABEL[gold_label_id]

        # 엔티티 위치에 특수 마커 추가 ([S], [/S], [O], [/O] 사용 - train과 동일하게)
        subj_start, subj_end = subject_entity["start_idx"], subject_entity["end_idx"]
        obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]

        if subj_start < obj_start:
            marked_sentence = (
                sentence[:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                sentence[subj_end+1:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                sentence[obj_end+1:]
            )
        else:
            marked_sentence = (
                sentence[:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                sentence[obj_end+1:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                sentence[subj_end+1:]
            )

        # 모델 입력 프롬프트 구성 (정답 레이블 제외)
        prompt = f"문장에서 주어진 두 개체 간의 관계를 분류하세요.\n\n문장: {marked_sentence}\n\n관계: "

        # 토큰화
        encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS) # 생성 공간 확보
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)

        generated_ids = None
        generated_text = ""
        try:
            # 예측 (텍스트 생성)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
                # 생성된 부분만 추출 (입력 프롬프트 제외)
                generated_ids = outputs[0][input_ids.shape[1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 생성된 텍스트를 레이블 ID로 변환
            # 생성 결과가 정의된 레이블 이름과 정확히 일치하는지 확인
            predicted_label_id = LABEL_NAME_TO_ID.get(generated_text, LABEL2ID["no_relation"]) # 없으면 no_relation 처리

        except Exception as e:
            logger.warning(f"Error during generation for item {item.get('guid', 'N/A')}: {e}")
            logger.warning(f"Prompt: {prompt}")
            if generated_ids is not None:
                 logger.warning(f"Generated IDs: {generated_ids}")
            predicted_label_id = LABEL2ID["no_relation"] # 오류 발생 시 no_relation

        true_label_ids.append(gold_label_id)
        pred_label_ids.append(predicted_label_id)

        # 로그 기록 (생성된 텍스트 포함)
        logs.append({
            "guid": item.get('guid', 'N/A'),
            "sentence": sentence,
            "subject_entity": subject_entity["word"],
            "object_entity": object_entity["word"],
            "gold_label": gold_relation_name,
            "generated_text": generated_text, # 생성된 실제 텍스트
            "pred_label": ID2LABEL[predicted_label_id] # 매핑된 예측 레이블
        })

    # 메트릭 계산 (ID 기반)
    accuracy = accuracy_score(true_label_ids, pred_label_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_label_ids, pred_label_ids, average="macro", zero_division=0
    )

    # 클래스별 메트릭 계산
    labels_present = sorted(list(set(true_label_ids) | set(pred_label_ids)))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_label_ids, pred_label_ids, labels=labels_present, average=None, zero_division=0
    )

    per_class_metrics = {
        ID2LABEL[label_id]: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s)
        }
        for label_id, p, r, f, s in zip(labels_present, precision_per_class, recall_per_class, f1_per_class, support_per_class)
    }

    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro Precision: {precision:.4f}")
    logger.info(f"Macro Recall: {recall:.4f}")
    logger.info(f"Macro F1: {f1:.4f}")
    # logger.info("Per-class metrics:")
    # logger.info(json.dumps(per_class_metrics, indent=2, ensure_ascii=False)) # 한글 깨짐 방지

    results = {
        "model": model_config.name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_samples": len(val_subset),
        "per_class_metrics": per_class_metrics
    }

    # 로그 및 결과 저장 (기존과 동일)
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")

    return results

# Main execution (기존과 거의 동일, 평가 전용 로직 약간 수정)
if __name__ == "__main__":
    logger.info("Starting KLUE-RE evaluation only") # 로그 메시지 수정

    all_results = {}

    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model for evaluation: {model_config.name}")

        try:
            # 출력 디렉토리는 여전히 결과 저장을 위해 필요할 수 있음
            os.makedirs(model_config.output_dir, exist_ok=True)

            # === Train (주석 처리) ===
            # # 학습을 원할 경우 아래 주석 해제
            # trained_model, trained_tokenizer = train_model(model_config)
            # logger.info(f"Training finished for {model_config.name}. Starting evaluation...")
            # # 학습된 모델과 토크나이저로 바로 평가 진행
            # results = evaluate_model(trained_model, trained_tokenizer, model_config)


            # === Evaluate Only (활성화) ===
            # 저장된 모델로 평가만 수행
            logger.info("Loading base model and tokenizer for evaluation...")
            # 기본 모델과 토크나이저 로드
            base_model, tokenizer = load_model_and_tokenizer(model_config)

            # 평가 시에도 학습과 동일하게 특수 토큰 추가 및 임베딩 리사이즈 필요
            special_tokens = ["[S]", "[/S]", "[O]", "[/O]"]
            num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True)
            logger.info(f"Added {num_added_toks} special tokens for evaluation.")
            # 토큰이 추가된 경우에만 리사이즈 수행
            if num_added_toks > 0:
                base_model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Resized base model token embeddings to {len(tokenizer)} for evaluation.")

            # 저장된 PEFT 어댑터 경로 (train_model에서 저장한 경로)
            peft_model_path = os.path.join(model_config.output_dir, "final")
            logger.info(f"Attempting to load PEFT model from: {peft_model_path}")

            # PEFT 어댑터 디렉토리 존재 여부 확인
            if not os.path.exists(peft_model_path):
                 logger.error(f"PEFT model directory not found at {peft_model_path}. Cannot proceed with evaluation for {model_config.name}.")
                 # FileNotFoundError 대신 로깅하고 다음 모델로 건너뛰기
                 continue # 다음 model_config 로 이동

            # Load PEFT model (어댑터를 기본 모델에 적용)
            try:
                model = PeftModel.from_pretrained(
                    base_model,
                    peft_model_path,
                    torch_dtype=torch.bfloat16, # 기본 모델 로드 시 사용한 타입과 일치
                    device_map="auto"           # 자동 디바이스 매핑 사용
                )
                logger.info("PEFT model loaded successfully onto base model for evaluation")
            except Exception as load_error:
                logger.error(f"Failed to load PEFT model from {peft_model_path}: {load_error}")
                logger.exception("PEFT loading exception details:")
                continue # 다음 model_config 로 이동

            # Evaluate the loaded PEFT model
            logger.info(f"Starting evaluation for {model_config.name}...")
            results = evaluate_model(model, tokenizer, model_config)
            # === End Evaluate Only ===========================================
            

            all_results[model_config.name] = results

            logger.info(f"Completed evaluation processing for {model_config.name}")

            # 메모리 정리 (GPU 메모리 해제 시도)
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del model       # PEFT 모델 (또는 병합된 모델) 삭제
            del base_model  # 기본 모델 삭제
            del tokenizer   # 토크나이저 삭제
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Resources cleaned up for {model_config.name}")


        except Exception as e:
            # 루프 내의 다른 예외 처리 (예: evaluate_model 내부 오류)
            logger.error(f"An unexpected error occurred while processing {model_config.name}: {str(e)}")
            logger.exception("Overall processing exception details:")
            # 필요시 메모리 정리 시도
            if 'model' in locals(): del model
            if 'base_model' in locals(): del base_model
            if 'tokenizer' in locals(): del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # 다음 model_config 로 이동

    # Save combined results (기존과 동일)
    combined_results_path = os.path.join("klue_re_results", "combined_results.json")
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"All evaluation results saved to: {combined_results_path}")
    logger.info("KLUE-RE evaluation completed")