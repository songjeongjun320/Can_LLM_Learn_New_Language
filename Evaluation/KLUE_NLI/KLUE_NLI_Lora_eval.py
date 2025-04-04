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
    prepare_model_for_kbit_training,
    TaskType # TaskType은 Causal LM으로 사용
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForCausalLM, # SequenceClassification 대신 CausalLM 사용
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling # SFTTrainer와 함께 사용
)
# TRL SFTTrainer 임포트
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_nli_prompt_training.log") # 로그 파일명 변경
    ]
)
logger = logging.getLogger(__name__)

# KLUE NLI Label Definitions (텍스트 라벨 중요)
NLI_LABELS = ["entailment", "neutral", "contradiction"]
# 한국어 라벨 사용 시 (프롬프트에 한국어 라벨 사용 권장)
NLI_LABELS_KO = ["함의", "중립", "모순"]
NUM_LABELS = len(NLI_LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(NLI_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(NLI_LABELS)}
ID2LABEL_KO = {idx: label for idx, label in enumerate(NLI_LABELS_KO)} # 한국어 매핑 추가
# 평가 시 생성된 텍스트와 비교할 라벨 목록 (토크나이저 영향 고려)
# 모델이 정확히 '함의', '중립', '모순'을 생성하도록 유도
EXPECTED_GENERATED_LABELS = NLI_LABELS_KO

logger.info(f"Total number of KLUE-NLI labels: {NUM_LABELS}")
logger.info(f"Label to ID mapping: {LABEL2ID}")
logger.info(f"ID to Label mapping (EN): {ID2LABEL}")
logger.info(f"ID to Label mapping (KO): {ID2LABEL_KO}")

# Model configuration class (동일)
class ModelConfig:
    def __init__(self, name, model_path, output_dir, is_local):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir
        self.is_local = is_local

# Model configurations (동일)
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-OLMo-1b-org",
        model_path="allenai/OLMo-1B",
        output_dir="klue_nli_results/lora-olmo1B-org-klue-nli-prompt", # 출력 디렉토리명 변경
        is_local=False
    ),
    ModelConfig(
        name="lora-OLMo-1b-Tuned",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B",
        output_dir="klue_nli_results/lora-olmo1B-v12-klue-nli-prompt", # 출력 디렉토리명 변경
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B", 
    #     output_dir="klue_re_results/lora-olmo7B-org-klue-re",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_re_results/lora-olmo7B-v13-klue-re",
    #     is_local=True
    # ),
    ModelConfig(
        name="lora-Llama-3.2-3b",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b",
        output_dir="klue_nli_results/lora-llama3.2-3b-klue-nli-prompt", # 출력 디렉토리명 변경
        is_local=True
    )
]

# Configuration parameters (동일)
DATA_CACHE_DIR = "./klue_nli_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_nli_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_nli_validation.json"
MAX_LENGTH = 350 # 프롬프트 길이 고려하여 약간 늘릴 수 있음
MAX_EVAL_SAMPLES = 200 # 평가 샘플 수
EVAL_BATCH_SIZE = 8  # 평가 배치 크기 (Generate 시 메모리 사용량 고려)

# Model and tokenizer loading function (Causal LM으로 복귀)
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 Causal LM 모델과 토크나이저를 로드합니다."""
    logger.info(f"Load model for Causal LM: {model_config.model_path}")

    is_local = model_config.is_local

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path,
        local_files_only=is_local,
        trust_remote_code=True,
        # 모델에 따라 필요한 경우 padding_side 설정 (generate 시 중요)
        # padding_side='left' # OLMo나 Llama3.2 등 Causal LM에 종종 필요
    )

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # 일부 모델은 pad_token_id 명시적 설정 필요
        if model_config.name == "lora-Llama-3.2-3b": # 예시
             tokenizer.pad_token_id = tokenizer.eos_token_id

    # BitsAndBytesConfig 설정 (옵션, 4비트/8비트 로드 시)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True,
        # quantization_config=bnb_config, # 4비트/8비트 로드 시 주석 해제
        pad_token_id=tokenizer.pad_token_id # 패딩 토큰 ID 명시
    )

    # 모델 config에도 pad_token_id 설정 (generate 경고 방지)
    if hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

# 프롬프트 포맷팅 함수
def format_nli_prompt(example, tokenizer, include_answer=True):
    """주어진 예시를 NLI 프롬프트 문자열로 변환합니다."""
    premise = example['premise']
    hypothesis = example['hypothesis']
    label_id = example['label']
    # 한국어 라벨 사용
    label_text = ID2LABEL_KO.get(label_id, "알 수 없음")

    # 프롬프트 구조 정의 (모델 지시에 더 명확하게)
    instruction = "다음 주어진 '전제'와 '가설' 문장 사이의 논리적인 관계를 '함의', '중립', '모순' 중 하나로 분류하세요."
    prompt = f"{instruction}\n\n전제: {premise}\n가설: {hypothesis}\n\n관계:"

    if include_answer:
        # 학습 시: 정답 라벨 포함 및 EOS 토큰 추가
        full_prompt = f"{prompt} {label_text}{tokenizer.eos_token}"
        return {'text': full_prompt}
    else:
        # 평가 시: 정답 라벨 미포함 (모델이 생성할 부분)
        return {'text': prompt, 'label': label_id} # 평가 시 비교를 위해 원본 라벨 유지


# Training function
def train_model(model_config):
    """프롬프트 기반으로 Causal LM을 NLI 작업에 맞게 파인튜닝합니다."""
    logger.info("=========================================")
    logger.info(f"Starting Prompt-based training for {model_config.name}")
    logger.info("=========================================")

    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Load Causal LM model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)

    # kbit 트레이닝 준비 (4비트/8비트 로드 시 필요)
    # if hasattr(model, 'is_loaded_in_4bit') or hasattr(model, 'is_loaded_in_8bit'):
    #     logger.info("Preparing model for kbit training...")
    #     model = prepare_model_for_kbit_training(model)

    # Load datasets
    logger.info(f"Loading NLI training dataset from {JSON_TRAIN_DATASET_PATH}")
    try:
        with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load train data: {e}")
        return None, None

    logger.info(f"Loading NLI validation dataset from {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load validation data: {e}. Proceeding without validation.")
        val_data = []

    # Convert to Hugging Face Dataset
    raw_train_dataset = Dataset.from_list(train_data)
    raw_val_dataset = Dataset.from_list(val_data) if val_data else None

    # Apply prompt formatting
    logger.info("Formatting datasets into prompts...")
    # functools.partial 사용 가능 또는 lambda
    format_train_func = lambda ex: format_nli_prompt(ex, tokenizer, include_answer=True)
    train_dataset = raw_train_dataset.map(format_train_func, remove_columns=raw_train_dataset.column_names)

    if raw_val_dataset:
         # 검증셋은 Loss 계산용이므로 학습셋과 동일하게 포맷팅
        format_val_func = lambda ex: format_nli_prompt(ex, tokenizer, include_answer=True)
        val_dataset = raw_val_dataset.map(format_val_func, remove_columns=raw_val_dataset.column_names)
        # 평가 샘플 수 제한 (필요 시)
        # if MAX_EVAL_SAMPLES and MAX_EVAL_SAMPLES > 0 and len(val_dataset) > MAX_EVAL_SAMPLES:
        #     val_dataset = val_dataset.select(range(MAX_EVAL_SAMPLES))
        #     logger.info(f"Limited validation set for loss calculation to {len(val_dataset)} samples.")
    else:
        val_dataset = None

    logger.info(f"Formatted data - train: {len(train_dataset)} examples")
    if val_dataset: logger.info(f"Formatted data - validation (for loss): {len(val_dataset)} examples")


    # LoRA 설정 (Causal LM 용)
    lora_target_modules = ["att_proj", "attn_out"] # OLMo 기본값 시도
    if "llama" in model_config.name.lower():
        lora_target_modules = ["q_proj", "v_proj"] # Llama 일반적인 타겟 시도

    peft_params = LoraConfig(
        task_type="CAUSAL_LM", # Causal LM으로 변경
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=lora_target_modules
    )

    # 모델에 PEFT 적용
    model = get_peft_model(model, peft_params)
    logger.info("PEFT model created.")
    model.print_trainable_parameters()

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=200, # Loss 평가 주기
        learning_rate=2e-5,
        per_device_train_batch_size=4, # 배치 크기 (메모리 따라 조절)
        per_device_eval_batch_size=4,  # 배치 크기 (메모리 따라 조절)
        gradient_accumulation_steps=4, # 배치 크기 작을 시 늘림 (유효 배치=4*4=16)
        num_train_epochs=3,            # 에폭 수
        weight_decay=0.01,
        save_total_limit=2,
        save_strategy="steps",
        save_steps=400,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=False, # bf16 사용 시 False
        bf16=True,  # bf16 활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None, # Loss 기준 최적 모델 저장
        report_to="none",
        # SFTTrainer 사용 시 remove_unused_columns는 보통 필요 없음
        # gradient_checkpointing=True, # 메모리 부족 시 활성화 고려
    )

    # Data Collator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # SFTTrainer 초기화
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        # 여기!! train_data (list) 대신 train_dataset (Dataset) 객체를 전달해야 함
        train_dataset=train_dataset,
        # 여기!! val_data (list) 대신 val_dataset (Dataset) 객체를 전달해야 함
        eval_dataset=val_dataset,
        peft_config=peft_params,
        data_collator=data_collator,
    )

    # 학습 실행
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.exception("Training failed.")
        return None, None

    # 최종 모델 저장 (PEFT adapter)
    final_adapter_path = os.path.join(model_config.output_dir, "final_adapter")
    logger.info(f"Saving final PEFT adapter to: {final_adapter_path}")
    trainer.save_model(final_adapter_path) # PEFT 어댑터 저장
    tokenizer.save_pretrained(final_adapter_path) # 토크나이저도 함께 저장

    # 학습 완료된 PEFT 모델 반환 (평가에서 사용)
    # trainer.model 은 PEFT 모델 객체
    return trainer.model, tokenizer


# Evaluation function using model.generate
def evaluate_model(model, tokenizer, model_config):
    """학습된 모델을 사용하여 NLI 예측을 생성하고 평가합니다."""
    logger.info("============================================")
    logger.info(f"Evaluating prompt-based model: {model_config.name}")
    logger.info("============================================")

    os.makedirs(model_config.output_dir, exist_ok=True)

    logger.info(f"Loading validation data from: {JSON_VAL_DATASET_PATH}")
    try:
        with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        logger.info(f"Loaded {len(val_data)} validation samples.")
    except Exception as e:
        logger.error(f"Failed to load validation data for evaluation: {e}")
        return None

    val_subset = val_data
    logger.info("Using the full validation dataset for evaluation.")

    # 모델을 평가 모드로 설정
    model.eval()
    if torch.cuda.is_available():
        # 모델이 이미 device_map="auto"로 로드되었으므로 device 지정 불필요할 수 있음
        # device = torch.device("cuda")
        # model.to(device)
        pass # device_map="auto" 사용 시 불필요
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU for evaluation.")
        # model.to(device) # 이미 device_map이 처리했을 수 있음

    # 결과 저장을 위한 리스트
    true_labels_ids = []
    pred_labels_ids = []
    evaluation_logs = []

    # 평가 데이터 처리 (배치 단위)
    for i in tqdm(range(0, len(val_subset), EVAL_BATCH_SIZE), desc="Evaluating"):
        batch_items = val_subset[i:i + EVAL_BATCH_SIZE]

        # 배치 데이터로 평가용 프롬프트 생성 (정답 미포함)
        batch_prompts = [format_nli_prompt(item, tokenizer, include_answer=False)['text'] for item in batch_items]
        batch_gold_label_ids = [item['label'] for item in batch_items]

        # 토크나이징 (왼쪽 패딩 중요 for Causal LM generation)
        tokenizer.padding_side = "left" # Generate를 위해 왼쪽 패딩 설정
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True, # 배치 내 가장 긴 프롬프트 기준으로 패딩
            truncation=True,
            max_length=MAX_LENGTH - 10 # 생성 여유 공간 확보
        )
        tokenizer.padding_side = "right" # 다음 처리를 위해 오른쪽 패딩으로 복구 (필요 시)

        # 토큰을 모델과 같은 디바이스로 이동
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        # 텍스트 생성
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,  # 예측 라벨 길이 고려 (ex: 'contradiction' 토큰 수)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, # 일관된 예측을 위해 샘플링 비활성화
                num_beams=1      # Beam search 비활성화 (greedy)
            )

        # 생성된 결과에서 예측 라벨 부분만 디코딩
        # generated_ids 에는 입력 프롬프트 부분도 포함되어 있음
        generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

        # 생성된 텍스트 파싱 및 라벨 ID 변환
        batch_pred_label_ids = []
        for idx, gen_text in enumerate(generated_texts):
            premise = batch_items[idx]['premise']
            hypothesis = batch_items[idx]['hypothesis']
            gold_label_id = batch_gold_label_ids[idx]
            gold_label_text = ID2LABEL_KO.get(gold_label_id, "INVALID_GOLD")

            # 생성된 텍스트 앞뒤 공백 제거 및 첫 단어 추출 시도
            parsed_pred_text = gen_text.strip().split()[0] if gen_text.strip() else "파싱실패"

            # 예측 텍스트를 라벨 ID로 변환 (가장 유사한 라벨 선택)
            predicted_id = -1 # 기본값: 매칭 실패
            # 대소문자, 공백 등 고려하여 매칭
            for label_text, label_id in zip(EXPECTED_GENERATED_LABELS, range(NUM_LABELS)):
                 # 간단한 시작 문자열 비교 또는 더 복잡한 유사도 측정 가능
                if parsed_pred_text.startswith(label_text):
                    predicted_id = label_id
                    break # 첫 매칭 사용

            if predicted_id == -1:
                 logger.warning(f"Could not parse generated text '{gen_text}' into a valid label. Original prompt ended with '{batch_prompts[idx][-20:]}'. Assigning -1.")

            batch_pred_label_ids.append(predicted_id)

            # 로그 기록
            evaluation_logs.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "prompt": batch_prompts[idx],
                "generated_text": gen_text,
                "parsed_prediction": parsed_pred_text,
                "gold_label_id": gold_label_id,
                "gold_label_text": gold_label_text,
                "predicted_label_id": predicted_id,
                "predicted_label_text": ID2LABEL_KO.get(predicted_id, "파싱실패/매칭실패")
            })

        true_labels_ids.extend(batch_gold_label_ids)
        pred_labels_ids.extend(batch_pred_label_ids)

    # 유효한 예측만 필터링 (파싱/매칭 실패 제외)
    valid_indices = [i for i, pid in enumerate(pred_labels_ids) if pid != -1]
    if len(valid_indices) < len(pred_labels_ids):
        logger.warning(f"Excluded {len(pred_labels_ids) - len(valid_indices)} samples due to parsing/matching failure.")

    filtered_true_labels = [true_labels_ids[i] for i in valid_indices]
    filtered_pred_labels = [pred_labels_ids[i] for i in valid_indices]

    if not filtered_true_labels:
        logger.error("No valid predictions were made. Cannot calculate metrics.")
        return None

    # 메트릭 계산
    try:
        accuracy = accuracy_score(filtered_true_labels, filtered_pred_labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            filtered_true_labels, filtered_pred_labels, average="macro", zero_division=0
        )
        logger.info(f"Evaluation results for {model_config.name}:")
        logger.info(f"Total Samples Evaluated (Valid): {len(filtered_true_labels)}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro Precision: {precision_macro:.4f}")
        logger.info(f"Macro Recall: {recall_macro:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")

        results = {
            "model": model_config.name,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "total_samples_evaluated": len(val_subset),
            "valid_samples_evaluated": len(filtered_true_labels),
            "parsing_failures": len(pred_labels_ids) - len(filtered_true_labels)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        results = None

    # 로그 및 결과 저장
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log_prompt.json")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation logs saved to: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation logs: {e}")

    if results:
        results_file_path = os.path.join(model_config.output_dir, "eval_results_prompt.json")
        try:
            with open(results_file_path, "w", encoding="utf-8") as f:
                # 결과를 JSON으로 저장하기 전에 numpy 타입을 float으로 변환
                serializable_results = {k: float(v) if isinstance(v, np.generic) else v for k, v in results.items()}
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation results saved to: {results_file_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    return results


# Main execution
# 메인 실행 함수
if __name__ == "__main__":
    # 작업 유형에 맞게 로그 메시지 수정 (예: KLUE-NLI)
    logger.info("Starting KLUE-NLI processing") # <-- 작업 이름에 맞게 수정하세요

    # <<< CONTROL FLAG >>>
    # Set to True to skip training and ONLY run evaluation using pre-trained adapters.
    # Assumes adapters are saved in 'model_config.output_dir / final'.
    # Set to False to run training first, then evaluation.
    EVAL_ONLY = True # <-- 이 값을 True 또는 False로 변경하세요

    if EVAL_ONLY:
        logger.info(">>> Running in EVALUATION-ONLY mode <<<")
        logger.info(">>> Training step will be SKIPPED. Attempting to load adapters... <<<")
    else:
        logger.info(">>> Running in TRAINING and EVALUATION mode <<<")

    all_results = {}

    # --- 단일 루프 시작 ---
    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")

        # 변수 초기화
        model_for_eval = None
        tokenizer_for_eval = None
        eval_results = None
        base_model = None # EVAL_ONLY 모드용
        base_tokenizer = None # EVAL_ONLY 모드용

        try:
            os.makedirs(model_config.output_dir, exist_ok=True)
            # NLI 태스크의 경우 데이터 캐시 디렉토리가 필요하면 생성
            # os.makedirs(DATA_CACHE_DIR, exist_ok=True) # 필요 시 주석 해제

            # --- EVAL_ONLY 플래그에 따른 분기 ---
            if EVAL_ONLY:
                # --- Evaluation-Only Mode ---
                logger.info(f"Attempting to load model and adapter for evaluation-only: {model_config.name}")

                # 1. Load the base model and tokenizer
                logger.info("Loading base model and tokenizer...")
                # load_model_and_tokenizer는 해당 작업에 맞는 함수 사용 가정
                base_model, base_tokenizer = load_model_and_tokenizer(model_config)
                tokenizer_for_eval = base_tokenizer # 토크나이저는 기본 토크나이저 사용

                # 2. Define the path to the pre-trained adapter
                # train_model에서 저장하는 경로와 일치해야 함 (보통 'final')
                adapter_path = os.path.join(model_config.output_dir, "final")
                logger.info(f"Looking for pre-trained adapter at: {adapter_path}")

                # 3. Check if the adapter directory exists and load the PEFT model
                if os.path.isdir(adapter_path): # 디렉토리 존재 확인
                    try:
                        logger.info(f"Loading PEFT adapter from {adapter_path}...")
                        # 기본 모델 위에 어댑터 로드
                        # 필요한 경우 torch_dtype 등 옵션 추가
                        peft_model = PeftModel.from_pretrained(
                            base_model,
                            adapter_path,
                            # torch_dtype=torch.bfloat16 # 필요 시 기본 모델과 맞춤
                        )
                        logger.info("PEFT adapter loaded successfully onto the base model.")
                        model_for_eval = peft_model # 평가에 사용할 모델

                    except Exception as e:
                        logger.error(f"Failed to load PEFT adapter from {adapter_path}: {e}")
                        logger.exception("Adapter loading failed. Skipping evaluation for this model.")
                        # 실패 시 리소스 정리
                        if base_model: del base_model
                        if base_tokenizer: del base_tokenizer
                        base_model, base_tokenizer = None, None
                else:
                    logger.warning(f"Adapter directory not found at {adapter_path}. Skipping evaluation for {model_config.name}.")
                    # 어댑터 없으면 리소스 정리
                    if base_model: del base_model
                    if base_tokenizer: del base_tokenizer
                    base_model, base_tokenizer = None, None

            else: # if not EVAL_ONLY:
                # --- Training and Evaluation Mode ---
                logger.info(f"Starting training for {model_config.name}...")
                # train_model은 학습된 PEFT 모델 객체와 토크나이저를 반환해야 함
                # train_model 함수는 해당 작업(NLI)에 맞게 구현되어 있어야 함
                trained_peft_model, trained_tokenizer = train_model(model_config)

                if trained_peft_model and trained_tokenizer:
                    logger.info(f"Training successful for {model_config.name}.")
                    model_for_eval = trained_peft_model
                    tokenizer_for_eval = trained_tokenizer
                else:
                    logger.warning(f"Training failed or returned None for {model_config.name}. Skipping evaluation.")

            # --- Perform Evaluation (if model and tokenizer are ready) ---
            if model_for_eval and tokenizer_for_eval:
                logger.info(f"Starting evaluation for {model_config.name}...")
                # evaluate_model 함수는 해당 작업(NLI)에 맞게 구현되어 있어야 함
                # 필요한 경우 배치 크기 등의 인자 전달 (예: evaluate_model(..., eval_batch_size=8))
                eval_results = evaluate_model(model_for_eval, tokenizer_for_eval, model_config)

                if eval_results:
                    # 결과를 JSON으로 저장하기 전에 numpy 타입을 float으로 변환 (필요 시)
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
            # --- Memory Cleanup (루프 내에서 각 모델 처리 후 실행) ---
            logger.info(f"Cleaning up resources for {model_config.name}...")
            if model_for_eval is not None:
                # PeftModel 객체 자체를 삭제
                del model_for_eval
                model_for_eval = None
            if tokenizer_for_eval is not None:
                 # EVAL_ONLY 모드에서는 base_tokenizer와 같을 수 있으므로 아래에서 base 삭제 시 처리됨
                 # Training 모드에서는 독립적이므로 여기서 삭제
                 if not EVAL_ONLY and 'trained_tokenizer' in locals():
                    del trained_tokenizer # 명시적 삭제
                 tokenizer_for_eval = None # 참조 제거

            # EVAL_ONLY 모드에서 로드한 base 모델/토크나이저 삭제
            if EVAL_ONLY and base_model is not None:
                 del base_model
                 base_model = None
            if EVAL_ONLY and base_tokenizer is not None:
                 del base_tokenizer
                 base_tokenizer = None

            # 학습 모드에서 생성된 로컬 변수 명시적 삭제 (참조 카운트 감소)
            if not EVAL_ONLY:
                if 'trained_peft_model' in locals(): del trained_peft_model
                # 'trained_tokenizer'는 위에서 처리
                if 'trainer' in locals(): del trainer # trainer 객체가 train_model 스코프 밖에서 생성/사용되지 않았다면 불필요

            if eval_results is not None:
                del eval_results
                eval_results = None

            # GPU 캐시 비우기
            torch.cuda.empty_cache()
            logger.info(f"Finished cleaning up resources for {model_config.name}")
    # --- 단일 루프 끝 ---


    # Save combined results (루프 완료 후 최종 결과 저장)
    # 결과 파일 경로 수정 (예: KLUE-NLI)
    combined_results_path = "klue_nli_results/combined_eval_results.json" # <-- 작업 이름과 모드에 맞게 수정
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)

    try:
        # all_results에 numpy 타입이 있을 경우 float으로 변환
        serializable_all_results = {}
        for model_name, results in all_results.items():
             serializable_all_results[model_name] = {k: float(v) if isinstance(v, (np.generic, np.ndarray)) else v for k, v in results.items()}

        with open(combined_results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"All evaluation results saved to: {combined_results_path}")
    except Exception as e:
        logger.error(f"Failed to save combined results to {combined_results_path}: {e}")

    # 작업 유형에 맞게 로그 메시지 수정
    logger.info("KLUE-NLI processing completed") # <-- 작업 이름에 맞게 수정