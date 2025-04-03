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
    """Evaluate the instruction-tuned model using text generation AND likelihood estimation."""
    logger.info("=============================")
    logger.info(f"Evaluating model: {model_config.name} using generation & likelihood")
    logger.info("=============================")

    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    val_subset = val_data
    # val_subset = val_data[:20] # DEBUG: AUPRC 계산 속도 때문에 작게 테스트

    model.eval()
    device = next(model.parameters()).device
    loss_fct = CrossEntropyLoss(reduction='none') # Likelihood 계산용 loss 함수

    # 결과 저장을 위한 리스트
    true_label_ids = []
    pred_label_ids_generation = [] # 생성 기반 예측 저장
    all_likelihood_scores = []     # 모든 레이블에 대한 Likelihood 점수 저장
    logs = []

    # === Part 1: Standard Generation for Accuracy, Macro F1, Micro F1 ===
    logger.info("--- Starting Part 1: Evaluation based on Generation ---")
    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "do_sample": False,
        "early_stopping": True,
    }

    for item in tqdm(val_subset, desc="Part 1: Generating Predictions"):
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]
        gold_label_id = item["label"]
        true_label_ids.append(gold_label_id) # 실제 레이블 저장

        # 프롬프트 구성 (train과 동일)
        subj_start, subj_end = subject_entity["start_idx"], subject_entity["end_idx"]
        obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]
        if subj_start < obj_start:
            marked_sentence = ( sentence[:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                                sentence[subj_end+1:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                                sentence[obj_end+1:] )
        else:
            marked_sentence = ( sentence[:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                                sentence[obj_end+1:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                                sentence[subj_end+1:] )
        prompt = f"문장에서 주어진 두 개체 간의 관계를 분류하세요.\n\n문장: {marked_sentence}\n\n관계: "
        prompt_encoding = tokenizer(prompt, return_tensors="pt", truncation=False) # 프롬프트 부분만 토큰화 (길이 계산용)
        prompt_input_ids = prompt_encoding.input_ids.to(device)
        prompt_attention_mask = prompt_encoding.attention_mask.to(device)

        generated_text = ""
        predicted_label_id_gen = NO_RELATION_ID # 기본값

        try:
            with torch.no_grad():
                # 입력 길이 확인 및 조정 (모델 최대 길이 - 생성 길이)
                max_prompt_len = MAX_SEQ_LENGTH - MAX_NEW_TOKENS
                if prompt_input_ids.shape[1] > max_prompt_len:
                    logger.warning(f"Prompt length {prompt_input_ids.shape[1]} exceeds max {max_prompt_len}. Truncating.")
                    prompt_input_ids = prompt_input_ids[:, :max_prompt_len]
                    prompt_attention_mask = prompt_attention_mask[:, :max_prompt_len]

                outputs = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    **generation_config
                )
                generated_ids = outputs[0][prompt_input_ids.shape[1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            predicted_label_id_gen = LABEL_NAME_TO_ID.get(generated_text, NO_RELATION_ID)

        except Exception as e:
            logger.warning(f"Error during generation for item {item.get('guid', 'N/A')}: {e}")
            predicted_label_id_gen = NO_RELATION_ID

        pred_label_ids_generation.append(predicted_label_id_gen)

        # 로그 기록 (생성 기반) - AUPRC 계산 후 상세 로그에 점수 추가 예정
        logs.append({
            "guid": item.get('guid', 'N/A'),
            "sentence": sentence,
            "subject_entity": subject_entity["word"],
            "object_entity": object_entity["word"],
            "gold_label": ID2LABEL[gold_label_id],
            "generated_text": generated_text,
            "pred_label_generation": ID2LABEL[predicted_label_id_gen],
            # "likelihood_scores": {} # 나중에 채움
        })

    # === Part 2: Likelihood Estimation for AUPRC ===
    logger.info("--- Starting Part 2: Calculating Likelihood Scores for AUPRC (This will be slow) ---")
    for i, item in enumerate(tqdm(val_subset, desc="Part 2: Calculating Likelihoods")):
        sentence = item["sentence"]
        subject_entity = item["subject_entity"]
        object_entity = item["object_entity"]

        # 프롬프트 구성 (동일)
        subj_start, subj_end = subject_entity["start_idx"], subject_entity["end_idx"]
        obj_start, obj_end = object_entity["start_idx"], object_entity["end_idx"]
        if subj_start < obj_start:
             marked_sentence = ( sentence[:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                                 sentence[subj_end+1:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                                 sentence[obj_end+1:] )
        else:
             marked_sentence = ( sentence[:obj_start] + "[O]" + sentence[obj_start:obj_end+1] + "[/O]" +
                                 sentence[obj_end+1:subj_start] + "[S]" + sentence[subj_start:subj_end+1] + "[/S]" +
                                 sentence[subj_end+1:] )
        base_prompt = f"문장에서 주어진 두 개체 간의 관계를 분류하세요.\n\n문장: {marked_sentence}\n\n관계: "
        base_prompt_tokenized = tokenizer(base_prompt, return_tensors='pt', truncation=False)
        base_prompt_len = base_prompt_tokenized.input_ids.shape[1]

        scores_for_item = {} # 현재 샘플의 레이블별 점수 저장

        for label_id, label_name in ID2LABEL.items():
            target_text = base_prompt + label_name # 프롬프트 + 레이블 이름
            try:
                # 전체 텍스트 토큰화 및 모델 최대 길이 확인/조정
                target_encoding = tokenizer(
                    target_text,
                    return_tensors='pt',
                    truncation=True, # 전체 길이가 넘으면 잘라냄
                    max_length=MAX_SEQ_LENGTH,
                    padding=False # 패딩 불필요
                )
                target_input_ids = target_encoding.input_ids.to(device)
                target_attention_mask = target_encoding.attention_mask.to(device)

                # 실제 프롬프트 길이와 레이블 길이 (잘렸을 수 있으므로 다시 계산)
                current_prompt_len = base_prompt_len
                # 만약 target_input_ids가 잘렸다면, 프롬프트 길이도 잘린 길이 기준으로 조정
                if target_input_ids.shape[1] < base_prompt_len + len(tokenizer.encode(label_name, add_special_tokens=False)):
                    # 이 경우는 보통 프롬프트 자체가 너무 길어서 잘린 경우
                    current_prompt_len = target_input_ids.shape[1] - len(tokenizer.encode(label_name, add_special_tokens=False))
                    current_prompt_len = max(0, current_prompt_len) # 음수 방지

                label_len = target_input_ids.shape[1] - current_prompt_len
                if label_len <= 0: # 레이블 부분이 아예 잘려나간 경우
                    scores_for_item[label_id] = -float('inf') # 매우 낮은 점수 부여
                    continue

                with torch.no_grad():
                    outputs = model(input_ids=target_input_ids, attention_mask=target_attention_mask)
                    logits = outputs.logits

                # Shift logits and labels for next token prediction loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_input_ids[..., 1:].contiguous()

                # Calculate loss only for the label part
                # loss 계산 범위: [current_prompt_len - 1, target_input_ids.shape[1] - 2] (shift 기준)
                loss_start_index = current_prompt_len - 1
                loss_end_index = target_input_ids.shape[1] - 1 # shift_labels 기준 끝 인덱스

                if loss_start_index >= loss_end_index : # loss 계산할 토큰이 없는 경우
                     scores_for_item[label_id] = -float('inf')
                     continue

                # 배치 차원 제거 (view) 및 해당 부분만 loss 계산
                label_logits = shift_logits.view(-1, shift_logits.size(-1))[loss_start_index:loss_end_index]
                label_targets = shift_labels.view(-1)[loss_start_index:loss_end_index]

                loss = loss_fct(label_logits, label_targets) # shape: [label_len]
                avg_neg_log_likelihood = loss.mean().item()

                # 점수는 likelihood에 비례해야 하므로, -loss 사용 (높을수록 좋음)
                scores_for_item[label_id] = -avg_neg_log_likelihood

            except Exception as e:
                logger.warning(f"Error calculating likelihood for label '{label_name}' in item {item.get('guid', 'N/A')}: {e}")
                scores_for_item[label_id] = -float('inf') # 오류 시 매우 낮은 점수

        # 점수 벡터 정렬 및 저장
        score_vector = [scores_for_item.get(j, -float('inf')) for j in range(NUM_LABELS)]
        all_likelihood_scores.append(score_vector)
        # 상세 로그에 점수 추가 (필요시)
        # logs[i]["likelihood_scores"] = {ID2LABEL[k]: v for k, v in scores_for_item.items()}

    # === Part 3: Calculate All Metrics ===
    logger.info("--- Starting Part 3: Calculating Final Metrics ---")

    # 1. Accuracy & Macro F1 (from generation)
    accuracy = accuracy_score(true_label_ids, pred_label_ids_generation)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_label_ids, pred_label_ids_generation, average="macro", zero_division=0
    )

    # 2. Micro F1 (without no_relation) (from generation)
    precision_micro_nr, recall_micro_nr, f1_micro_nr, _ = precision_recall_fscore_support(
        true_label_ids,
        pred_label_ids_generation,
        labels=RE_LABELS_WITHOUT_NR, # no_relation 제외한 라벨 ID 리스트 사용
        average="micro",
        zero_division=0
    )

    # 3. Macro AUPRC (without no_relation) (from likelihood scores)
    macro_auprc_nr = None
    if all_likelihood_scores:
        y_scores_np = np.array(all_likelihood_scores)
        y_true_binarized = label_binarize(true_label_ids, classes=list(range(NUM_LABELS)))

        average_precisions = []
        valid_classes_count = 0
        for label_id in RE_LABELS_WITHOUT_NR:
            # 해당 클래스가 실제 레이블에 존재하는지 확인
            if np.sum(y_true_binarized[:, label_id]) > 0:
                # 해당 클래스의 실제값과 예측 점수 추출
                class_true = y_true_binarized[:, label_id]
                class_scores = y_scores_np[:, label_id]
                # average_precision_score 계산
                ap = average_precision_score(class_true, class_scores)
                average_precisions.append(ap)
                valid_classes_count += 1
            else:
                 logger.debug(f"Skipping AUPRC calculation for class {ID2LABEL[label_id]} as it's not present in true labels.")


        if valid_classes_count > 0:
            macro_auprc_nr = np.mean(average_precisions)
        else:
            logger.warning("No relevant classes found in true labels to calculate Macro AUPRC.")
            macro_auprc_nr = 0.0 # 계산 불가 시 0 처리 또는 None

    # 4. Per-class metrics (from generation)
    labels_present = sorted(list(set(true_label_ids) | set(pred_label_ids_generation)))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_label_ids, pred_label_ids_generation, labels=labels_present, average=None, zero_division=0
    )
    per_class_metrics = {
        ID2LABEL[label_id]: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label_id, p, r, f, s in zip(labels_present, precision_per_class, recall_per_class, f1_per_class, support_per_class)
    }

    # 결과 로깅
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Accuracy (Generation): {accuracy:.4f}")
    logger.info(f"Macro Precision (Generation): {precision_macro:.4f}")
    logger.info(f"Macro Recall (Generation): {recall_macro:.4f}")
    logger.info(f"Macro F1 (Generation, All Classes): {f1_macro:.4f}")
    logger.info(f"Micro F1 (Generation, without no_relation): {f1_micro_nr:.4f}")
    logger.info(f"Macro AUPRC (Likelihood, without no_relation): {macro_auprc_nr if macro_auprc_nr is not None else 'N/A'}")
    # logger.info("Per-class metrics (Generation):")
    # logger.info(json.dumps(per_class_metrics, indent=2, ensure_ascii=False))

    results = {
        "model": model_config.name,
        "accuracy_generation": float(accuracy),
        "precision_macro_generation": float(precision_macro),
        "recall_macro_generation": float(recall_macro),
        "f1_macro_generation_all_classes": float(f1_macro),
        "f1_micro_generation_without_no_relation": float(f1_micro_nr),
        "macro_auprc_likelihood_without_no_relation": float(macro_auprc_nr) if macro_auprc_nr is not None else None,
        "total_samples": len(val_subset),
        "per_class_metrics_generation": per_class_metrics
    }

    # 로그 및 결과 저장
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log_likelihood.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    results_file_path = os.path.join(model_config.output_dir, "eval_results_likelihood.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")

    return results


# Main execution (기존과 거의 동일, 평가 전용 로직 사용)
if __name__ == "__main__":
    logger.info("Starting KLUE-RE Training and Evaluation (Likelihood-based)")

    all_results = {}

    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model: {model_config.name}")

        try:
            os.makedirs(model_config.output_dir, exist_ok=True)

            # === Train & Evaluate ===
            # logger.info(f"Starting training for {model_config.name}...")
            # trained_model, trained_tokenizer = train_model(model_config)
            # logger.info(f"Training finished for {model_config.name}. Starting evaluation...")
            # # 학습된 모델과 토크나이저로 바로 평가 진행
            # results = evaluate_model(trained_model, trained_tokenizer, model_config)

            # === Evaluate Only (필요시 아래 코드 사용) ===
            logger.info("Loading base model and tokenizer for evaluation...")
            base_model, tokenizer = load_model_and_tokenizer(model_config)
            special_tokens = ["[S]", "[/S]", "[O]", "[/O]"]
            num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True)
            if num_added_toks > 0:
                base_model.resize_token_embeddings(len(tokenizer))
            peft_model_path = os.path.join(model_config.output_dir, "final")
            logger.info(f"Attempting to load PEFT model from: {peft_model_path}")
            if not os.path.exists(peft_model_path):
                 logger.error(f"PEFT model directory not found at {peft_model_path}.")
                 continue
            try:
                model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=torch.bfloat16, device_map="auto")
                logger.info("PEFT model loaded successfully onto base model for evaluation")
            except Exception as load_error:
                logger.error(f"Failed to load PEFT model from {peft_model_path}: {load_error}")
                continue
            logger.info(f"Starting evaluation for {model_config.name}...")
            results = evaluate_model(model, tokenizer, model_config)
            # === End Evaluate Only ===========================================

            all_results[model_config.name] = results
            logger.info(f"Completed evaluation processing for {model_config.name}")

            # 메모리 정리
            logger.info(f"Cleaning up resources for {model_config.name}...")
            del trained_model # 또는 model (evaluate only 시)
            # if 'base_model' in locals(): del base_model # evaluate only 시
            del trained_tokenizer # 또는 tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Resources cleaned up for {model_config.name}")

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {model_config.name}: {str(e)}")
            logger.exception("Overall processing exception details:")
            if 'trained_model' in locals(): del trained_model
            if 'trained_tokenizer' in locals(): del trained_tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

    # Save combined results
    combined_results_path = os.path.join("klue_re_results_v2", "combined_results_likelihood.json")
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"All evaluation results saved to: {combined_results_path}")
    logger.info("KLUE-RE training and evaluation completed")