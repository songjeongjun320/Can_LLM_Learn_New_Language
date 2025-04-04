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
from seqeval.metrics import f1_score as entity_f1_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("klue_mrc_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Model configuration class
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
        output_dir="klue_mrc_results/lora-olmo1B-org-klue-mrc",
        is_local=False
    ),
    ModelConfig(
        name="lora-OLMo-1b-Tuned", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo1B", 
        output_dir="klue_mrc_results/lora-olmo1B-v12-klue-mrc",
        is_local=True
    ),
    # ModelConfig(
    #     name="lora-OLMo-7b-org", 
    #     model_path="allenai/OLMo-7B",
    #     output_dir="klue_mrc_results/lora-olmo7B-org-klue-mrc",
    #     is_local=False
    # ),
    # ModelConfig(
    #     name="lora-OLMo-7b-Tuned", 
    #     model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/Fine_Tuned_Results/Full_olmo7B", 
    #     output_dir="klue_mrc_results/lora-olmo7B-v13-klue-mrc",
    #     is_local=True
    # ),
        ModelConfig(
        name="lora-Llama-3.2:3B", 
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b", 
        output_dir="klue_mrc_results/lora-llama3.2-3b-klue-mrc",
        is_local=True
    )
]


# Configuration parameters
DATA_CACHE_DIR = "./klue_mrc_cache"
JSON_TRAIN_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_train.json"
JSON_VAL_DATASET_PATH = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/klue_all_tasks_json/klue_mrc_validation.json"
MAX_LENGTH = 1024  # Increased length for MRC which often has longer contexts

# Model and tokenizer loading function
# 모델 로드 부분을 수정
def load_model_and_tokenizer(model_config):
    """모델 설정에 따라 모델과 토크나이저를 로드합니다."""
    logger.info("===========================================")
    logger.info(f"Load model: {model_config.model_path}")
    logger.info("===========================================")

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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True
    )
    
    return model, tokenizer

def train_model(model_config):
    """Train the model for machine reading comprehension using language modeling approach."""
    logger.info(f"Starting training for {model_config.name}")
    
    # Create output directory
    os.makedirs(model_config.output_dir, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load datasets
    logger.info("Loading train and validation datasets")
    # Load data from JSON
    with open(JSON_TRAIN_DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Convert to Hugging Face Dataset directly
    full_dataset = Dataset.from_dict({
        "title": [item.get("title", "") for item in raw_data],
        "context": [item.get("context", "") for item in raw_data],
        "question": [item.get("question", "") for item in raw_data],
        "answers": [item.get("answers", {"text": [""]}) for item in raw_data]
    })
    
    # Split into train and validation sets
    dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]
    
    logger.info(f"Loaded data - train: {len(train_dataset)} examples, validation: {len(val_dataset)} examples")
    
    # Define a preprocessing function to format data for training
    def preprocess_function(examples):
        formatted_examples = []
        
        for i in range(len(examples["question"])):
            title = examples["title"][i]
            context = examples["context"][i]
            question = examples["question"][i]
            answers = examples["answers"][i]
            answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
            
            prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
            completion = f" {answer_text}"
            
            formatted_examples.append({
                "prompt": prompt,
                "completion": completion
            })
        
        return formatted_examples
    
    # Process datasets
    train_data = train_dataset.map(
        lambda x: {"text": [f"{item['prompt']}{item['completion']}" for item in preprocess_function(x)]},
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_data = val_dataset.map(
        lambda x: {"text": [f"{item['prompt']}{item['completion']}" for item in preprocess_function(x)]},
        batched=True, 
        remove_columns=val_dataset.column_names
    )

    # Rest of your code for training...
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    # LoRA 설정 추가
    peft_params = LoraConfig(
        lora_alpha=16,  # LoRA 스케일링 팩터
        lora_dropout=0.1,  # LoRA 드롭아웃 비율
        r=64,  # LoRA 랭크
        bias="none",  
        task_type="CAUSAL_LM",
        target_modules=["att_proj", "attn_out"]
    )    
    if (model_config.name == "lora-Llama-3.2:3B"):
        peft_params = LoraConfig(
            lora_alpha=16,  # LoRA 스케일링 팩터
            lora_dropout=0.1,  # LoRA 드롭아웃 비율
            r=64,  # LoRA 랭크
            bias="none",  
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj"]
        )

    # 모델 및 토크나이저 로드 시 LoRA 설정 적용
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # 배치 크기 증가
        per_device_eval_batch_size=4,  # 배치 크기 증가
        gradient_accumulation_steps=4,  # 축적 단계 감소
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=200,
        logging_dir=os.path.join(model_config.output_dir, "logs"),
        logging_steps=100,
        fp16=True,  # FP16으로 전환
        bf16=False,  # BF16 비활성화
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # Warmup 비율 감소
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=False,  # 체크포인팅 비활성화
        optim="adamw_torch",  # 필요 시 "adamw_8bit"로 변경
    )

    # SFTTrainer 초기화 시 tokenizer와 packing 제거
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
        eval_dataset=val_data,
        peft_config=peft_params,
    )

    # 학습 실행
    logger.info("Starting training...")
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)

    
    # 최종 모델 저장 (PEFT 모델로)
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    
    # PEFT 모델과 토크나이저 저장
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Fine-tuning completed!")
    return model, tokenizer

# MRC evaluation metrics
def compute_exact_match(prediction, ground_truth):
    """
    Calculate exact match score between prediction and ground truth.
    """
    prediction = prediction.strip().lower()
    ground_truths = [gt.strip().lower() for gt in ground_truth]
    return max(int(prediction == gt) for gt in ground_truths)

def compute_f1_score(prediction, ground_truth):
    """
    Calculate token-level F1 score between prediction and ground truth.
    """
    def get_tokens(s):
        return s.strip().lower().split()
    
    prediction_tokens = get_tokens(prediction)
    f1_scores = []
    
    for gt in ground_truth:
        gt_tokens = get_tokens(gt)
        
        common_tokens = set(prediction_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue
            
        precision = len(common_tokens) / len(prediction_tokens) if prediction_tokens else 0
        recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return max(f1_scores)

def evaluate_model(model, tokenizer, model_config):
    """Evaluate the model on KLUE-MRC metrics."""
    logger.info(f"Evaluating model: {model_config.name}")
    
    # Load MRC dataset
    with open(JSON_VAL_DATASET_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Limit number of samples for faster evaluation
    val_subset = val_data
    
    model.eval()
    
    # Results tracking
    exact_match_scores = []
    f1_scores = []
    
    # Generate detailed logs
    log_file_path = os.path.join(model_config.output_dir, "evaluation_log.json")
    logs = []
    
    for item in tqdm(val_subset, desc="Evaluating"):
        title = item.get("title", "")
        context = item.get("context", "")
        question = item.get("question", "")
        
        # Skip items with missing fields
        if not context or not question:
            continue
        
        ground_truth_answers = item.get("answers", {}).get("text", [""])
        
        # Create prompt for model
        prompt = f"Read the following passage and answer the question.\n\nTitle: {title}\n\nPassage: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=64,  # Limiting output tokens for answers
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        answer = generated_text[len(prompt):].strip()
        
        # Calculate metrics
        em_score = compute_exact_match(answer, ground_truth_answers)
        f1 = compute_f1_score(answer, ground_truth_answers)
        
        exact_match_scores.append(em_score)
        f1_scores.append(f1)
        
        # Log details
        log_entry = {
            "title": title,
            "context": context[:200] + "..." if len(context) > 200 else context,  # Truncate for log
            "question": question,
            "ground_truth": ground_truth_answers,
            "prediction": answer,
            "em_score": em_score,
            "f1_score": f1
        }
        logs.append(log_entry)
    
    # Calculate final metrics
    avg_em = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Log results
    logger.info(f"Evaluation results for {model_config.name}:")
    logger.info(f"Exact Match: {avg_em:.4f}")
    logger.info(f"F1 Score: {avg_f1:.4f}")
    logger.info(f"Total examples evaluated: {len(exact_match_scores)}")
    
    # Save logs and results
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
        
    results = {
        "model": model_config.name,
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "total_examples": len(exact_match_scores)
    }
    
    results_file_path = os.path.join(model_config.output_dir, "eval_results.json")
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Evaluation logs saved to: {log_file_path}")
    logger.info(f"Evaluation results saved to: {results_file_path}")
    
    return results

# Main execution function
if __name__ == "__main__":
    logger.info("Starting KLUE-MRC evaluation only") # 로그 메시지 변경

    # Process each model configuration
    all_results = {}

    for model_config in MODEL_CONFIGS:
        logger.info(f"Processing model for evaluation: {model_config.name}")

        try:
            # Create output directories (로그 및 결과 저장을 위해 필요)
            os.makedirs(model_config.output_dir, exist_ok=True)

            # === Train (주석 처리 또는 제거) ===
            model, tokenizer = train_model(model_config) # 학습 건너뛰기


            # === Evaluate Only ===================================
            # === Evaluate Only ===================================
            # logger.info("Loading base model and tokenizer for evaluation...")
            # base_model, tokenizer = load_model_and_tokenizer(model_config)
            # if tokenizer is None:
            #      logger.error(f"Failed to load tokenizer for {model_config.name}. Skipping evaluation.")
            #      # 필요한 경우 base_model 정리
            #      del base_model
            #      if torch.cuda.is_available(): torch.cuda.empty_cache()
            #      continue

            # peft_model_path = os.path.join(model_config.output_dir, "final")
            # logger.info(f"Attempting to load PEFT adapter from: {peft_model_path}")

            # if not os.path.exists(peft_model_path) or not os.path.isdir(peft_model_path):
            #      logger.error(f"PEFT adapter directory not found or is not a directory at {peft_model_path}. Skipping evaluation for {model_config.name}.")
            #      # 리소스 정리
            #      del base_model
            #      del tokenizer
            #      if torch.cuda.is_available(): torch.cuda.empty_cache()
            #      continue # 다음 model_config 로 이동
            # try:
            #     # PeftModel.from_pretrained를 사용하여 어댑터 로드
            #     model = PeftModel.from_pretrained(
            #         base_model,
            #         peft_model_path,
            #         torch_dtype=torch.bfloat16, # 기본 모델 로드 시 사용한 타입과 일치시키는 것이 좋음
            #         device_map="auto"           # 자동 디바이스 매핑 사용
            #     )

            #     logger.info("PEFT model loaded successfully onto base model for evaluation.")

            # except Exception as load_error:
            #     logger.error(f"Failed to load PEFT model from {peft_model_path}: {load_error}")
            #     logger.exception("PEFT loading exception details:")
            #     # 리소스 정리
            #     del base_model
            #     del tokenizer
            #     if torch.cuda.is_available(): torch.cuda.empty_cache()
            #     continue # 다음 model_config 로 이동

            # === Evaluate Only ===================================
            # === Evaluate Only ===================================

            # 로드된 PEFT 모델로 평가 수행
            if model is not None and tokenizer is not None:
                logger.info(f"Starting evaluation for {model_config.name}...")
                results = evaluate_model(model, tokenizer, model_config)
                all_results[model_config.name] = results
            else:
                 logger.warning(f"Skipping evaluation for {model_config.name} due to model or tokenizer loading issues.")

            logger.info(f"Completed evaluation processing for {model_config.name}")

        except Exception as e:
            # 루프 내 다른 예외 처리 (예: evaluate_model 내부 오류)
            logger.error(f"An unexpected error occurred while processing {model_config.name}: {str(e)}")
            logger.exception("Overall processing exception details:")
            # 현재 반복에서 생성된 객체가 있다면 정리 시도
            # (주의: 에러 발생 위치에 따라 'model', 'base_model' 등이 정의되지 않았을 수 있음)
            if 'model' in locals() and model is not None: del model
            if 'base_model' in locals() and base_model is not None: del base_model
            if 'tokenizer' in locals() and tokenizer is not None: del tokenizer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # 다음 model_config 로 이동
        finally:
            # 각 모델 처리 후 메모리 정리 (성공/실패 여부와 관계없이 시도)
            logger.info(f"Cleaning up resources for {model_config.name}...")
            if 'model' in locals() and model is not None:
                 del model          # PEFT 모델 (또는 병합된 모델) 삭제
                 model = None       # 참조 제거
            if 'base_model' in locals() and base_model is not None:
                 del base_model     # 기본 모델 삭제
                 base_model = None  # 참조 제거
            if 'tokenizer' in locals() and tokenizer is not None:
                 del tokenizer      # 토크나이저 삭제
                 tokenizer = None   # 참조 제거
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Resources cleaned up for {model_config.name}")
    
    # Save combined results
    combined_results_path = "klue_mrc_results/combined_results.json"
    os.makedirs(os.path.dirname(combined_results_path), exist_ok=True)
    
    with open(combined_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    logger.info(f"All results saved to: {combined_results_path}")
    logger.info("KLUE-MRC training and evaluation completed")