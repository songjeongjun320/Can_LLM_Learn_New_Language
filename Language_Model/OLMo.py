from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 & 토크나이저 로드 (OLMo는 trust_remote_code 필요)
model = AutoModelForCausalLM.from_pretrained(
    "OLMoE-1B-7B-0924",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # GPU 메모리 절약
)
tokenizer = AutoTokenizer.from_pretrained("OLMoE-1B-7B-0924")

# 추론 파라미터 설정
question = """안녕 혹시 한국말 할줄 아니?,
Answer:"""

inputs = tokenizer(
    question,
    return_tensors="pt",
    max_length=256,
    truncation=True
)

# 생성 설정
outputs = model.generate(
    inputs.input_ids.to(model.device),
    max_new_tokens=150,
    temperature=0.3,  # 창의성 ↓ → 논리적 답변 ↑
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id  # OLMo 토크나이저 이슈 방지
)

# 결과 디코딩
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)