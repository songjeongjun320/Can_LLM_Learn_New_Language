from transformers import pipeline
from PIL import Image
import torch

# 1. CPU 강제 설정 및 모델 초기화
# Use a pipeline as a high-level helper
from transformers import pipeline


# 2. 메모리 최적화 파이프라인 설정
pipe = pipeline(
    "image-to-text",
    model="meta-llama/Llama-3.2-11B-Vision",
    device="cpu",
    torch_dtype=torch.float32,  # CPU는 float32 권장
    device_map="auto",
    load_in_8bit=True  # 8비트 양자화 적용 (메모리 50% 감소)
)

# 3. 이미지 전처리
image_path = r"C:\Users\songj\OneDrive\Desktop\Can-LLM-learn-new-Language\cropped_img\cropped_00_00_21.png"
image = Image.open(image_path).resize((336, 336))  # 해상도 축소

# 4. CPU 최적화 프롬프트 설정
question = """
<image>
USER: Explain what circumstance you can guess from this image.
ASSISTANT:
"""

# 5. 메모리 절약 추론 설정
outputs = pipe(
    image,
    prompt=question,
    generate_kwargs={
        "max_new_tokens": 100,  # 출력 길이 축소
        "temperature": 0.1,
        "repetition_penalty": 1.5  # 반복 문장 방지
    }
)

print("[Answer]\n", outputs[0]['generated_text'])