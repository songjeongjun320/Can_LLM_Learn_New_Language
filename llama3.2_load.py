import os
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# 다운로드 경로 설정
download_path = "/scratch/jsong132/Can_LLM_Learn_New_Language/llama3.2_3b"

# 디렉토리가 존재하지 않으면 생성
os.makedirs(download_path, exist_ok=True)

# 모델 파일 다운로드
model_file_1 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00001-of-00002.safetensors",
    local_dir=download_path
)
model_file_2 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00002-of-00002.safetensors",
    local_dir=download_path
)
index_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model.safetensors.index.json",
    local_dir=download_path
)

# 모델을 지정된 경로에 저장
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", 
    cache_dir=download_path
)