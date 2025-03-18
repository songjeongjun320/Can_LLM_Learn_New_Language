#!/bin/bash
#SBATCH --job-name=drama_image_processing
#SBATCH --output=drama_process_%j.out
#SBATCH --error=drama_process_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
# Add any other SLURM parameters you need for your cluster

# Parse command line arguments
drama_folder=""
ollama_host=""
base_path=""
version=""

# Help text
print_usage() {
    echo "Usage: sbatch $0 --drama DRAMA_FOLDER --host OLLAMA_HOST --path BASE_PATH --version VERSION"
    echo ""
    echo "Arguments:"
    echo "  --drama    Name of the drama folder to process"
    echo "  --host     Ollama host address (e.g., http://sg022:11434)"
    echo "  --path     Base path where data folders are located"
    echo "  --version  Version string for output directories"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --drama)
            drama_folder="$2"
            shift
            shift
            ;;
        --host)
            ollama_host="$2"
            shift
            shift
            ;;
        --path)
            base_path="$2"
            shift
            shift
            ;;
        --version)
            version="$2"
            shift
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if required parameters are provided
if [[ -z "$drama_folder" || -z "$ollama_host" || -z "$base_path" || -z "$version" ]]; then
    echo "Error: Missing required parameters"
    print_usage
    exit 1
fi

# Create a temporary Python script with the parameters inserted
temp_script=$(mktemp --suffix=.py)

cat > $temp_script << EOF
import os
import json
import base64
import ollama
import time
import traceback
from pathlib import Path
from datetime import datetime

# Set up Ollama client with the provided host
ollama_host = "$ollama_host"
client = ollama.Client(host=ollama_host)

# Set up the prompt for image analysis
prompt = """Analyze the given Circumstance and describe the specific actions and interactions of the people in this circumstance.
Focus on what they are doing, their gestures, expressions, and interactions, and provide general details about the environment or objects.
Guess what kind of conversation might typically occur in this situation. Ignore any text visible in the Circumstance.

Example 1:
[Image shows: A group of people are gathered around a table, looking at documents and gesturing.]
Description: This circumstance shows a business meeting in progress. There are four people sitting around a wooden conference table, reviewing printed documents. The woman in the red blazer appears to be presenting, gesturing with her hands as she explains something on the paper. The others are listening attentively, with one man taking notes on his laptop.

**Possible conversation:
"Let's review the project updates"
"We need to discuss the new marketing strategy" 
"What's the status of the client proposal?"
"I think we should prioritize the eastern market expansion"
"Has everyone received the quarterly forecast report?"
"Could you explain the decline in these numbers?"
"What feedback did we get from the focus group?"

Example 2:
[Circumstance shows: A person is sitting on a bench in a park, reading a book and looking relaxed.]
Description: This circumstance shows someone enjoying leisure time outdoors. A young woman with curly hair is sitting on a wooden bench in a public park. She's reading a paperback book while smiling, suggesting she's enjoying the content. The park around her has green trees and a walking path visible in the background.

**Possible conversation:
"This book is really interesting"
"The weather is perfect today"
"I'm glad I took some time to relax"
"I've been meaning to finish this novel for weeks"
"This park is my favorite spot in the city"
"The author's perspective on climate change is fascinating"
"I should come here more often to disconnect from work"

Example 3:
[Circumstance shows: Two friends having coffee at an outdoor café, one showing something on their phone to the other.]
Description: This circumstance shows a casual social meetup between friends. Two young adults are sitting at a small round table outside a café. The person on the left is holding up their smartphone and showing something on the screen to their friend, who is leaning in with interest. There are two coffee cups on the table along with a small plate of pastries. The café has a striped awning overhead and there are other patrons visible in the background.

**Possible conversation:
"Look at these photos from my weekend trip"
"Have you seen this funny video that's going viral?"
"What do you think about this apartment I'm considering?"
"I can't believe what our old classmate posted on social media"
"This coffee shop has the best pastries in town"
"Should I buy these shoes? They're on sale"
"I'm thinking about applying for this job opportunity"

Example 4:
[Circumstance shows: A classroom with a teacher standing at the front and students engaged in a group activity.]
Description: This circumstance shows an educational environment in progress. A middle-aged teacher with glasses is standing at the whiteboard, pointing to diagrams while explaining a concept. The students are seated in small groups of three to four, with notebooks open and colorful project materials spread across their desks. Some students are raising their hands to ask questions while others are collaborating on what appears to be a science project with small models on their tables. The classroom has educational posters on the walls and a digital projector displaying related content.

**Possible conversation:
"Can someone explain how photosynthesis works in your own words?"
"For your group project, focus on demonstrating the carbon cycle"
"Does anyone have questions before we move on to the next section?"
"Remember to include your research sources in your final presentation"
"Let's brainstorm solutions to this environmental challenge"
"Make sure everyone in your group has a chance to contribute"
"I like how your team approached this problem differently"
"We'll need to finish this activity before the bell rings"
"Can you explain your reasoning behind this conclusion?"
"Excellent question! Let's explore that concept further"

[Now, describe the Circumstance I've shared and guess possible conversation:]
"""

def process(base_path="/scratch/jsong132/Can_LLM_Learn_New_Language", drama_folder_name, version):
    # 경로 설정
    base_path = base_path
    drama_folder_name = drama_folder_name
    version = version
    
    image_dir = Path(f'{base_path}/Data_Images/{drama_folder_name}')
    output_file = Path(f'{base_path}/Refined_Datas/{version}/Data_llama_vision/{drama_folder_name}.json')
    
    #######################
    # model choose
    #######################
    used_model = "llama3.2-vision"
    
    # 출력 디렉토리 생성
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 로깅 설정
    def log(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    # 초기화
    results = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    total_images = len([f for f in image_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
    processed = 0
    
    log(f"Starting image processing for {total_images} images")
    
    # 이미지 처리
    for image_path in image_dir.iterdir():
        if not (image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS):
            continue
    
        processed += 1
        log(f"Processing image ({processed}/{total_images}): {image_path.name}")
        start_time = time.time()
        
        try:
            # 이미지 인코딩
            encode_start = time.time()
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            encode_time = time.time() - encode_start
            log(f"Image encoded in {encode_time:.2f}s")
    
            # API 요청
            api_start = time.time()
            used_model = "llama3.2-vision"
            response = client.chat(
            ###################### Choose Model ###################
                model="llama3.2-vision",
                # "llama3.2-vision:90b"
                # llama3.2-vision"
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]
                }]
            )
            api_time = time.time() - api_start
            log(f"API response received in {api_time:.2f}s")
            
            # print("API Response:")
            # print(response['message']['content'])
    
            # 결과 저장
            results.append({
                'image': str(image_path),
                'response': response['message']['content'],
                'processing_time': {
                    'encoding': encode_time,
                    'api_call': api_time,
                    'total': time.time() - start_time
                },
                'status': 'success'
            })
        
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            error_trace = traceback.format_exc()
            log(error_msg, "ERROR")
            log(f"Error details:\n{error_trace}", "DEBUG")
            
            results.append({
                'image': str(image_path),
                'error': error_msg,
                'error_trace': error_trace,
                'status': 'failed'
            })
            

    
    # 결과 저장
    save_start = time.time()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    save_time = time.time() - save_start
    
    log("=========================================")
    log(f"Results saved to {output_file} in {save_time:.2f}s")
    log(f"Processing completed. Success: {len([x for x in results if x['status']=='success'])}, Failed: {len([x for x in results if x['status']=='failed'])}")
    log("=========================================")
    
    #######################
    # Organize by frame number
    #######################
    
    # 파일 경로 설정
    input_file = Path(f'{base_path}/Refined_Datas/{version}/Data_llama_vision/{drama_folder_name}.json')
    output_file = Path(f'{base_path}/Refined_Datas/{version}/Data_llama_vision/{drama_folder_name}_organized.json')
    
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 아이템 처리 함수
    def process_item(item):
        try:
            filename = Path(item['image']).name
            
            # 파일명 구조: frame_0001.png → ['frame', '0001.png']
            parts = filename.split('_')
            
            # 숫자 부분 추출 (frame_0001.png → 0001)
            frame_number = int(parts[1].split('.')[0])  # frame_0001.png → 0001
            
            return {
                'image': filename,
                'frame_number': frame_number,
                'response': item['response'],
                'status': item['status']
            }
        except Exception as e:
            print(f"파일명 형식 오류: {filename} → {str(e)}")
            return None
    
    # 데이터 처리 (오류 항목 필터링)
    processed_data = [item for item in (process_item(i) for i in data) if item is not None]
    
    # 숫자 순으로 정렬 (frame_number 기준)
    sorted_data = sorted(processed_data, key=lambda x: x['frame_number'])
    
    # 최종 출력 형식
    final_data = [
        {
            'used_model': 'llama3.2-vision',  # 모델 이름을 하드코딩 (필요 시 수정)
            'image': sorted_data[0]['image'],
            'response': sorted_data[0]['response'],
        }
    ] + [
        {
            'image': item['image'],
            'response': item['response'],
        }
        for item in sorted_data[1:]
    ]
    
    # JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    # 파일 삭제
    if input_file.exists():  # 파일이 존재하는지 확인
        input_file.unlink()  # 파일 삭제
        print(f"파일 제거: {input_file}")
    else:
        print(f"{input_file} 파일이 존재하지 않습니다.")
        
    print(f"정렬 완료! 결과 파일: {output_file}")
    
    ################################################
    # Merging llama-vision result + subtitle + timestamp
    ################################################
    output_file = Path(f'{base_path}/Refined_Datas/{version}/Data_llama_vision/{drama_folder_name}_organized.json')
    
    # llama_vision_data.json 파일 로드
    with open(output_file, "r", encoding="utf-8") as f:
        llama_vision_data = json.load(f)
    
    subtitle = Path(f'{base_path}/Data_Subtitles/{drama_folder_name}_organized.ko.json')
    # subtitle.json 파일 로드
    with open(subtitle, "r", encoding="utf-8") as f:
        subtitle_data = json.load(f)
    
    # final_output.json으로 저장할 데이터 리스트 초기화
    dataset = []
    
    # 두 파일의 데이터를 매칭하여 dataset 생성
    for result_item, subtitle_item in zip(llama_vision_data, subtitle_data):
        input_text = subtitle_item.get("context", "")  # subtitle_data.json의 "context"를 input으로
        timestamp = subtitle_item.get("timestamp", "")  # subtitle_data.json의 "timestamp"를 timestamp
        output_text = result_item.get("response", "")  # llama_vision_data.json의 "response"를 output으로
        
        # input, output, timestamp가 모두 비어있지 않은 경우만 추가
        if input_text and output_text:
            dataset.append({"timestamp": timestamp, "input": input_text, "output": output_text})
    
    # 디렉토리가 없으면 생성
    final_output = Path(f'{base_path}/Refined_Datas/{version}/Data_Final/')
    final_output.mkdir(parents=True, exist_ok=True)
    print(f"디렉토리가 생성되었습니다: {final_output}")
    
    final_output = Path(f'{base_path}/Refined_Datas/{version}/Data_Final/{drama_folder_name}_final.json')
    # dataset.json 파일로 저장
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"/Refined_Datas/{version}/Data_Final/{drama_folder_name}_final.json 파일이 생성되었습니다.")
    
    ###############################################
    # Reverse input <-> output
    # Professor suggestion
    ################################################
    
    # final_output 파일 경로 설정
    final_output = Path(f'{base_path}/Refined_Datas/{version}/Data_Final/{drama_folder_name}_final.json')
    
    # final_output에서 데이터 읽기
    with open(final_output, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # input과 output 값을 교환하는 작업
    reversed_dataset = []
    for item in dataset:
        # input과 output을 서로 바꾸기
        reversed_item = {
            'timestamp': item['timestamp'],
            'input': item['output'],  # output을 input으로
            'output': item['input'],  # input을 output으로
        }
        reversed_dataset.append(reversed_item)
    
    # 기존 dataset과 reversed_dataset을 합치기
    combined_dataset = dataset + reversed_dataset
    
    # 디렉토리가 없으면 생성
    reversed_final_output = Path(f'{base_path}/Refined_Datas/{version}/Data_Final_Reversed/')
    reversed_final_output.mkdir(parents=True, exist_ok=True)
    print(f"디렉토리가 생성되었습니다: {reversed_final_output}")
    
    # reversed_final_output 파일 경로 설정
    reversed_final_output = Path(f'{base_path}/Refined_Datas/{version}/Data_Final_Reversed/{drama_folder_name}_reversed_final.json')
    
    # combined dataset을 새로운 JSON 파일로 저장
    with open(reversed_final_output, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=4)
    
    print(f"/Refined_Datas/{version}/Data_Final_Reversed/{drama_folder_name}_reversed_final.json 파일이 생성되었습니다.")

if __name__ == "__main__":
    process("$base_path", "$drama_folder_name", "$version")
EOF

# Log the parameters
echo "Starting job with parameters:"
echo "  Drama folder: $drama_folder"
echo "  Ollama host: $ollama_host"
echo "  Base path: $base_path"
echo "  Version: $version"
echo "  Script location: $temp_script"

# Ensure Python environment with required packages
# Uncomment and modify if you use a specific conda env or virtualenv
# source /path/to/your/virtualenv/bin/activate
# or
# module load python/3.9
# module load conda
# conda activate your_env_name

# Run the Python script
python $temp_script

# Cleanup temp file
rm $temp_script

echo "Job completed"