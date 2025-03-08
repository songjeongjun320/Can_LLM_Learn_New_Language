from PIL import ImageGrab
import os
import time
import extract_timestamp

def time_to_seconds(time_str):
    """HH:MM:SS 형식을 초로 변환하는 함수"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    """초를 HH:MM:SS 형식으로 변환하는 함수"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

# 우선 드라마 타이틀은 메뉴얼로
drama_title = "하우스 오브 카드1"

# .json 자막 경로 넣기
json_path = "C:\\Users\\songj\\OneDrive\\Desktop\\Can-LLM-learn-new-Language\\Samples_Result\\하우스.오브.카드 (1)\\하우스.오브.카드.S01E01.WEBRip.Netflix.ko.json"
video_full_length = "00:56:38" # HH:MM:SS 형식으로 넣어서, 시간 교체

# video_full_length를 초로 변환
video_full_length_seconds = time_to_seconds(video_full_length)

# 타임스탬프 리스트 가져오기
timestamp_list = extract_timestamp.extract_timestamp(json_path)
inverse_timestamp_list = []

# timestamp_list의 각 타임스탬프 처리
for i in range(len(timestamp_list)):
    # timestamp_list[i]를 초로 변환
    timestamp_seconds = time_to_seconds(timestamp_list[i])

    # video_full_length에서 timestamp_list[i]를 빼서 시간 차이 계산
    time_difference_seconds = video_full_length_seconds - timestamp_seconds

    # 초를 HH:MM:SS 형식으로 변환
    inverse_timestamp_list.append(seconds_to_time(time_difference_seconds))

# 결과 출력
# print("--LOG : 타임스탬프 역변환: ", inverse_timestamp_list)

# 경로 설정: screenshots
folder_path = os.path.join("screenshots", drama_title)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # 폴더가 없으면 생성

while len(inverse_timestamp_list) != 0:
    # 첫 번째 타임스탬프를 파일 이름으로 설정
    file_name = f"{timestamp_list[0].replace(':', '_')}.png"
    file_path = os.path.join(folder_path, file_name)

    # 사용자 입력 기다리기 (enter 키를 입력 받으면 스크린샷 찍기)
    input(f"타임스탬프 {inverse_timestamp_list[0]}에서 스크린샷을 찍으려면 엔터 키를 누르세요...")

    # 스크린샷 캡처
    screenshot = ImageGrab.grab()
    
    # 파일로 저장
    screenshot.save(file_path)
    print(f"스크린샷 저장됨: {file_path}")

    # timestamp_list, inverse 에서 첫 번째 원소 제거
    timestamp_list.pop(0)
    inverse_timestamp_list.pop(0)

    # 반복
    time.sleep(0.5)  # 잠시 대기
