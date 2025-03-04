from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from PIL import ImageGrab
import os
import time
import json
#######################
# 폴더 가져오기
#######################
import extract_timestamp


#########################################################
# 필요함수
#########################################################
# 시간 형식 'HH:MM:SS'를 초로 변환하는 함수
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# 초를 'HH:MM:SS' 형식으로 변환하는 함수
def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"

#########################################################
# 로그인 자동화
#########################################################

# .env 파일 로드
load_dotenv()

# 환경 변수에서 ID와 PW 읽어오기
netflix_id = os.getenv("NETFLIX_ID")
netflix_pw = os.getenv("NETFLIX_PW")
netflix_profile_name = os.getenv("NETFLIX_PROFILE_NAME")
netflix_profile_pw0 = os.getenv("NETFLIX_PROFILE_PW0")
netflix_profile_pw1 = os.getenv("NETFLIX_PROFILE_PW1")
netflix_profile_pw2 = os.getenv("NETFLIX_PROFILE_PW2")
netflix_profile_pw3 = os.getenv("NETFLIX_PROFILE_PW3")

# 웹 드라이버 실행
driver = webdriver.Chrome()

# 넷플릭스 로그인 페이지로 이동
url = 'https://www.netflix.com/login'
print("--LOG : 넷플릭스 로그인 페이지로 이동 중...")
driver.get(url)

time.sleep(2)

# ID와 PW 입력 및 로그인 시도
try:
    # 이메일 입력란 찾기
    print("--LOG : 이메일 입력란 찾기...")
    email_input = driver.find_element(By.NAME, "userLoginId")
    email_input.send_keys(netflix_id)
    
    # 비밀번호 입력란 찾기
    print("--LOG : 비밀번호 입력란 찾기...")
    password_input = driver.find_element(By.NAME, "password")
    password_input.send_keys(netflix_pw)
    
    # 로그인 버튼 클릭
    print("--LOG : 로그인 버튼 클릭 시도...")
    password_input.send_keys(Keys.RETURN)

    # 프로필 선택 화면이 로드될 때까지 대기
    print("--LOG : 프로필 선택 화면 대기...")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "profile-name"))
    )

    # 프로필 선택
    print("--LOG : 프로필 선택 시도...")
    profiles = driver.find_elements(By.CLASS_NAME, "profile-name")
    for profile in profiles:
        if profile.text == netflix_profile_name:
            profile.click()
            break

    # 4자리 비밀번호 입력
    print("--LOG : 4자리 비밀번호 입력 시도...")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "pin-input-container"))
    )

    # 각 PIN 입력 필드에 값을 입력
    pin_inputs = driver.find_elements(By.CLASS_NAME, "pin-number-input")
    pin_inputs[0].send_keys(netflix_profile_pw0)
    pin_inputs[1].send_keys(netflix_profile_pw1)
    pin_inputs[2].send_keys(netflix_profile_pw2)
    pin_inputs[3].send_keys(netflix_profile_pw3)

    # PIN 입력 후 Enter 키 전송 (필요한 경우)
    pin_inputs[3].send_keys(Keys.RETURN)

except Exception as e:
    print(f"--LOG : 에러 발생 - {e}")

print(f"--LOG : 로그인 자동화 완료")
time.sleep(5)

#########################################################
# 로그인 자동화 완료, 드라마 검색 이후 페이지 이동
#########################################################

# drama_title.txt에서 첫 번째 줄 읽기
with open('drama_title.txt', 'r', encoding='utf-8') as file:
    drama_title = file.readline().strip()

# 경로 설정: screenshots
folder_path = os.path.join("screenshots", drama_title)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # 폴더가 없으면 생성

# 드라마 제목을 입력한 후, 검색창에 제목을 입력
try:
    # 검색 아이콘 클릭
    search_button = driver.find_element(By.CLASS_NAME, "searchTab")
    search_button.click()
    print(f"'{drama_title}' 검색 준비 완료!")

    # 검색창이 로드될 때까지 대기
    WebDriverWait(driver, 2).until(
        EC.presence_of_element_located((By.CLASS_NAME, "searchInput"))
    )

    # 검색창에 drama_title을 한 글자씩 0.5초마다 입력
    search_box = driver.find_element(By.CLASS_NAME, "focus-visible")
    for char in drama_title:
        search_box.send_keys(char)
        time.sleep(0.5)  # 0.5초 대기
    print(f"'{drama_title}' 입력 완료!")

    # 검색 결과 대기 (적절한 시간 설정)
    time.sleep(5)

    # 첫 번째 드라마 클릭 (제목을 입력한 후 클릭)
    WebDriverWait(driver, 3).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".title-card"))
    )

    first_drama = driver.find_element(By.CSS_SELECTOR, ".title-card")
    first_drama.click()
    print(f"첫 번째 드라마 '{drama_title}' 클릭 완료!")

except Exception as e:
    print(f"오류 발생: {e}")

print(f"--LOG : 드라마 검색 성공", drama_title)
time.sleep(3)

#########################################################
# 재생 클릭릭
#########################################################
print("--LOG : 재생 클릭 시도")
try:
    # 1. aria-label이 "재생"인 a 태그 찾기
    play_link = driver.find_element(By.XPATH, "//a[@aria-label='재생']")

    # 2. a 태그 밑에 있는 버튼 찾기
    button = play_link.find_element(By.XPATH, ".//button")

    # 3. 버튼 클릭
    button.click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "ltr-1qtcbde")))

except Exception as e:
    print(f"에러 발생: {e}")

######################
# timestamp 리스트 extract_timestamp.py 로부터 받아오기
# 나중에 자동화 해야함.
######################
timestamp_list = extract_timestamp.extract_timestamp()
# print("--LOG : timestamp = ", timestamp_list)

video_length = ""
checked = False

while True:
    try:
        # 남은 시간 정보가 있는 <span> 태그 찾기
        time_remaining_element = driver.find_element(By.CLASS_NAME, "ltr-1qtcbde")
        time_remaining = time_remaining_element.text.strip()  # 남은 시간 텍스트 가져오기
        print("--LOG 남은시간: ", time_remaining)

        # 체크할 데이터 없으면 정지
        if len(timestamp_list) == 0:
            break

        # 마우스를 아래쪽으로 계속 이동시켜서 비디오 제어 바가 사라지지 않게 유지
        action = ActionChains(driver)
        action.move_to_element(time_remaining_element).move_by_offset(0, 50).perform()  # y값을 증가시켜 마우스를 아래로 이동시킴
       
        if not checked:
            video_length = time_remaining  # 전체 길이 저장
            checked = True
            print("--LOG 비디오 길이: ", video_length)

        # video_length와 time_remaining을 초 단위로 변환
        video_length_seconds = time_to_seconds(video_length)
        time_remaining_seconds = time_to_seconds(time_remaining)
        time_need_to_be_checked = time_to_seconds(timestamp_list[0])
        action.move_to_element(time_remaining_element).move_by_offset(0, 50).perform()  # y값을 증가시켜 마우스를 아래로 이동시킴

        # 현재 시간 계산 (전체 시간 - 남은 시간)
        current_time_seconds = video_length_seconds - time_remaining_seconds

        print("--LOG current_time : ", current_time_seconds)
        print("--LOG timestamp[0] : ", time_need_to_be_checked)

        # timestamp_list[0] 안에 있는 시간 - current_time이 > 11 일경우, 10초 건너뛰기 버튼 누르기
        # print("--LOG : 찍혀야 하는 시간 - 현재시간: ", time_need_to_be_checked - current_time_seconds)
        # if abs(current_time_seconds - time_need_to_be_checked) > 11:
        #     # 건너뛰기 버튼 클릭 (버튼을 찾아 클릭)
        #     skip_button = driver.find_element(By.XPATH, "//button[@aria-label='앞으로 가기']")
        #     skip_button.click()
        
        # 계산된 초를 다시 'HH:MM:SS' 형식으로 변환
        current_time = seconds_to_time(current_time_seconds)

        # 남은 시간 출력
        print(f"남은 시간: {current_time}")
        action.move_to_element(time_remaining_element).move_by_offset(0, 50).perform()  # y값을 증가시켜 마우스를 아래로 이동시킴

        # 파일 경로 생성
        file_name = f"{drama_title}_{current_time.replace(':', '_')}.png"
        file_path = os.path.join(folder_path, file_name)
        print("--LOG : 파일경로 생성 : ", file_path)

        # 스크린샷을 해당 경로에 저장
        if current_time_seconds == time_need_to_be_checked:
            screenshot = ImageGrab.grab()
            screenshot.save(file_path)

            print(f"스크린샷이 저장되었습니다: {file_path}")

        action.move_to_element(time_remaining_element).move_by_offset(0, 50).perform()  # y값을 증가시켜 마우스를 아래로 이동시킴

    finally:
        time.sleep(0.3)