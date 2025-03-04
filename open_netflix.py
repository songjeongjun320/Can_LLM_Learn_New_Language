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