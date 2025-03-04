from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from dotenv import load_dotenv
import os
import time
from datetime import timedelta
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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


#########################################################
# 로그인 자동화
#########################################################



timestamp_list = ["00:02:17", "00:02:28", "00:02:31", "00:02:35"]

# 비디오가 로드될 때까지 기다리기
try:
    video_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "video"))
    )
except:
    print("비디오 로드 실패")
    driver.quit()

# 비디오가 로드된 후 Enter 키를 눌러서 스크립트 진행
input("비디오가 실행되면 Enter 키를 눌러주세요.")  # Enter를 눌러야 비디오가 실행된 후 진행

# 각 timestamp에 맞춰서 이동하고 스크린샷을 찍는 함수
def capture_screenshot_at_timestamps():
    for timestamp in timestamp_list:
        hour, minute, second = map(float, timestamp.split(":"))
        target_time = timedelta(hours=hour, minutes=minute, seconds=second)

        # 비디오의 currentTime 속성을 통해 시간을 이동
        driver.execute_script(f"arguments[0].currentTime = {target_time.total_seconds()};", video_element)

        # 시간 이동 후 잠시 대기
        time.sleep(2)  # 2초 기다려서 비디오가 시간대로 이동했는지 확인

        # 스크린샷 파일명 설정
        timestamp_str = timestamp.replace(":", "_")  # "00:02:17" => "00_02_17"
        screenshot_filename = os.path.join(screenshot_folder, f"{drama_name}_{episode_number}_{timestamp_str}.png")

        # 전체 화면 캡처
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_filename)

        print(f"스크린샷이 {screenshot_filename}으로 저장되었습니다.")

# 스크린샷 저장 경로 설정
screenshot_folder = "C:/Users/songj/OneDrive/Desktop/Can-LLM-learn-new-Language/screenshots"
os.makedirs(screenshot_folder, exist_ok=True)  # 폴더가 없으면 생성

# 타임스탬프에 맞춰서 스크린샷을 찍음
capture_screenshot_at_timestamps()

# 브라우저 종료
driver.quit()
