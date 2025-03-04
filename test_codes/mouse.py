from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from dotenv import load_dotenv
import os
import time
from selenium.webdriver.support.ui import WebDriverWait

# 드라이버 설정
driver = webdriver.Chrome()

# 예시로 이동할 위치 (x, y 좌표)
x_pos = 300
y_pos = 500

# 마우스를 특정 위치로 계속 이동
action = ActionChains(driver)

while True:  # 무한 반복으로 마우스 이동
    action.move_by_offset(x_pos, y_pos).perform()  # 현재 위치에서 지정된 좌표만큼 이동
    time.sleep(1)  # 이동 후 잠시 대기 (원하는 시간만큼 설정 가능)