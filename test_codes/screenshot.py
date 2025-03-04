from PIL import ImageGrab

# 전체 화면 캡처
screenshot = ImageGrab.grab()

# 캡처한 이미지를 파일로 저장
screenshot.save("screenshot.png")