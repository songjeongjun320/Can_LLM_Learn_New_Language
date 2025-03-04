import cv2
import os

def capture_frame(video_path, time_seconds, output_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_seconds)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(output_path, frame)
        print(f"Screenshot saved at {output_path}")
    else:
        print("Error: Could not capture frame.")
    
    cap.release()
    return success

# 사용 예시
video_file = "example.mp4"  # 비디오 파일 경로
screenshot_time = 10  # 캡처할 초
output_file = "screenshot.jpg"  # 저장할 파일 경로

capture_frame(video_file, screenshot_time, output_file)
