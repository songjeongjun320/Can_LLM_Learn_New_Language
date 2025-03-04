import os
from PIL import Image

# 이미지가 있는 폴더 경로
image_folder_path = r"C:\Users\songj\OneDrive\Desktop\Can-LLM-learn-new-Language\screenshots\하우스 오브 카드1"

# 저장할 폴더 경로
cropped_folder_path = r"./cropped_img"

# cropped_img 폴더가 없으면 생성
if not os.path.exists(cropped_folder_path):
    os.makedirs(cropped_folder_path)

# 이미지 경로에 있는 모든 파일을 확인하고 처리
for filename in os.listdir(image_folder_path):
    # 이미지 파일만 처리하기 위해 확장자 확인
    if filename.endswith((".png", ".jpg", ".jpeg")):  # 이미지 확장자 필터링
        image_path = os.path.join(image_folder_path, filename)
        
        # 이미지 열기
        image = Image.open(image_path)

        # 이미지 크기 가져오기 (width, height)
        width, height = image.size

        # 잘라낼 범위 정의: 윗부분 10px, 아랫부분 10px 자르기 (crop_box 값)
        crop_box = (150, 210, width - 150, height - 320)

        # 이미지 자르기
        cropped_image = image.crop(crop_box)

        # 잘라낸 이미지 저장
        cropped_image.save(os.path.join(cropped_folder_path, f"cropped_{filename}"))

        print(f"이미지 {filename}을 잘라서 저장했습니다.")
