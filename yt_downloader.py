import os
import json
from yt_dlp import YoutubeDL
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import time
import shutil
import re

def download_video_and_subtitles(url, video_name, output_folder_video='yt_dataset/videos', output_folder_subtitle='yt_dataset/subtitles'):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder_video):
        os.makedirs(output_folder_video)
    if not os.path.exists(output_folder_subtitle):
        os.makedirs(output_folder_subtitle)

    cookies_path = "cookies.txt"
    # yt-dlp 설정
    ydl_opts = {
        'cookies': cookies_path,  # 쿠키 파일 경로
        'format': 'best',  # 최고 품질로 다운로드
        'writesubtitles': True,  # 자막 다운로드
        'writeautomaticsub': True,  # 자동 생성 자막 다운로드
        'subtitleslangs': ['ko'],  # 한국어 자막만 선택
        'subtitlesformat': 'vtt',  # 자막을 VTT 형식으로 저장
        'outtmpl': os.path.join(output_folder_video, f'{video_name}.%(ext)s'),  # 사용자 정의 비디오 파일명
        'quiet': True,  # 로그 출력 방지
    }

    # 영상 다운로드 및 자막 추출
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_filename = ydl.prepare_filename(info_dict)

    # 자막 파일 경로
    subtitle_filename = os.path.splitext(video_filename)[0] + ".ko.vtt"
    time.sleep(2)

    # 자막 파일을 다른 폴더로 이동
    if os.path.exists(subtitle_filename):
        subtitle_target_path = os.path.join(output_folder_subtitle, f'{video_name}.ko.vtt')
        shutil.move(subtitle_filename, subtitle_target_path)

    return video_filename, subtitle_target_path

def clean_vtt(vtt_data):
    # WEBVTT, NOTE 라인과 빈 줄 모두 제거
    cleaned_data = re.sub(r'^(WEBVTT|NOTE.*\n)', '', vtt_data, flags=re.MULTILINE)
    cleaned_data = re.sub(r'^\s*\n', '', cleaned_data, flags=re.MULTILINE)

    # 특정 태그 제거
    cleaned_data = re.sub(r'<c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.korean>', '', cleaned_data)
    cleaned_data = re.sub(r'<c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'</c\.bg_transparent>', '', cleaned_data)
    cleaned_data = re.sub(r'NETFLIX오리지널시리즈', '', cleaned_data)
    # 타임스탬프 뒤의 불필요한 position 정보 제거
    cleaned_data = re.sub(r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}) .*', r'\1', cleaned_data)
    # 대괄호나 소괄호 안의 내용 제거
    cleaned_data = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned_data)

    return cleaned_data

def process_vtt_file(file_path, output_file_path):
    # 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        vtt_data = file.read()

    # VTT 데이터 정리
    cleaned_vtt = clean_vtt(vtt_data)

    # 줄 단위로 분리
    lines = cleaned_vtt.strip().split('\n')
    entries = []
    current_index = 1
    current_timestamp = None
    current_context = []
    total_sentences = 0

    for line in lines:
        if '-->' in line:  # 타임스탬프 줄인 경우
            # 이전 블록 처리: 타임스탬프와 모아진 텍스트가 있다면
            if current_timestamp is not None and current_context:
                context_text = " ".join(current_context).strip()
                # 모든 띄어쓰기와 특수문자 제거 (한글, 영문, 숫자만 남김)
                context_text = re.sub(r'[^\w\s]', '', context_text)  # 특수문자만 제거, 공백 유지
                if context_text:
                    entry = {
                        "frame": current_index,
                        "timestamp": current_timestamp,
                        "context": context_text
                    }
                    entries.append(entry)
                    current_index += 1
            # 새 타임스탬프 갱신 및 텍스트 버퍼 초기화
            current_timestamp = line.strip()
            current_context = []
        else:
            # 만약 줄이 단순 숫자(인덱스 번호)라면 무시
            if re.match(r'^\d+$', line.strip()):
                continue
            # 타임스탬프가 없는 상태에서는 해당 텍스트 무시
            if current_timestamp is not None:
                current_context.append(line.strip())
                total_sentences += 1
            else:
                continue

    # 마지막 블록 처리
    if current_timestamp is not None and current_context:
        context_text = " ".join(current_context).strip()
        context_text = re.sub(r'[\s\W]+', '', context_text)
        if context_text:
            entry = {
                "frame": current_index,
                "timestamp": current_timestamp,
                "context": context_text
            }
            entries.append(entry)

    # JSON 파일로 저장 (한글 깨짐 방지를 위해 ensure_ascii=False)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(entries, output_file, ensure_ascii=False, indent=4)

    print(f"Cleaned VTT has been saved to {output_file_path}")
    print(f"Total sentences: {total_sentences}")

import re

def convert_timestamp_to_seconds(timestamp):
    """
    '00:00:08.750' 또는 '00:00:08' 형태의 타임스탬프를 초로 변환.
    """
    # 타임스탬프가 밀리초를 포함할 수도 있고 아닐 수도 있음
    parts = re.split('[:.]', timestamp)
    
    if len(parts) == 4:  # HH:MM:SS.mmm 형식
        hours, minutes, seconds, milliseconds = map(float, parts)
    elif len(parts) == 3:  # MM:SS 형식
        hours, minutes, seconds = map(float, parts)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def convert_seconds_to_timestamp(seconds):
    """
    초 값을 받아서 '00:00:08' 형태로 변환.
    밀리초는 제거됩니다.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{int(seconds):02}"  # 밀리초는 제거

def extract_subtitles(subtitle_filename):
    # VTT 파일을 JSON 형식으로 변환
    json_filename = subtitle_filename.replace('.vtt', '.json')
    process_vtt_file(subtitle_filename, json_filename)
    
    # 변환된 JSON 파일을 로드
    with open(json_filename, 'r', encoding='utf-8') as f:
        subtitle_data = json.load(f)
    
    # "context" 안에 "0000"이 포함된 항목 제거
    subtitle_data_organized = [
        entry for entry in subtitle_data 
        if "0000" not in entry.get("context", "") and "000" not in entry.get("context", "") and not re.search(r'[a-zA-Z]', entry.get("context", ""))
    ]
    # 재정렬된 index 값 다시 부여
    for idx, entry in enumerate(subtitle_data_organized):
        entry['frame'] = idx + 1
    
    count = 0
    # 타임스탬프 계산 및 업데이트
    for i in range(len(subtitle_data_organized)):
        prev_end_timestamp = subtitle_data_organized[i]['timestamp'].split(' --> ')[0]
        next_start_timestamp = subtitle_data_organized[i]['timestamp'].split(' --> ')[1]

        # 타임스탬프를 초로 변환
        prev_end_seconds = convert_timestamp_to_seconds(prev_end_timestamp)
        next_start_seconds = convert_timestamp_to_seconds(next_start_timestamp)

        # 중간값 계산
        mid_seconds = (prev_end_seconds + next_start_seconds) / 2

        # 중간값을 새로운 타임스탬프 형태로 변환
        new_timestamp = convert_seconds_to_timestamp(mid_seconds)
        
        # 새로운 타임스탬프 업데이트 (시간:00:00 형태로 저장)
        subtitle_data_organized[i]['timestamp'] = f"{new_timestamp}"
        print(f"new_timestamp: ", new_timestamp)
        count += 1
    
    organized_json_filename = json_filename.replace('.ko.json', '_organized.ko.json')
    # 재정렬된 자막 데이터를 다시 저장
    with open(organized_json_filename, 'w', encoding='utf-8') as f:
        json.dump(subtitle_data_organized, f, ensure_ascii=False, indent=4)

    print(f"Filtered, re-indexed, and timestamp-adjusted subtitles saved to {organized_json_filename}")

    # .vtt 파일 삭제
    if os.path.exists(json_filename):
        os.remove(json_filename)
        print(f"Deleted the original .json: {json_filename}")    

    # .vtt 파일 삭제
    if os.path.exists(subtitle_filename):
        os.remove(subtitle_filename)
        print(f"Deleted the original .vtt file: {subtitle_filename}")
    
    print("Total subtitles: ", count)
    return organized_json_filename, count


def take_screenshots(video_name, video_filename, subtitle_data, output_folder='yt_dataset/screenshots'):
    output_folder = output_folder + "/" + video_name
    print("Screenshot output folder: ", output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 영상 로드
    video = VideoFileClip(video_filename)

    # 각 자막 시간대에 맞춰 스크린샷 찍기
    for sub in subtitle_data:
        timestamp = sub['timestamp']  # 여기 오류남.
        frame_number = sub['frame']  # sub['frame']이 정수인지 확인
        if isinstance(frame_number, int):
            # 네 자리로 맞춰서 파일 이름 생성 (예: 1 -> 0001, 14 -> 0014)
            output_file = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
            frame = video.get_frame(timestamp)
            Image.fromarray(frame).save(output_file)
            print(f"Screenshot saved: {output_file}")
        else:
            print(f"Invalid frame number: {frame_number}")

    video.close()

def main(url, video_name):
    # 영상 및 자막 다운로드
    video_filename, subtitle_filename = download_video_and_subtitles(url, video_name)
    print("영상 및 자막 다운로드 완료")
    print("비디오 파일: ", video_filename, " 자막 파일: ", subtitle_filename)

    time.sleep(2)
    # 자막 파일을 정제하고 JSON으로 저장
    json_filename, count = extract_subtitles(subtitle_filename)

    time.sleep(2)
    # 스크린샷 찍기
    subtitle_data = json.load(open(json_filename, 'r', encoding='utf-8'))
    take_screenshots(video_name, video_filename, subtitle_data)

    print("모든 작업이 완료되었습니다!")
    print("Subtitle Counts: ", count)

if __name__ == "__main__":
    # YouTube 영상 URL과 비디오 이름 입력
    video_url = input("YouTube video url: ")
    video_name = input("Video Title: ")
    main(video_url, video_name)
