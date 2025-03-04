import json

def extract_timestamp(json_path="C:\\Users\\songj\\OneDrive\\Desktop\\Can-LLM-learn-new-Language\\Samples_Result\\60일,.지정생존자\\60일,.지정생존자.S01E01.WEBRip.Netflix.ko[cc].json"):
    # timestamp 값을 저장할 리스트
    timestamp_list = []

    # JSON 파일 열기
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # JSON 데이터 로드
        
        # 각 항목의 timestamp에서 --> 앞부분만 추출하여 리스트에 추가
        for item in data:
            timestamp = item.get("timestamp", "")
            start_time = timestamp.split("-->")[0].strip()  # --> 앞의 시간만 추출
            timestamp_list.append(start_time)

    # 결과 출력
    # print(timestamp_list)
    print(len(timestamp_list))
    return timestamp_list