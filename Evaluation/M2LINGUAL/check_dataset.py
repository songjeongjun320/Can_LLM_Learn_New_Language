#!/usr/bin/env python3
from datasets import load_dataset
from collections import Counter
import pandas as pd

def main():
    try:
        # 1. 데이터셋 로드
        print("🔍 데이터셋 로드 중...")
        ds = load_dataset("ServiceNow-AI/M2Lingual", "full_data")
        print(f"✅ 로드 완료! (전체 샘플 수: {len(ds['train'])})")
        
        # 2. 실제 데이터 구조 확인 (샘플 1개 출력)
        print("\n📊 데이터 구조 예시:")
        print(ds["train"][0])

        # 3. 한국어 데이터 필터링 (수정된 버전)
        print("\n🔄 한국어 데이터 필터링...")
        ko_data = ds["train"].filter(lambda x: x["language"] == "ko")
        print(f"🇰🇷 한국어 샘플 수: {len(ko_data)}")

        if len(ko_data) > 0:
            # 4. 한국어 샘플 분석
            print("\n📝 한국어 샘플 분석:")
            
            # 4-1. 태스크 유형 분포
            task_counts = Counter(ko_data["task"])
            print("\n🔧 태스크 유형 분포:")
            for task, count in task_counts.most_common():
                print(f"- {task}: {count}개")
            
            # 4-2. 대화 턴 수 분석
            avg_turns = sum(ko_data["no_of_turns"]) / len(ko_data)
            print(f"\n🔄 평균 대화 턴 수: {avg_turns:.1f}")
            
            # 4-3. 샘플 출력
            print("\n💬 샘플 대화:")
            sample = ko_data[0]
            for turn in sample["conversation"]:
                print(f"[{turn['role']}] {turn['content']}")

            # 5. 데이터 저장
            print("\n💾 데이터 저장 중...")
            pd.DataFrame(ko_data).to_json("m2lingual_korean.json", orient="records", force_ascii=False)
            print("✅ 저장 완료: m2lingual_korean.json")
        else:
            print("⚠️ 한국어 데이터가 없습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()