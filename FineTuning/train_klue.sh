#!/bin/bash
#SBATCH --job-name=train_klue  # 작업 이름
#SBATCH --output=train_klue_%j.log  # 출력 로그 파일
#SBATCH --error=train_klue_%j.err  # 오류 로그 파일
#SBATCH --time=12:00:00  # 최대 실행 시간 (12시간)
#SBATCH --partition=gpu  # GPU 노드를 사용
#SBATCH --gres=gpu:1  # 1개의 GPU 할당
#SBATCH --mem=80G  # 메모리 64GB 할당
#SBATCH --cpus-per-task=8  # 8개의 CPU 할당
#SBATCH --exclusive  # 독점 자원 할당 (여러 작업과 공유하지 않음)

# 가상 환경 활성화 (필요한 경우)
# module load python/3.8  # SLURM에서 Python 모듈이 필요하다면 로드
source /scratch/jsong132/Can_LLM_Learn_New_Language/venv/bin/activate  # 가상 환경 경로로 수정

# 필요한 패키지 설치 (가상환경에 이미 설치된 경우 생략)
# pip install -r requirements.txt

# Python 스크립트 실행
python3 train_klue.py  # train_klue.py는 사용자가 제공한 파이썬 코드 파일명으로 수정
