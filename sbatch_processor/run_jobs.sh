#!/bin/bash
# 작업 목록을 배열로 정의
drama_list=(
  "여러분들은 대전에 대해 오해가 많으십니다"
  "여우짓 본인 시점"
)
host="http://sg007:11434"

# 각 작업을 순차적으로 처리
for drama in "${drama_list[@]}"; do
  echo "처리 중: $drama"
  
  # 작업 제출 및 작업 ID 저장
  job_id=$(./extract_job.sh --drama "$drama" --host "$host" | grep -oP 'Submitted batch job \K\d+')
  
  if [ -n "$job_id" ]; then
    echo "작업 ID: $job_id - 제출됨, 완료를 기다리는 중..."
    
    # 이 작업이 완료될 때까지 기다림
    scontrol wait jobid=$job_id
    
    # 작업 상태 확인
    if [ $? -eq 0 ]; then
      echo "완료: $drama (작업 ID: $job_id)"
    else
      echo "실패: $drama (작업 ID: $job_id)"
      exit 1
    fi
  else
    echo "작업 제출 실패: $drama"
    exit 1
  fi
  
  # 다음 작업으로 넘어가기 전에 잠시 대기 (필요시)
  sleep 2
done

echo "모든 작업이 완료되었습니다."