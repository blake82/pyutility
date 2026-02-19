#!/bin/bash

# 설정
CPU_THRESHOLD=80.0
CHECK_INTERVAL=30
LOG_FILE="/tada/monitoring/system_monitor.log"
TARGET_USER="blake"

while true; do
    # 1. 고부하 CPU 프로세스 확인
    TOP_PROC=$(ps -eo pcpu,pid,user,args --sort=-pcpu | head -n 2 | tail -n 1)
    CPU_USAGE=$(echo $TOP_PROC | awk '{print $1}')
    PROC_NAME=$(echo $TOP_PROC | awk '{print $4}')

    if (( $(echo "$CPU_USAGE > $CPU_THRESHOLD" | bc -l) )); then
        MSG="[⚠️ 경고] 고부하 프로세스 감지: $PROC_NAME (CPU: $CPU_USAGE%)"
        echo "$(date): $MSG" >> $LOG_FILE
        wall -n "$MSG" # 접속 중인 모든 터미널에 공지
    fi

    # 2. 알려진 악성 프로세스(syst3md) 감시
    if pgrep -f "syst3md" > /dev/null; then
        MSG="[🚨 긴급] 악성 프로세스 'syst3md' 재감지됨!"
        echo "$(date): $MSG" >> $LOG_FILE
        wall -n "$MSG"
    fi

    # 3. 비정상적 외부 접속 시도 (실패한 SSH 로그인) 확인
    FAILED_ATTEMPTS=$(tail -n 20 /var/log/auth.log | grep "Failed password" | wc -l)
    if [ "$FAILED_ATTEMPTS" -gt 5 ]; then
        MSG="[🔐 보안] 다수의 SSH 로그인 실패 감지!"
        echo "$(date): $MSG" >> $LOG_FILE
        wall -n "$MSG"
    fi

    sleep $CHECK_INTERVAL
done
