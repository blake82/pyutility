# 서비스 활성
sudo systemctl daemon-reload
sudo systemctl enable sys-monitor.service
sudo systemctl start sys-monitor.service

# 서비스 재시작
sudo systemctl restart sys-monitor.service

