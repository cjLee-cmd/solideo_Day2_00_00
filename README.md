# 시스템 리소스 실시간 모니터링 시스템

실시간으로 시스템 리소스를 모니터링하고 시각화하여 PDF 리포트를 생성하는 Python 기반 모니터링 도구입니다.

## 기능

### 모니터링 항목
- **CPU 사용률**: 전체 및 코어별 CPU 사용률 추적
- **메모리 사용률**: 실시간 메모리 사용량 모니터링
- **디스크 I/O**: 읽기/쓰기 속도 측정 (MB/s)
- **네트워크 트래픽**: 송신/수신 데이터 추적 (MB/s)
- **시스템 온도**: CPU 온도 모니터링 (지원되는 경우)
- **시스템 정보**: CPU 코어 수, 메모리 크기, 디스크 용량 등

### 시각화
- 실시간 그래프 업데이트 (1초 간격)
- 6개의 동시 그래프 표시
- 색상별로 구분된 직관적인 UI
- 투명도를 활용한 영역 그래프

### 리포트 생성
- 자동 PDF 리포트 생성
- 통계 테이블 (평균, 최대, 최소값)
- 누적 데이터 전송량 표시
- 고해상도 그래프 이미지
- 시스템 정보 요약

## 설치

### 필수 요구사항
- Python 3.7 이상
- pip

### 패키지 설치

```bash
pip install -r requirements.txt
```

또는 수동 설치:

```bash
pip install psutil matplotlib reportlab Pillow
```

## 사용법

### 기본 실행 (5분 모니터링)

```bash
python3 system_monitor.py
```

### 비대화형 모드 (GUI 없이 실행)

```bash
python3 system_monitor_headless.py
```

### 실행 결과

1. **실시간 그래프 창**: 모니터링 중 실시간으로 데이터 시각화
2. **PDF 리포트**: `system_monitoring_report.pdf` 파일 자동 생성
3. **콘솔 통계**: 모니터링 완료 후 주요 통계 출력

## 출력 파일

### system_monitoring_report.pdf
- **1페이지**: 통계 테이블 (평균/최대/최소값)
- **2-3페이지**: 시각화 그래프
  - CPU 사용률 추이
  - 메모리 사용률 추이
  - 디스크 I/O 추이
  - 네트워크 트래픽 추이
  - 코어별 CPU 사용률
- **4페이지**: 시스템 정보

## 모니터링 항목 상세

### CPU 모니터링
- 전체 CPU 사용률 (%)
- 코어별 개별 사용률
- 평균/최대/최소 사용률 통계

### 메모리 모니터링
- 실시간 메모리 사용률 (%)
- 총 메모리 용량
- 사용 중인 메모리 크기

### 디스크 I/O
- 읽기 속도 (MB/s)
- 쓰기 속도 (MB/s)
- 총 읽기/쓰기 데이터량

### 네트워크 트래픽
- 송신 속도 (MB/s)
- 수신 속도 (MB/s)
- 총 송신/수신 데이터량

### 시스템 온도
- CPU 온도 (지원되는 경우)
- 평균/최대/최소 온도

## 기술 스택

- **psutil**: 시스템 리소스 정보 수집
- **matplotlib**: 실시간 그래프 시각화
- **reportlab**: PDF 리포트 생성
- **Pillow**: 이미지 처리

## 시스템 요구사항

- **운영체제**: Linux, Windows, macOS
- **Python**: 3.7+
- **메모리**: 최소 512MB
- **디스크**: 최소 100MB 여유 공간

## 주의사항

- 모니터링 중에는 시스템 리소스를 약간 사용합니다
- GUI 환경이 없는 경우 headless 버전을 사용하세요
- 온도 센서는 시스템에 따라 지원되지 않을 수 있습니다
- root 권한이 필요할 수 있습니다 (온도 센서 접근)

## 문제 해결

### GUI 창이 열리지 않는 경우
```bash
# 비대화형 모드 사용
python3 system_monitor_headless.py
```

### 패키지 설치 오류
```bash
# 사용자 디렉토리에 설치
pip install --user -r requirements.txt
```

### 권한 오류
```bash
# sudo로 실행 (Linux/macOS)
sudo python3 system_monitor.py
```

## 커스터마이징

### 모니터링 시간 변경
```python
# system_monitor.py 수정
monitor = SystemMonitor(duration=600, update_interval=1)  # 10분
```

### 업데이트 간격 변경
```python
monitor = SystemMonitor(duration=300, update_interval=2)  # 2초마다 업데이트
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 버전

- **v1.0.0** (2025-11-06): 초기 릴리스
  - 실시간 모니터링 기능
  - PDF 리포트 생성
  - 6가지 리소스 추적
