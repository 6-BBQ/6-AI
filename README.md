# 🎮 던파 스펙업 가이드 AI 챗봇

던전앤파이터(DNF) 게임의 캐릭터 스펙업에 대한 지능형 가이드를 제공하는 AI 챗봇입니다. 내부 데이터베이스와 웹 검색을 결합한 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 사용하여 사용자의 질문에 정확하고 최신 정보를 제공합니다.

> **프로덕션 레디**: 이 프로젝트는 로깅, 모니터링, 보안 기능이 강화되어 실제 운영 환경에 배포 가능합니다.

## ✨ 주요 기능

### 🔍 하이브리드 RAG 시스템
- **내부 데이터베이스**: 크롤링된 던파 커뮤니티 정보를 벡터화하여 저장
- **실시간 웹 검색**: Gemini Search Grounding을 통한 최신 정보 검색
- **지능형 검색**: BM25 + 벡터 검색 + Cross-Encoder 재랭킹으로 정확도 향상

### 🎯 맞춤형 스펙업 가이드
- **캐릭터 정보 기반**: 직업, 명성, 장비 정보를 고려한 개인화된 조언
- **실시간 이벤트 정보**: 진행 중인 이벤트 및 업데이트 반영
- **단계별 가이드**: 현재 상황에서 다음 단계로의 명확한 로드맵 제시

### 📊 다양한 정보 소스
- **공식 채널**: 던파 공식 홈페이지, 공지사항
- **커뮤니티**: 디시인사이드, 아카라이브 게시글
- **동영상**: 유튜브 가이드 영상 및 트랜스크립트
- **실시간 검색**: Google을 통한 최신 정보 수집

## 🚀 빠른 시작

### 1. 프로젝트 설치
```bash
# 저장소 클론
git clone https://github.com/6-BBQ/6-AI.git
cd 6-AI

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# 필요한 API 키 설정
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
JWT_SECRET_KEY=your_jwt_secret_key

# 로깅 및 운영 환경 설정
ENVIRONMENT=development  # development 또는 production
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE=true         # 파일 로깅 여부
ENABLE_WEB_GROUNDING=true # 웹 검색 그라운딩 활성화
```

### 3. 데이터 준비 (초기 설정)
```bash
# 전체 파이프라인 실행 (크롤링 → 전처리 → 벡터 DB 구축)
python pipeline.py

# 또는 단계별 실행
python crawlers/crawler.py --pages 20 --merge --incremental
python preprocessing/preprocess.py
python vectorstore/build_vector_db.py
```

### 4. API 서버 실행
```bash
# FastAPI 서버 시작
python -m api.main

### 5. API 테스트
```bash
# 테스트 실행
python test.py

# 또는 브라우저에서 API 문서 확인
# http://localhost:8000/docs
```

## 🏗️ 시스템 아키텍처

```
📁 프로젝트 구조
├── 🕷️ crawlers/           # 데이터 수집 모듈
│   ├── official_crawler.py  # 공식 홈페이지 크롤러
│   ├── dc_crawler.py        # 디시인사이드 크롤러
│   ├── arca_crawler.py      # 아카라이브 크롤러
│   └── youtube_crawler.py   # 유튜브 크롤러
├── 🔧 preprocessing/       # 데이터 전처리
│   └── preprocess.py        # 텍스트 정제 및 청킹
├── 🗄️ vectorstore/        # 벡터 데이터베이스
│   └── build_vector_db.py   # ChromaDB 구축
├── 🧠 rag/                # RAG 시스템 핵심
│   └── rag_service.py       # 하이브리드 검색 + LLM 생성
├── 🌐 api/                # FastAPI 서버
│   ├── main.py             # 메인 애플리케이션
│   ├── endpoints.py        # API 엔드포인트
│   ├── models.py           # 데이터 모델
│   └── auth.py             # JWT 인증
└── 🚀 deploy/             # 배포 관련
    ├── setup_ec2.sh        # EC2 자동 설정
    └── README.md           # 배포 가이드
```

## 🔄 데이터 처리 파이프라인

1. **📥 데이터 수집**: 다양한 소스에서 던파 관련 정보 크롤링
2. **🧹 데이터 정제**: HTML 태그 제거, 텍스트 정규화, 중복 제거
3. **✂️ 청킹**: 긴 문서를 검색 최적화된 작은 조각으로 분할
4. **🔢 벡터화**: OpenAI Embeddings로 텍스트를 벡터로 변환
5. **💾 저장**: ChromaDB에 벡터 및 메타데이터 저장
6. **🔍 검색**: 사용자 질문에 대해 하이브리드 검색 수행
7. **🤖 생성**: Gemini를 사용해 검색 결과 기반 답변 생성

## 📊 로깅 및 모니터링

### 로깅 시스템
이 프로젝트는 체계적인 로깅 시스템을 제공합니다:

- **단계별 로깅**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **파일 로테이션**: 로그 파일 자동 순환 (10MB 단위)
- **에러 전용 로그**: 에러/오류 별도 추적
- **색상 로그**: 개발 환경에서 가독성 향상

```bash
# 로그 파일 위치
logs/
├── api_main.log          # API 서버 로그
├── api_endpoints.log      # 엔드포인트 로그
├── rag_service.log        # RAG 서비스 로그
├── crawlers_crawler.log   # 크롤링 로그
└── *_error.log            # 에러 전용 로그
```

### 성능 모니터링
```python
# 실행 시간 데코레이터 사용 예시
from utils import log_execution_time

@log_execution_time()
def your_function():
    # 함수 실행 시간이 자동으로 로깅됨
    pass
```

### 시스템 정보 로깅
서버 시작 시 자동으로 로깅되는 정보:
- OS 및 Python 버전
- CPU 코어 수 및 메모리 용량
- GPU 정보 (사용 가능한 경우)
- 환경 설정 및 로그 레벨

## 🚀 대규모 배포 가이드

### AWS EC2에 배포

#### 1. EC2 인스턴스 설정
```bash
# 권장 사양: g4dn.xlarge (GPU 지원)
# - 4 vCPU, 16GB RAM, Tesla T4 GPU
# - 연간 예상 비용: $2-3 (Spot Instance 사용 시)

# Ubuntu 22.04 LTS 기본 설정
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y

# NVIDIA 드라이버 설치 (GPU 인스턴스용)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-drivers
```

#### 2. 애플리케이션 배포
```bash
# 프로젝트 복제
git clone https://github.com/6-BBQ/6-AI.git
cd 6-AI

# 가상환경 설정
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 편집으로 API 키 설정

# 운영 환경 로깅 설정
echo "ENVIRONMENT=production" >> .env
echo "LOG_LEVEL=INFO" >> .env
echo "LOG_TO_FILE=true" >> .env
```

#### 3. Systemd 서비스 등록
```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/df-rag-api.service

# 다음 내용 입력:
[Unit]
Description=DF RAG API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/6-AI
Environment=PATH=/home/ubuntu/6-AI/venv/bin
ExecStart=/home/ubuntu/6-AI/venv/bin/python -m api.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# 서비스 활성화
sudo systemctl enable df-rag-api
sudo systemctl start df-rag-api
sudo systemctl status df-rag-api
```

#### 4. Nginx 리버스 프록시 설정
```bash
# Nginx 설치
sudo apt install nginx -y

# 설정 파일 생성
sudo nano /etc/nginx/sites-available/df-rag-api

# 다음 내용 입력:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 로그 파일 액세스 제한
    location /logs {
        deny all;
    }
}

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/df-rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 비용 최적화 전략

#### Spot Instance 사용
```bash
# AWS CLI를 통한 Spot Instance 요청
aws ec2 request-spot-instances \
    --spot-price "0.16" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification \
    '{"ImageId":"ami-xxxxx","InstanceType":"g4dn.xlarge","SecurityGroupIds":["sg-xxxxx"]}'
```

#### 자동 스케줄링
```bash
# 주간 임베딩 작업 자동화
crontab -e

# 매주 일요일 새벽 2시에 증분 크롤링 및 임베딩
0 2 * * 0 cd /home/ubuntu/6-AI && /home/ubuntu/6-AI/venv/bin/python pipeline.py --incremental
```

## 🔍 모니터링 및 알림

### 로그 모니터링
```bash
# 주요 로그 모니터링 명령어

# 실시간 로그 확인
tail -f logs/api_main.log

# 에러 로그 확인
tail -f logs/*_error.log

# 로그 크기 확인
du -sh logs/

# 오래된 로그 정리 (30일 이상)
find logs/ -name "*.log.*" -mtime +30 -delete
```

### 시스템 메트릭스
```bash
# GPU 사용률 모니터링
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 5

# 메모리 사용률
free -h

# 디스크 사용률
df -h

# 프로세스 모니터링
ps aux | grep python
```

## 🔒 보안 및 노출 예방

### API 보안
- JWT 토큰 기반 인증
- CORS 정책 적용 (운영 환경에서 도메인 제한)
- 입력 값 검증 및 살짜

### 데이터 보안
- API 키 환경변수 관리
- 로그 내 민감 정보 마스킹
- 데이터베이스 액세스 제한

### 네트워크 보안
```bash
# 방화벽 설정 (UFW)
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# SSL 인증서 설정 (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

## ⚠️ 주의사항

- **교육 목적**: 이 프로젝트는 교육 및 연구 목적으로 제작되었습니다
- **게임 판단**: 실제 게임 내 결정은 자신의 판단에 따라 이루어져야 합니다
- **API 보안**: API 키는 절대 공개 저장소에 커밋하지 마세요
- **사용 제한**: 크롤링 시 해당 사이트의 robots.txt를 준수하세요
- **업데이트**: 게임 업데이트에 따라 정보가 변경될 수 있습니다
- **로그 관리**: 로그 파일이 커질 수 있으므로 정기적인 정리가 필요합니다
- **리소스 모니터링**: GPU 메모리 부족 시 모델 로딩이 실패할 수 있습니다

---

**던파 스펙업의 새로운 차원을 경험해보세요! 🚀**
