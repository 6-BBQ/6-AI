# 🎮 던파 스펙업 가이드 AI 챗봇

던전앤파이터(DNF) 게임의 종합적인 지능형 가이드를 제공하는 AI 챗봇입니다. 내부 데이터베이스와 웹 검색을 결합한 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 사용하여 사용자의 질문에 정확하고 최신 정보를 제공합니다.

> **🚀 프로덕션 레디**: 이 프로젝트는 로깅, 모니터링, 보안 기능이 강화되어 실제 운영 환경에 배포 가능합니다.

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
- **동영상**: 유튜브 가이드 영상 및 트랜스크립트 (선택적)
- **실시간 검색**: Gemini를 통한 최신 정보 수집

## 🚀 빠른 시작

### 🎯 원클릭 배포 (추천)

**Linux/Mac:**
```bash
bash deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

위 스크립트가 자동으로 다음을 처리합니다:
- 환경 설정 및 검증
- 가상환경 생성 및 의존성 설치
- 데이터 파이프라인 실행
- API 서버 시작

### 🔍 서비스 상태 확인

```bash
python health_check.py
```

### 🛠️ 수동 설치

#### 1. 환경 준비
```bash
# 저장소 클론
git clone https://github.com/6-BBQ/6-AI.git
cd 6-AI

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

#### 2. 환경 변수 설정
```bash
# 환경변수 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
# 필수 설정:
# - GEMINI_API_KEY: Google AI Studio에서 발급
# - JWT_SECRET_KEY: JWT 토큰용 시크릿 키
```

**주요 환경변수:**
```bash
# 필수 API 키
GEMINI_API_KEY=your_gemini_api_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# 임베딩 모델 설정 (gemini 권장)
EMBEDDING_TYPE=gemini
EMBED_MODEL_NAME=models/text-embedding-004

# 또는 HuggingFace 한국어 모델 (무료)
# EMBEDDING_TYPE=huggingface
# EMBED_MODEL_NAME=dragonkue/bge-m3-ko

# 환경 설정
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_WEB_GROUNDING=true
```

#### 3. 데이터 준비 (초기 설정)
```bash
# 전체 파이프라인 실행 (크롤링 → 전처리 → 벡터 DB 구축)
python pipeline.py

# 또는 옵션별 실행
python pipeline.py --full          # 전체 재처리
python pipeline.py --skip-crawl    # 전처리부터 실행
python pipeline.py --pages 100     # 크롤링 페이지 수 조정
```

#### 4. API 서버 실행
```bash
# FastAPI 서버 시작
python -m api.main

# 또는 uvicorn 직접 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### 5. API 테스트
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
│   └── crawler.py           # 통합 크롤링 스크립트
├── 🔧 preprocessing/       # 데이터 전처리
│   └── preprocess.py        # 텍스트 정제 및 청킹
├── 🗄️ vectorstore/        # 벡터 데이터베이스
│   └── build_vector_db.py   # ChromaDB 구축
├── 🧠 rag/                # RAG 시스템 핵심
│   ├── service.py           # 하이브리드 검색 + LLM 생성
│   ├── retrievers.py        # 검색 엔진 컴포넌트
│   ├── search_factory.py    # 검색 팩토리
│   ├── cache_utils.py       # 캐시 관리
│   └── text_utils.py        # 텍스트 처리 유틸
├── 🌐 api/                # FastAPI 서버
│   ├── main.py             # 메인 애플리케이션
│   ├── endpoints.py        # API 엔드포인트
│   ├── models.py           # 데이터 모델
│   └── auth.py             # JWT 인증
├── 🛠️ utils/              # 유틸리티
│   └── logger.py           # 로깅 시스템
├── 📄 config.py           # 중앙화된 설정 관리
├── 🚀 pipeline.py         # 전체 데이터 파이프라인
└── 📋 requirements.txt    # Python 의존성
```

## 🔄 데이터 처리 파이프라인

1. **📥 데이터 수집**: 다양한 소스에서 던파 관련 정보 크롤링
2. **🧹 데이터 정제**: HTML 태그 제거, 텍스트 정규화, 중복 제거
3. **✂️ 청킹**: 긴 문서를 검색 최적화된 작은 조각으로 분할
4. **🔢 벡터화**: Google Gemini 또는 HuggingFace 임베딩으로 텍스트를 벡터로 변환
5. **💾 저장**: ChromaDB에 벡터 및 메타데이터 저장
6. **🔍 검색**: 사용자 질문에 대해 하이브리드 검색 수행
7. **🤖 생성**: Gemini를 사용해 검색 결과 기반 답변 생성

## 🔧 주요 기술 스택

### 핵심 프레임워크
- **FastAPI**: 고성능 웹 API 프레임워크
- **LangChain**: RAG 시스템 구축 프레임워크
- **ChromaDB**: 벡터 데이터베이스

### AI/ML 모델
- **Google Gemini**: LLM 및 임베딩 (메인)
- **Cross-Encoder**: 검색 결과 재랭킹

### 데이터 수집
- **CloudScraper**: Cloudflare 우회

### 한국어 NLP
- **Kiwi**: 한국어 형태소 분석

## 📊 로깅 및 모니터링

### 로깅 시스템
체계적인 로깅 시스템을 제공합니다:

- **단계별 로깅**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **파일 로테이션**: 로그 파일 자동 순환 (10MB 단위)
- **에러 전용 로그**: 에러/오류 별도 추적
- **색상 로그**: 개발 환경에서 가독성 향상

```bash
# 로그 파일 위치
logs/
├── api_main.log          # API 서버 로그
├── api_endpoints.log     # 엔드포인트 로그
├── rag_service.log       # RAG 서비스 로그
├── crawler.log           # 크롤링 로그
└── *_error.log           # 에러 전용 로그
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

## 🚀 배포 가이드

### Docker 배포 (권장)

```dockerfile
# Dockerfile 예시
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "api.main"]
```

```yaml
# docker-compose.yml 예시
version: '3.8'
services:
  df-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./vector_db:/app/vector_db
      - ./logs:/app/logs
```

### AWS EC2 배포

#### 1. EC2 인스턴스 설정
```bash
# 권장 사양: t3.large (2 vCPU, 8GB RAM) 이상
# GPU 사용 시: g4dn.xlarge (Tesla T4 GPU)

# Ubuntu 22.04 LTS 기본 설정
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y

# GPU 인스턴스용 NVIDIA 드라이버 설치
sudo apt install nvidia-driver-470 -y
```

#### 2. 애플리케이션 배포
```bash
# 프로젝트 클론
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

# 데이터 파이프라인 실행
python pipeline.py --pages 100

# 서비스 시작
python -m api.main
```

#### 3. Systemd 서비스 등록
```bash
# 서비스 파일 생성
sudo tee /etc/systemd/system/df-ai.service > /dev/null <<EOF
[Unit]
Description=DF AI Service
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
EOF

# 서비스 활성화
sudo systemctl enable df-ai
sudo systemctl start df-ai
sudo systemctl status df-ai
```

#### 4. Nginx 리버스 프록시
```bash
# Nginx 설치
sudo apt install nginx -y

# 설정 파일 생성
sudo tee /etc/nginx/sites-available/df-ai > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # 로그 파일 액세스 제한
    location /logs {
        deny all;
    }
}
EOF

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/df-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🔒 보안 고려사항

### API 보안
- JWT 토큰 기반 인증
- CORS 정책 적용 (운영 환경에서 도메인 제한)
- 입력 값 검증 및 살균
- 레이트 리미팅 적용

### 데이터 보안
- API 키 환경변수 관리 (.env 파일은 Git에서 제외)
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

## 🔧 환경별 설정

### 개발 환경
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_WEB_GROUNDING=true
EMBEDDING_TYPE=huggingface  # 무료 옵션
```

### 운영 환경
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_WEB_GROUNDING=true
EMBEDDING_TYPE=gemini       # 성능 최적화
ALLOWED_ORIGINS=https://your-domain.com
```

## 📈 성능 최적화

### 캐시 최적화
- BM25 검색기 캐시 (12시간)
- Cross-Encoder 모델 캐시 (24시간)
- 검색 결과 캐시 (쿼리별)

### GPU 가속
```bash
# GPU 사용 설정
DEVICE=cuda

# 또는 자동 감지
DEVICE=auto
```

### 배치 처리 최적화
```bash
# 임베딩 배치 크기 조정
EMBED_BATCH_SIZE=200  # 기본값
EMBED_BATCH_SIZE=100  # 메모리 부족 시
EMBED_BATCH_SIZE=500  # 고성능 GPU 사용 시
```

## 🔍 모니터링 및 문제해결

### 로그 모니터링
```bash
# 실시간 로그 확인
tail -f logs/api_main.log

# 에러 로그 확인
tail -f logs/*_error.log

# 특정 패턴 검색
grep "ERROR" logs/*.log
```

### 일반적인 문제 해결

**1. API 키 오류**
```bash
# .env 파일 확인
cat .env | grep API_KEY

# 환경변수 로드 확인
python -c "from config import config; print(config.GEMINI_API_KEY[:10] + '...')"
```

**2. 메모리 부족**
```bash
# 배치 크기 줄이기
echo "EMBED_BATCH_SIZE=50" >> .env

# 캐시 정리
rm -rf cache/*
```

**3. 크롤링 실패**
```bash
# 네트워크 연결 확인
ping df.nexon.com

# 크롤링 재시도
python crawlers/crawler.py --pages 10 --sources official
```