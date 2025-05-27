# 🎮 던파 스펙업 가이드 AI 챗봇

던전앤파이터(DNF) 게임의 캐릭터 스펙업에 대한 지능형 가이드를 제공하는 AI 챗봇입니다. 내부 데이터베이스와 웹 검색을 결합한 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 사용하여 사용자의 질문에 정확하고 최신 정보를 제공합니다.

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

## ⚠️ 주의사항

- **교육 목적**: 이 프로젝트는 교육 및 연구 목적으로 제작되었습니다
- **게임 판단**: 실제 게임 내 결정은 자신의 판단에 따라 이루어져야 합니다
- **API 보안**: API 키는 절대 공개 저장소에 커밋하지 마세요
- **사용 제한**: 크롤링 시 해당 사이트의 robots.txt를 준수하세요
- **업데이트**: 게임 업데이트에 따라 정보가 변경될 수 있습니다

---

**던파 스펙업의 새로운 차원을 경험해보세요! 🚀**
