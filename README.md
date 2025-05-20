# 던파 스펙업 가이드 AI 챗봇

던전앤파이터(DNF) 게임의 캐릭터 스펙업에 대한 지능형 가이드를 제공하는 AI 챗봇입니다. 내부 데이터베이스와 웹 검색을 결합한 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 사용하여 사용자의 질문에 정확하고 최신 정보를 제공합니다.

## 기능

- **하이브리드 RAG 시스템**: 내부 데이터베이스(30%)와 웹 검색(70%)을 결합하여 정확하고 최신 정보 제공
- **다양한 정보 소스**: 공식 홈페이지, 디시인사이드, 아카라이브, 유튜브 등 다양한 소스에서 데이터 수집
- **맞춤형 답변**: 사용자 질문에 따른 정확한 스펙업 가이드 제공
- **간결한 응답**: 핵심 정보만 3-5문장으로 정리하여 제공

## 설치 방법

1. 저장소 클론
   ```
   git clone https://github.com/yourusername/dunfa-specup-guide.git
   cd dunfa-specup-guide
   ```

2. 의존성 설치
   ```
   pip install -r requirements.txt
   ```

3. 환경 변수 설정
   ```
   # .env 파일 생성 및 API 키 설정
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

## 사용 방법

### 데이터 크롤링 및 벡터 DB 구축
```
# 크롤링 실행
python -m crawlers.crawler --pages 5 --depth 2 --yt-list data/youtube_ids.txt

# 데이터 전처리
python -m preprocessing.preprocess

# 벡터 DB 구축
python -m vectorstore.build_vector_db
```

### 챗봇 실행
```
# 대화형 모드
python rag_chat.py

# 명령행 인수로 질문 전달
python rag_chat.py 명성 5만으로 할 수 있는 던전은?
```

## 시스템 구조

1. **크롤링 모듈**: 다양한 소스에서 던파 관련 정보 수집
2. **전처리 모듈**: 수집된 데이터 정제 및 청킹
3. **벡터 저장소**: 전처리된 문서를 임베딩하여 검색 가능한 형태로 저장
4. **RAG 시스템**: 내부 벡터 DB와 Perplexity API를 사용한 웹 검색을 결합
5. **응답 생성**: 검색된 정보를 기반으로 LLM을 사용해 응답 생성

## 라이선스

MIT License

## 주의사항

- 이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.
- 실제 게임 내 결정은 자신의 판단에 따라 이루어져야 합니다.
- API 키는 절대 공개 저장소에 커밋하지 마세요.
