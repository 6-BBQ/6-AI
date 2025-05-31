"""
6-AI 프로젝트 설정 관리
환경변수 기반 중앙화된 설정 시스템
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """중앙화된 설정 관리 클래스"""
    
    # ================================
    # 🔑 API 키 설정
    # ================================
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
    
    # ================================
    # 🏗️ 서버 및 환경 설정
    # ================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    PORT: int = int(os.getenv("PORT", "8000"))
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    
    # ================================
    # 📊 로깅 설정
    # ================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    LOG_SYSTEM_INFO: bool = os.getenv("LOG_SYSTEM_INFO", "true").lower() == "true"
    
    # ================================
    # 🤖 RAG 시스템 설정
    # ================================
    ENABLE_WEB_GROUNDING: bool = os.getenv("ENABLE_WEB_GROUNDING", "true").lower() == "true"
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "models/text-embedding-004")
    EMBEDDING_TYPE: str = os.getenv("EMBEDDING_TYPE", "gemini")  # gemini, models/text-embedding-004 // huggingface, dragonkue/bge-m3-ko
    CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-pro-preview-05-06")
    
    # ================================
    # 💾 데이터베이스 설정
    # ================================
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "vector_db/chroma")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "cache")
    PROCESSED_DOCS_PATH: str = os.getenv(
        "PROCESSED_DOCS_PATH", "data/processed/processed_docs.jsonl"
    )
    VECTORDB_CACHE_PATH: str = os.getenv(
        "VECTORDB_CACHE_PATH", "vector_db/vectordb_cache.json"
    )
    JOB_EMBEDDINGS_PATH: str = os.getenv(
        "JOB_EMBEDDINGS_PATH", "vector_db/job_embeddings.json"
    )
    JOB_NAMES_PATH: str = os.getenv(
        "JOB_NAMES_PATH", "job_names.json"
    )
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "200"))
    JOB_SIMILARITY_THRESHOLD: float = float(
        os.getenv("JOB_SIMILARITY_THRESHOLD", "0.75")
    )
    
    # ================================
    # 🕷️ 크롤링 설정
    # ================================
    DEFAULT_CRAWL_PAGES: int = int(os.getenv("DEFAULT_CRAWL_PAGES", "10"))
    DEFAULT_CRAWL_DEPTH: int = int(os.getenv("DEFAULT_CRAWL_DEPTH", "2"))
    VISITED_URLS_PATH: str = os.getenv("VISITED_URLS_PATH", "data/visited_urls.json")
    
    # 크롤러별 URL 설정
    OFFICIAL_BASE_URL: str = os.getenv("OFFICIAL_BASE_URL", "https://df.nexon.com")
    DC_BASE_URL: str = os.getenv("DC_BASE_URL", "https://gall.dcinside.com")
    ARCA_BASE_URL: str = os.getenv("ARCA_BASE_URL", "https://arca.live")
    
    # 크롤러 요청 설정
    CRAWLER_USER_AGENT: str = os.getenv("CRAWLER_USER_AGENT", "Mozilla/5.0")
    CRAWLER_TIMEOUT: int = int(os.getenv("CRAWLER_TIMEOUT", "10"))
    CRAWLER_DELAY: float = float(os.getenv("CRAWLER_DELAY", "0.05"))
    DC_CRAWLER_TIMEOUT: int = int(os.getenv("DC_CRAWLER_TIMEOUT", "30"))
    DC_CRAWLER_DELAY: float = float(os.getenv("DC_CRAWLER_DELAY", "3"))
    ARCA_CRAWLER_DELAY: float = float(os.getenv("ARCA_CRAWLER_DELAY", "0.1"))
    ARCA_CRAWLER_TIMEOUT: int = int(os.getenv("ARCA_CRAWLER_TIMEOUT", "15"))
    
    # 품질 임계값 설정
    OFFICIAL_QUALITY_THRESHOLD: int = int(os.getenv("OFFICIAL_QUALITY_THRESHOLD", "30"))
    DC_QUALITY_THRESHOLD: int = int(os.getenv("DC_QUALITY_THRESHOLD", "20"))
    ARCA_QUALITY_THRESHOLD: int = int(os.getenv("ARCA_QUALITY_THRESHOLD", "25"))
    GUIDE_QUALITY_THRESHOLD: int = int(os.getenv("GUIDE_QUALITY_THRESHOLD", "25"))
    
    # 저장 경로 설정
    RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "data/raw")
    OFFICIAL_RAW_PATH: str = os.getenv("OFFICIAL_RAW_PATH", "data/raw/official_raw.json")
    DC_RAW_PATH: str = os.getenv("DC_RAW_PATH", "data/raw/dc_raw.json")
    ARCA_RAW_PATH: str = os.getenv("ARCA_RAW_PATH", "data/raw/arca_raw.json")
    
    # 필터 키워드 설정 (문자열로 저장하고 런타임에 분할)
    FILTER_KEYWORDS: str = os.getenv(
        "FILTER_KEYWORDS", 
        "명성,상급 던전,스펙업,장비,파밍,뉴비,융합석,중천,세트,가이드,에픽,태초,레기온,레이드,현질,세리아,마법부여,스킬트리,종말의 숭배자,베누스,나벨"
    )
    EXCLUDE_KEYWORDS: str = os.getenv(
        "EXCLUDE_KEYWORDS",
        "이벤트,선계,커스텀,카지노,기록실,서고,바칼,이스핀즈,어둑섬,깨어난 숲,ㅅㅂ,ㅂㅅ,ㅄ,ㅗ,시발,씨발,병신,좆"
    )
    
    # 사이트별 정규화 설정 (JSON 문자열로 저장)
    SITE_NORMALIZATION_CONFIG: str = os.getenv(
        "SITE_NORMALIZATION_CONFIG",
        '{"arca":{"views_base":5000,"likes_base":20,"likes_ratio_range":[0.003,0.015]},"dcinside":{"views_base":15000,"likes_base":30,"likes_ratio_range":[0.0015,0.008]},"official":{"views_base":120000,"likes_base":50,"likes_ratio_range":[0.0002,0.002]}}'
    )
    
    # ================================
    # 🕷️ 전처리 설정
    # ================================
    MERGED_DIR: str = os.getenv("MERGED_DIR", "data/merged")
    PROCESSED_SAVE_PATH: str = os.getenv(
        "PROCESSED_SAVE_PATH", "data/processed/processed_docs.jsonl"
    )
    PROCESSED_CACHE_PATH: str = os.getenv(
        "PROCESSED_CACHE_PATH", "data/processed/processed_cache.json"
    )
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ================================
    # 🕷️ 파이프라인 설정
    # ================================
    CRAWLER_SCRIPT: str = os.getenv("CRAWLER_SCRIPT", "crawlers/crawler.py")
    PREPROCESS_SCRIPT: str = os.getenv("PREPROCESS_SCRIPT", "preprocessing/preprocess.py")
    BUILD_VECTORDB_SCRIPT: str = os.getenv("BUILD_VECTORDB_SCRIPT", "vectorstore/build_vector_db.py")
    
    # ================================
    # ⚡ 성능 설정
    # ================================
    CACHE_EXPIRY_SHORT: int = int(os.getenv("CACHE_EXPIRY_SHORT", "43200"))  # 12시간
    CACHE_EXPIRY_LONG: int = int(os.getenv("CACHE_EXPIRY_LONG", "86400"))    # 24시간
    DEVICE: str = os.getenv("DEVICE", "auto")
    
    # ================================
    # 🔒 보안 설정
    # ================================
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "60"))
    
    # ================================
    # 📈 모니터링 설정
    # ================================
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    
    @classmethod
    def validate_required_keys(cls) -> bool:
        """필수 API 키들이 설정되어 있는지 확인"""
        required_keys = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("JWT_SECRET_KEY", cls.JWT_SECRET_KEY),
        ]
        
        missing_keys = []
        for key_name, key_value in required_keys:
            if not key_value or key_value.strip() == "":
                missing_keys.append(key_name)
        
        if missing_keys:
            raise ValueError(f"필수 환경변수가 설정되지 않았습니다: {', '.join(missing_keys)}")
        
        return True
    
    @classmethod
    def get_device(cls) -> str:
        """디바이스 설정 자동 감지"""
        if cls.DEVICE.lower() == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return cls.DEVICE.lower()
    
    @classmethod
    def is_production(cls) -> bool:
        """운영 환경 여부 확인"""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """개발 환경 여부 확인"""
        return cls.ENVIRONMENT.lower() == "development"
    
    @classmethod
    def get_cors_origins(cls) -> list:
        """CORS 허용 도메인 리스트 반환"""
        if cls.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in cls.ALLOWED_ORIGINS.split(",")]
    
    @classmethod
    def get_filter_keywords(cls) -> list[str]:
        """필터 키워드 리스트 반환"""
        return [kw.strip() for kw in cls.FILTER_KEYWORDS.split(",") if kw.strip()]
    
    @classmethod
    def get_exclude_keywords(cls) -> list[str]:
        """제외 키워드 리스트 반환"""
        return [kw.strip() for kw in cls.EXCLUDE_KEYWORDS.split(",") if kw.strip()]
    
    @classmethod
    def get_site_normalization(cls) -> dict:
        """사이트별 정규화 설정 반환"""
        try:
            import json
            return json.loads(cls.SITE_NORMALIZATION_CONFIG)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ 사이트 정규화 설정 파싱 오류: {e}")
            # 기본값 반환
            return {
                "arca": {"views_base": 5000, "likes_base": 20, "likes_ratio_range": [0.003, 0.015]},
                "dcinside": {"views_base": 15000, "likes_base": 30, "likes_ratio_range": [0.0015, 0.008]},
                "official": {"views_base": 120000, "likes_base": 50, "likes_ratio_range": [0.0002, 0.002]}
            }
    
    @classmethod
    def get_crawler_headers(cls) -> dict:
        """크롤러용 HTTP 헤더 반환"""
        return {"User-Agent": cls.CRAWLER_USER_AGENT}
    
    @classmethod
    def create_embedding_function(cls):
        """임베딩 타입에 따라 적절한 임베딩 함수 생성"""
        if cls.EMBEDDING_TYPE.lower() == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=cls.EMBED_MODEL_NAME,
                google_api_key=cls.GEMINI_API_KEY
            )
        elif cls.EMBEDDING_TYPE.lower() == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            device = cls.get_device()
            return HuggingFaceEmbeddings(
                model_name=cls.EMBED_MODEL_NAME,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(f"지원하지 않는 임베딩 타입: {cls.EMBEDDING_TYPE}")
    
    @classmethod
    def create_directories(cls):
        directories = [
            cls.LOG_DIR,
            cls.CACHE_DIR,
            cls.VECTOR_DB_DIR,
            Path(cls.VISITED_URLS_PATH).parent,
            Path(cls.PROCESSED_SAVE_PATH).parent,
            Path(cls.PROCESSED_CACHE_PATH).parent,
            cls.MERGED_DIR,
            cls.RAW_DATA_DIR,
        ]
        for d in directories:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config_summary(cls):
        """설정 요약 정보 출력 (민감 정보 제외)"""
        print("="*50)
        print("📋 6-AI 프로젝트 설정 정보")
        print("="*50)
        print(f"🏗️  환경: {cls.ENVIRONMENT}")
        print(f"🌐 포트: {cls.PORT}")
        print(f"📊 로그 레벨: {cls.LOG_LEVEL}")
        print(f"🤖 LLM 모델: {cls.LLM_MODEL_NAME}")
        print(f"🧠 임베딩 모델: {cls.EMBED_MODEL_NAME} ({cls.EMBEDDING_TYPE})")
        print(f"🔍 웹 그라운딩: {'ON' if cls.ENABLE_WEB_GROUNDING else 'OFF'}")
        print(f"💻 디바이스: {cls.get_device()}")
        print(f"📁 벡터 DB: {cls.VECTOR_DB_DIR}")
        print("="*50)


# 설정 인스턴스 (싱글톤 패턴)
config = Config()

# 필수 키 검증 (모듈 임포트 시 자동 실행)
try:
    config.validate_required_keys()
    config.create_directories()
except ValueError as e:
    print(f"❌ 설정 오류: {e}")
    print("💡 .env 파일을 확인하고 필요한 환경변수를 설정해주세요.")
