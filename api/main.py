import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .endpoints import router
from .models import ErrorResponse
from utils import get_logger, log_system_info
from config import config  # 중앙화된 설정 사용


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료 이벤트를 처리하는 lifespan 함수"""
    # 로거 초기화
    logger = get_logger(__name__)
    
    logger.info("🚀 DF RAG API 서버 시작 중...")
    logger.info("📚 RAG 시스템 워밍업...")
    
    # 시스템 정보 로깅
    if config.LOG_SYSTEM_INFO:
        log_system_info(logger)
    
    # 환경 설정 로깅
    logger.info(f"실행 환경: {config.ENVIRONMENT}")
    logger.info(f"로그 레벨: {config.LOG_LEVEL}")
    logger.info(f"웹 그라운딩: {'ON' if config.ENABLE_WEB_GROUNDING else 'OFF'}")
    logger.info(f"디바이스: {config.get_device()}")
    
    try:
        from rag import get_structured_rag_service
        get_structured_rag_service()  # 싱글톤 인스턴스 생성
        logger.info("✅ RAG 시스템 준비 완료")
    except Exception as e:
        logger.error(f"❌ RAG 시스템 초기화 실패: {e}", exc_info=True)
        raise  # 초기화 실패 시 서버 시작 중단

    # 서버 실행 유지
    yield

    # 종료 시 로직
    logger.info("🛑 DF RAG API 서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="DF RAG API",
    description="던전앤파이터 전용 RAG 시스템 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # 👈 lifespan 적용
)

# CORS 설정 (config에서 가져오기)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router, prefix="/api/df", tags=["RAG"])

# 루트 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "DF RAG API Server",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

# 전역 예외 처리
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger = get_logger(__name__)
    logger.warning(f"HTTP 예외 발생 [{request.method} {request.url}]: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(
        f"예상치 못한 오류 [{request.method} {request.url}]: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="내부 서버 오류가 발생했습니다",
            error_code="INTERNAL_SERVER_ERROR"
        ).dict()
    )


# 로컬 테스트 실행
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower()
    )
