import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .endpoints import router
from .models import ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료 이벤트를 처리하는 lifespan 함수"""
    print("🚀 DF RAG API 서버 시작 중...")
    print("📚 RAG 시스템 워밍업...")

    try:
        from rag import get_rag_service
        get_rag_service()  # 싱글톤 인스턴스 생성
        print("✅ RAG 시스템 준비 완료")
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")

    # 서버 실행 유지
    yield

    # 종료 시 로직
    print("🛑 DF RAG API 서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="DF RAG API",
    description="던전앤파이터 전용 RAG 시스템 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # 👈 lifespan 적용
)

# CORS 설정 (스프링 백엔드와 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 도메인 제한 필요
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
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] 예상치 못한 오류: {str(exc)}")
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
        port=8000,
        log_level="info"
    )
