import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from .models import (
    ChatRequest, ChatResponse, ErrorResponse, 
    HealthResponse, SourceDocument
)
from .auth import verify_jwt_token
from rag import get_rag_answer, get_rag_service

# 라우터 생성
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():

    try:
        test_result = get_rag_answer("테스트")
        rag_ready = bool(test_result and test_result.get('result'))
    except Exception:
        rag_ready = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        rag_system_ready=rag_ready
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    
    try:
        # 1. JWT 토큰 검증
        user_info = verify_jwt_token(request.jwt_token)
        print(f"[INFO] 인증된 사용자: {user_info.get('user_id')}")
        
        # 2. 캐릭터 정보 준비
        character_info = None
        if request.character_summary:
            character_info = {
                "character_id": request.character_summary.character_id,
                "character_name": request.character_summary.character_name,
                "class_name": request.character_summary.class_name,
                "fame": request.character_summary.fame
            }
        
        # 3. RAG 답변 생성
        print(f"[INFO] RAG 질문 처리: {request.query}")
        rag_result = get_rag_answer(request.query, character_info)
        
        # 4. 출처 정보 변환
        sources = []
        for doc in rag_result.get("source_documents", [])[:5]:  # 최대 5개만
            metadata = doc.metadata or {}
            sources.append(SourceDocument(
                title=metadata.get("title", "제목 없음"),
                url=metadata.get("url", ""),
                source=metadata.get("source", "")
            ))
        
        # 5. 응답 생성
        response = ChatResponse(
            success=True,
            answer=rag_result["result"],
            sources=sources,
            execution_time=rag_result["execution_times"]["total"],
            used_web_search=rag_result.get("used_web_search", False)
        )
        
        print(f"[INFO] RAG 처리 완료: {response.execution_time:.2f}초")
        return response
        
    except HTTPException:
        # JWT 인증 에러는 그대로 전파
        raise
        
    except Exception as e:
        print(f"[ERROR] RAG 처리 중 오류: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG 처리 중 오류가 발생했습니다: {str(e)}"
        )