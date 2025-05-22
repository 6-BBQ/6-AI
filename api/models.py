from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class CharacterInfo(BaseModel):
    """캐릭터 정보 모델"""
    character_id: str = Field(..., description="캐릭터 ID")
    character_name: Optional[str] = Field(None, description="캐릭터 이름")
    class_name: Optional[str] = Field(None, description="직업명")
    fame: Optional[int] = Field(None, description="명성")

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    query: str = Field(..., description="사용자 질문", min_length=1, max_length=500)
    jwt_token: str = Field(..., description="JWT 인증 토큰")
    character_summary: Optional[CharacterInfo] = Field(None, description="캐릭터 요약 정보")

class SourceDocument(BaseModel):
    """출처 문서 모델"""
    title: str = Field(..., description="문서 제목")
    url: Optional[str] = Field(None, description="문서 URL")
    source: Optional[str] = Field(None, description="소스 타입")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    answer: str = Field(..., description="RAG 답변 내용")
    sources: List[SourceDocument] = Field(default_factory=list, description="참고 출처들")
    character_specific_advice: Optional[str] = Field(None, description="캐릭터 맞춤 조언")
    execution_time: float = Field(..., description="실행 시간(초)")
    used_web_search: bool = Field(default=False, description="웹 검색 사용 여부")

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    success: bool = Field(default=False, description="성공 여부")
    error: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")

class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    timestamp: str = Field(..., description="현재 시간")
    rag_system_ready: bool = Field(..., description="RAG 시스템 준비 상태")
