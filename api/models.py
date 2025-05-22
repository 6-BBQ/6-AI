from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SourceDocument(BaseModel):
    title: str = Field(..., description="문서 제목")
    url: Optional[str] = Field(None, description="문서 URL")
    source: Optional[str] = Field(None, description="소스 타입")

# ────────────── 실행 시간 모델 ──────────────
class SearchTimes(BaseModel):
    internal_search: float = Field(..., description="내부 검색 시간")
    web_search: float = Field(..., description="웹 검색 시간")
    total_search: float = Field(..., description="검색 총 소요 시간")


class ExecutionTimes(BaseModel):
    total: float = Field(..., description="전체 소요 시간")
    llm: float = Field(..., description="LLM 응답 생성 시간")
    search: SearchTimes = Field(..., description="검색 세부 소요 시간")


# ────────────── 요청·응답 모델 ──────────────
class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    jwt_token: str = Field(..., description="JWT 인증 토큰")
    # character_data는 어떤 필드가 와도 수용
    character_data: Optional[Dict[str, Any]] = Field(
        None, description="스프링에서 내려주는 캐릭터 요약 정보(JSON)"
    )

class ChatResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    answer: str = Field(..., description="RAG 답변 내용")
    sources: List[SourceDocument] = Field(default_factory=list, description="참고 출처들")

    character_specific_advice: Optional[str] = Field(None, description="캐릭터 맞춤 조언")
    execution_time: float = Field(..., description="총 실행 시간(초)")
    used_web_search: bool = Field(default=False, description="웹 검색 사용 여부")

    # 디버깅 정보
    internal_docs: List[Dict[str, Any]] = Field(default_factory=list)
    web_docs: List[Dict[str, Any]] = Field(default_factory=list)
    enhanced_query: Optional[str] = None

    execution_times: Optional[Dict[str, Any]] = Field(None, description="상세 실행 시간")
    internal_context: Optional[str] = None
    web_context: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: str
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    rag_system_ready: bool