import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import json

from .models import (
    ChatRequest, ChatResponse, ErrorResponse, 
    HealthResponse, SourceDocument
)
from .auth import verify_jwt_token
from rag import get_structured_rag_answer, get_structured_rag_service

# 라우터 생성
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest): # ChatRequest 모델 사용
    
    # 1. JWT 토큰 검증
    try:
        verify_jwt_token(request.jwtToken)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"JWT 인증 실패: {str(e)}"
        )

    def transform_character_data(raw_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        원본 캐릭터 JSON 데이터를 RAG 서비스 및 프롬프트에 사용하기 적합한 형태로 변환합니다.
        """
        if not raw_data:
            return None

        transformed = {}

        # 1. job: jobGrowName과 jobName 조합
        job_grow_name = raw_data.get("jobGrowName", "")
        job_name = raw_data.get("jobName", "")
        if job_grow_name and job_name:
            processed_grow_name = job_grow_name.replace("眞 ", "")
            transformed["job"] = f"{processed_grow_name}({job_name.split('(')[-1]}" if '(' in job_name else f"{processed_grow_name}({job_name})"

        # 2. fame
        if "fame" in raw_data:
            transformed["fame"] = raw_data.get("fame")

        # 3. weapon: weaponEquip의 itemRarity 사용
        weapon_equip = raw_data.get("weaponEquip")
        if isinstance(weapon_equip, dict) and "itemRarity" in weapon_equip:
            transformed["weapon"] = f"{weapon_equip['itemRarity']} 무기"

        # 4. epicNum (에픽 개수)
        if "epicNum" in raw_data:
            transformed["epicNum"] = raw_data.get("epicNum")

        # 5. originalityNum (태초 개수)
        if "originalityNum" in raw_data:
            transformed["originalityNum"] = raw_data.get("originalityNum")

        # 6. title: titleName 사용
        if "titleName" in raw_data:
            transformed["title"] = raw_data.get("titleName")

        # 7. setItemName & 8. setItemRarityName
        set_item_info_ai = raw_data.get("setItemInfoAI")
        if isinstance(set_item_info_ai, list) and len(set_item_info_ai) > 0:
            first_set_item = set_item_info_ai[0]
            if isinstance(first_set_item, dict):
                if "setItemName" in first_set_item:
                    transformed["set_item_name"] = first_set_item.get("setItemName")
                if "setItemRarityName" in first_set_item:
                    transformed["set_item_rarity"] = first_set_item.get("setItemRarityName")
        
        # 9. creature: creatureName 사용
        if "creatureName" in raw_data:
            transformed["creature"] = raw_data.get("creatureName")

        # 10. aura: auraName 사용
        if "auraName" in raw_data:
            transformed["aura"] = raw_data.get("auraName")
            
        return transformed if transformed else None

    try:
        # 1) 캐릭터 정보 변환
        # request.characterData가 ChatRequest 모델에 정의되어 있다고 가정
        transformed_char_info = transform_character_data(request.characterData)
        
        if transformed_char_info:
            print(f"[INFO] 변환된 캐릭터 정보: {transformed_char_info}")
        else:
            print("[INFO] 캐릭터 정보 없음 또는 변환 실패")

        # 2) 이전 대화 기록 처리
        conversation_history = []
        if request.beforeQuestionList and request.beforeResponseList:
            # 두 리스트의 길이가 다를 수 있으므로 최소 길이로 맞춤
            min_length = min(len(request.beforeQuestionList), len(request.beforeResponseList))
            for i in range(min_length):
                conversation_history.extend([
                    {"role": "user", "content": request.beforeQuestionList[i]},
                    {"role": "assistant", "content": request.beforeResponseList[i]}
                ])
            print(f"[INFO] 이전 대화 기록: {len(conversation_history)//2}개 대화")
        else:
            print("[INFO] 이전 대화 기록 없음")

        # 3) RAG 호출 시, 변환된 character_info 전달
        print(f"[INFO] RAG 질문 처리: {request.query}")
        rag_result = get_structured_rag_answer(
            request.query,
            character_info=transformed_char_info # 변환된 딕셔너리 전달
        )
        
        # 3. 출처 정보 변환
        sources = []
        # source_documents가 없을 경우 빈 리스트로 처리 (rag_result.get의 기본값 활용)
        for doc in rag_result.get("source_documents", []):
            metadata = doc.metadata or {}
            sources.append(SourceDocument(
                title=metadata.get("title", "제목 없음"),
                url=metadata.get("url"), # None일 수 있음
                source=metadata.get("source") # None일 수 있음
            ))
        
        # 4. 디버깅용 문서 변환 함수 (기존과 동일)
        def convert_docs_to_dict(docs_list: Optional[List[Any]]): # 타입 힌트 명확화
            if not docs_list:
                return []
            result = []
            for doc in docs_list:
                # Document 객체인지 확인 (langchain.docstore.document.Document)
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    result.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata or {}
                    })
                elif isinstance(doc, dict): # 이미 딕셔너리 형태일 경우
                    result.append(doc)
            return result
        
        # 5. 응답 생성
        response = ChatResponse(
            success=True,
            answer=rag_result.get("result", "답변을 생성하지 못했습니다."), # result 키가 없을 경우 대비
            sources=sources,
            execution_time=rag_result.get("execution_times", {}).get("total", 0.0),
            used_web_search=rag_result.get("used_web_search", False),
            internal_docs=convert_docs_to_dict(rag_result.get("internal_docs")),
            web_docs=convert_docs_to_dict(rag_result.get("web_docs")),
            enhanced_query=rag_result.get("enhanced_query"),
            execution_times=rag_result.get("execution_times"),
            internal_context=rag_result.get("internal_context_provided_to_llm"), # RAGService 반환값 키 확인 필요
            web_context=rag_result.get("web_context_provided_to_llm") # RAGService 반환값 키 확인 필요
        )
        
        print(f"[INFO] RAG 처리 완료: {response.execution_time:.2f}초")
        return response
        
    except HTTPException:
        # JWT 인증 에러 등 FastAPI의 HTTPException은 그대로 전파
        raise
        
    except Exception as e:
        import traceback # 상세한 에러 로깅을 위해
        print(f"[ERROR] RAG 처리 중 예기치 않은 오류: {str(e)}")
        print(traceback.format_exc()) # 스택 트레이스 출력
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG 처리 중 오류가 발생했습니다: {str(e)}"
        )