"""
분리된 RAG 서비스 - 구조화된 버전
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

# LLM & 임베딩
from langchain_chroma import Chroma

# Google Gemini SDK for grounding
from google import genai

# 검색 관련
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate

# 분리된 유틸리티들
from .cache_utils import CacheManager
from .text_utils import TextProcessor
from .retrievers import MetadataAwareRetriever
from .search_factory import SearcherFactory
from utils import get_logger
from config import config  # 중앙화된 설정 사용


class StructuredRAGService:
    """구조화된 RAG 서비스 클래스"""

    def __init__(self):
        """RAG 서비스 초기화"""
        self.logger = get_logger(__name__)
        self.logger.info("=== RAG 서비스 초기화 시작 ===")
        
        # 설정값들을 config에서 가져오기
        self.cache_dir = Path(config.CACHE_DIR)
        self.vector_db_dir = config.VECTOR_DB_DIR
        self.embed_model_name = config.EMBED_MODEL_NAME
        self.cross_encoder_model_hf = config.CROSS_ENCODER_MODEL
        self.llm_model_name = config.LLM_MODEL_NAME
        self.enable_web_grounding = config.ENABLE_WEB_GROUNDING
        self.cache_expiry_short = config.CACHE_EXPIRY_SHORT
        self.cache_expiry_long = config.CACHE_EXPIRY_LONG
        
        # 캐시 파일명들
        self.bm25_cache_file = "bm25_retriever.pkl"
        self.cross_encoder_cache_file = "cross_encoder.pkl"
        
        start_time = time.time()
        
        self._setup_environment()
        self._initialize_utilities()
        self._initialize_core_components()
        self._initialize_retrievers()
        self._setup_llm_and_prompt()
        
        total_time = time.time() - start_time
        self.logger.info(f"=== RAG 서비스 초기화 완료 ({total_time:.2f}초) ===")

    def _setup_environment(self):
        """환경 설정"""
        self.logger.debug("환경 설정 시작")
        
        self.cache_dir.mkdir(exist_ok=True)
        self.logger.debug(f"캐시 디렉토리 설정: {self.cache_dir}")
        
        # API 키 설정
        self.gemini_api_key = config.GEMINI_API_KEY
        
        if not self.gemini_api_key:
            self.logger.error("GEMINI_API_KEY 환경변수가 설정되지 않았습니다")
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다!")
        
        self.logger.info("✅ API 키 확인 완료 - Gemini LLM + 임베딩 사용")

    def _initialize_utilities(self):
        """유틸리티 클래스들 초기화"""
        self.logger.debug("유틸리티 초기화 시작")
        
        self.cache_manager = CacheManager(
            self.cache_dir, 
            self.cache_expiry_short, 
            self.cache_expiry_long
        )
        self.text_processor = TextProcessor()
        self.search_factory = SearcherFactory()
        
        self.logger.debug("유틸리티 초기화 완료")

    def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        self.logger.info("🚀 RAG 시스템 핵심 컴포넌트 초기화 중...")
        
        # Grounding을 위한 Google SDK 초기화
        self.logger.info("Google GenAI SDK 사용 - 웹 검색 그라운딩 지원")
        try:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
            self.logger.debug("Google GenAI 클라이언트 초기화 성공")
        except Exception as e:
            self.logger.error(f"Google GenAI 클라이언트 초기화 실패: {e}")
            raise
        
        # 그라운딩 활성화 여부 설정
        self.logger.info(f"웹 검색 그라운딩: {'ON' if self.enable_web_grounding else 'OFF'}")
        
        # 임베딩 함수 초기화 (config 설정에 따라 동적 생성)
        embedding_type = config.EMBEDDING_TYPE
        self.logger.info(f"임베딩 모델 로드: {self.embed_model_name} (타입: {embedding_type})")
        
        try:
            self.embedding_fn = config.create_embedding_function()
            self.logger.debug(f"임베딩 모델 로드 성공 ({embedding_type})")
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
        
        # 벡터 DB 초기화
        try:
            self.vectordb = Chroma(
                persist_directory=self.vector_db_dir,
                embedding_function=self.embedding_fn
            )
            self.logger.info(f"벡터 DB 연결 성공: {self.vector_db_dir}")
        except Exception as e:
            self.logger.error(f"벡터 DB 연결 실패: {e}")
            raise

        self.logger.info(f"LLM 모델 설정: {self.llm_model_name}")
        
        self.logger.info("✅ 핵심 컴포넌트 초기화 완료")

    def _initialize_retrievers(self):
        """검색기 초기화"""
        self.logger.info("🔄 검색기 초기화 중...")
        start_time = time.time()
        
        # 벡터 검색기 설정 (검색 개수 대폭 증가)
        self.vector_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 40, "fetch_k": 120, "lambda_mult": 0.5},
        )
        self.logger.debug("벡터 검색기 설정 완료")
        
        # BM25 검색기 생성 (캐시 사용)
        self.bm25_retriever = self._get_bm25_retriever()
        self.logger.debug("BM25 검색기 로드 완료")
        
        # 앙상블 검색기 생성 - 기본 설정
        # 동적 가중치는 rag_search에서 처리
        self.rrf_retriever = None  # 나중에 동적으로 생성
        
        # CrossEncoder 모델만 미리 로드
        self.cross_encoder_model = self._get_cross_encoder_model()
        self.logger.debug("CrossEncoder 모델 로드 완료")
        
        # internal_retriever는 rag_search에서 동적으로 생성
        self.internal_retriever = None
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"🎉 검색기 초기화 완료! (소요시간: {elapsed_time:.2f}초)")

    def _setup_llm_and_prompt(self):
        """LLM 및 프롬프트 설정 (던파 전문가 버전)"""
        self.prompt = PromptTemplate(
            input_variables=["internal_context", "question", "character_info", "conversation_history"],
            template="""
당신은 'rpgpt 던파'의 "던파 최고의 네비게이터" AI 챗봇입니다. 사용자의 던전앤파이터 캐릭터 성장 및 게임 플레이에 필요한 질문에 대해, 정확하고 효율적인 답변을 제공하는 것을 최우선 목표로 합니다.

※ 반드시 아래 제공된 정보와 지침에 따라서만 답변해야 합니다.

[캐릭터 정보]
{character_info}

[이전 대화 기록]
{conversation_history}

[내부 데이터베이스]
{internal_context}

[정보 활용 전략 및 우선순위]
1.  **내부 데이터베이스 ({internal_context}) 우선 활용:** 최신 정보를 우선 사용하고, 충돌 시 최신 정보만 채택 (버전/날짜 기준).
2.  **웹 검색 (Web Grounding) 활용:**
    *   **활용 조건:** 내부 정보 부재/부족, 최신 업데이트 필요 시 (패치, 이벤트, 시세 등), 사용자 명시적 최신 정보 요구 시.
    *   **검색 지침:**
        *   **최신성 확보:** 검색 시 "2025년" 이후 날짜, 현재 시즌명, 최신 패치명을 키워드에 포함 (예: "2025 [직업] 스킬트리", "최신 던파 정보").
        *   **출처 및 검증:** 공식 웹사이트, 주요 커뮤니티/공략 사이트, 게임 매체를 참고. 객관적이고 검증된 정보를 선택하며, 커뮤니티 정보는 교차 확인으로 신뢰도 확보.
    *   **정보 통합:** 웹 검색 결과는 내부 정보와 교차 검증 후, 최신의 정확하고 신뢰성 높은 정보를 사용. 핵심만 요약 전달.
3.  **정보 부재 시 처리:** 내부/웹 검색으로도 정보 확인 불가 시, "요청하신 정보 확인이 어렵습니다. 질문을 더 구체화하거나 다른 질문을 해주세요."로 안내.

[답변 생성 규칙]
1.  **페르소나 및 어투:** 항상 'RPGPT 던파'의 "아라드 최고의 공략 네비게이터"로서 전문적이고 신뢰감 있는 어투를 사용하며, 친절하고 명료하게, 사용자 이해도를 고려하여 설명합니다.
2.  **답변 범위 및 형식 (매우 중요):**
    *   **질문 의도 명확히 파악:** 사용자가 무엇을 궁금해하는지 정확히 이해하고, 해당 질문에만 집중하여 답변합니다.
    *   **범위 엄수:** 사용자의 질문에서 직접적으로 묻지 않은 내용이나 연관성이 낮은 부가 정보는 **절대 먼저 제공하지 않습니다.** 사용자가 추가 정보를 명시적으로 요청할 경우에만 해당 정보를 제공합니다.
    *   **핵심 위주 전달:** 답변은 질문의 핵심 내용만을 간결하고 명확하게 전달합니다. 필요시 단계별/순서대로 설명할 수 있지만, 이 역시 질문의 범위를 벗어나지 않도록 주의합니다.
    *   **예시:** 사용자가 "A 스킬의 데미지"를 물었다면, A 스킬의 데미지만 답하고, B 스킬이나 A 스킬의 역사, 다른 활용법 등은 먼저 언급하지 않습니다.
3.  **콘텐츠 관련 답변:**
    *   '명성' 기준 답변, '권장 명성' 우선.
    *   '명성'과 '던담컷'은 별개임을 인지.
    *   커뮤니티 '던담컷' 형식(예: "30억/400만" - 딜러 데미지/버퍼 버프력)을 이해하고 맥락에 맞춰 참고. 이는 명성과 다른 보조 지표이며, 수치는 맥락에 따라 달라질 수 있음.
    *   남자/여자 직업, 전직 등 명확히 구분.
    *   안개신 레이드와 나벨 레이드는 별개임을 인지.
4.  **이벤트 안내 기준:** 종료 시 "종료됨". 종료일 미확인 시 "종료일 미확인, 공식 홈페이지 확인 요망". 진행 중이면 기간/보상/참여 방법 안내.

[제한 사항 및 금지 사항]
*   던파 외 질문 거부.
*   제공된 정보 외 사용 및 환각 금지.
*   추측/불확실 정보 기반 답변 금지.
*   주관적 의견/판단 배제.
*   **사용자가 질문하지 않은 내용에 대해 선제적으로 상세 정보를 제공하는 것을 금지합니다.**

[사용자 질문]
{question}

[답변 - 사용자의 질문에만 초점을 맞춰 간결하고 명확하게]
"""
        )
        self.logger.debug("✅ LLM 프롬프트 설정 완료")

    def _get_bm25_retriever(self):
        """BM25 검색기 생성 (캐시 활용)"""
        def creation_func():
            docs_for_bm25 = self.search_factory.create_bm25_data_from_vectordb(self.vectordb)
            return self.search_factory.create_bm25_retriever(docs_for_bm25)
        
        return self.cache_manager.load_or_create_cached_item(
            self.bm25_cache_file, creation_func, self.cache_expiry_short, "BM25 Retriever"
        )

    def _get_cross_encoder_model(self):
        """CrossEncoder 모델 생성 (캐시 활용)"""
        def creation_func():
            return self.search_factory.create_cross_encoder_model(self.cross_encoder_model_hf)
        
        return self.cache_manager.load_or_create_cached_item(
            self.cross_encoder_cache_file, creation_func, self.cache_expiry_long, "CrossEncoder 모델"
        )
    
    def _determine_weights(self, query: str, character_info: Optional[Dict]) -> List[float]:
        """쿼리와 캐릭터 정보를 기반으로 앙상블 가중치 결정"""
        query_lower = query.lower()

        # 기본값: BM25 우선 (직업도, 캐릭터 정보도 이미 갖고 있음)
        weights = [0.3, 0.7]

        # “최신·업데이트” 류 키워리면 벡터 가중치로 스왑
        if any(k in query_lower for k in ["최신", "업데이트", "현재", "패치", "종결"]):
            weights = [0.7, 0.3]
            self.logger.debug("🔄 최신·패치 관련 키워드 감지 → 벡터 가중치 증가")

        return weights
    
    def _build_conversation_context_for_llm(self, conversation_history: Optional[List[Dict]]) -> str:
        """이전 대화 기록을 LLM용 컨텍스트 문자열로 변환"""
        if not conversation_history or len(conversation_history) == 0:
            return "이전 대화 기록이 없습니다."
        
        context_parts = []
        for i, message in enumerate(conversation_history, 1):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            if role == 'user':
                context_parts.append(f"사용자 질문 {i//2 + 1}: {content}")
            elif role == 'assistant':
                context_parts.append(f"이전 답변 {i//2 + 1}: {content}")
        
        return "\n".join(context_parts) if context_parts else "이전 대화 기록이 없습니다."

    def rag_search(self, query: str, character_info: Optional[Dict]) -> Dict[str, Any]:
        # 캐시 확인
        cached_result = self.cache_manager.get_cached_search_result(query, 'rag_search', character_info)
        if cached_result:
            self.logger.debug("🔄 캐시된 RAG 검색 결과 사용")
            return cached_result

        search_start_time = time.time()
        # 캐릭터 정보로 쿼리 강화
        enhanced_query = self.text_processor.enhance_query_with_character(query, character_info)
        times = {"internal_search": 0.0}
        
        # 동적 가중치 설정
        weights = self._determine_weights(query, character_info)
        self.logger.debug(f"🎯 앙상블 가중치: 벡터={weights[0]:.2f}, BM25={weights[1]:.2f}")
        
        # 앙상블 검색기 동적 생성
        self.rrf_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=weights,
        )
        
        # CrossEncoder 재랭킹 추가
        compressor = CrossEncoderReranker(model=self.cross_encoder_model, top_n=60)
        base_retriever = ContextualCompressionRetriever(
            base_retriever=self.rrf_retriever,
            base_compressor=compressor,
        )
        
        # 메타데이터 인식 검색기로 래핑
        self.internal_retriever = MetadataAwareRetriever(base_retriever)

        def _search_internal():
            start = time.time()
            try:
                self.logger.debug("🔄 내부 RAG 검색 시작...")
                docs = self.internal_retriever.get_relevant_documents(enhanced_query)
                times["internal_search"] = time.time() - start
                self.logger.info(f"✅ 내부 RAG 검색 완료: {times['internal_search']:.2f}초, {len(docs)}개 문서")
                return docs
            except Exception as e:
                times["internal_search"] = time.time() - start
                self.logger.error(f"❌ 내부 RAG 검색 오류 ({times['internal_search']:.2f}초): {e}")
                return []

        internal_docs = _search_internal()
        
        times["internal_search"] = time.time() - search_start_time
        self.logger.debug(f"🎯 내부 검색 완료 - 총 {times['internal_search']:.2f}초")

        # 검색 결과를 컨텍스트 문자열로 변환
        internal_context_str = self.text_processor.format_docs_to_context_string(internal_docs, "내부")
        
        # 결과 구성
        result = {
            "internal_docs": internal_docs,
            "internal_context_provided_to_llm": internal_context_str,
            "enhanced_query": enhanced_query,
            "search_times": times
        }
        
        # 캐시에 저장
        self.cache_manager.save_search_result_to_cache(query, result, 'rag_search', character_info)
        return result

    def get_answer(self, query: str, character_info: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """RAG 답변 생성 (메인 API)"""
        total_start_time = time.time()
        
        self.logger.info(f"질문 처리 시작: \"{query}\"")
        
        # 이전 대화 기록 로그 출력
        if conversation_history and len(conversation_history) > 0:
            self.logger.info(f"이전 대화 기록: {len(conversation_history)}개 메시지")
        else:
            self.logger.info("이전 대화 기록 없음")

        # 캐릭터 정보를 LLM용 컨텍스트로 변환
        char_context_for_llm = self.text_processor.build_character_context_for_llm(character_info)
        
        # 이전 대화 기록을 LLM용 컨텍스트로 변환
        conversation_context_for_llm = self._build_conversation_context_for_llm(conversation_history)
        
        # 검색 수행
        search_results = self.rag_search(query, character_info)
        
        # LLM 답변 생성
        llm_start_time = time.time()
        self.logger.info("🔄 LLM 답변 생성 중...")
        
        formatted_prompt = self.prompt.format(
            internal_context=search_results["internal_context_provided_to_llm"],
            question=query,
            character_info=char_context_for_llm,
            conversation_history=conversation_context_for_llm
        )
        
        try:
            from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
            
            # 그라운딩 도구 설정
            tools = []
            if self.enable_web_grounding:
                google_search_tool = Tool(
                    google_search = GoogleSearch()
                )
                tools.append(google_search_tool)
                self.logger.debug("🔍 웹 검색 그라운딩 활성화됨")
            else:
                self.logger.debug("🚫 웹 검색 그라운딩 비활성화됨")
            
            # LLM 호출
            response = self.genai_client.models.generate_content(
                model=self.llm_model_name,
                contents=formatted_prompt,
                config=GenerateContentConfig(
                    tools=tools,
                    temperature=0,
                )
            )
            
            # 응답에서 텍스트 추출
            llm_response = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    llm_response += part.text
            
            # 그라운딩 메타데이터 확인
            if self.enable_web_grounding and hasattr(response.candidates[0], 'grounding_metadata'):
                grounding = response.candidates[0].grounding_metadata
                if hasattr(grounding, 'search_entry_point') and grounding.search_entry_point:
                    self.logger.info("🌐 웹 검색 그라운딩이 실제로 사용되었습니다!")
                    # 검색된 내용 일부 출력 (디버깅용)
                    if grounding.search_entry_point.rendered_content:
                        self.logger.debug(f"📄 검색 결과 미리보기: {grounding.search_entry_point.rendered_content[:200]}...")
        except Exception as e:
            self.logger.error(f"❌ LLM 답변 생성 오류: {e}")
            self.logger.error(f"상세 에러: {str(e)}")
            self.logger.error(f"에러 타입: {type(e).__name__}")
            llm_response = "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."

        llm_elapsed_time = time.time() - llm_start_time
        total_elapsed_time = time.time() - total_start_time
        
        self.logger.info(f"✅ LLM 답변 생성 완료 ({llm_elapsed_time:.2f}초)")
        self.logger.info(f"총 처리 시간: {total_elapsed_time:.2f}초")
        
        # 생성된 답변 출력 (디버그 레벨에서)
        self.logger.debug("\n" + "="*50)
        self.logger.debug("[답변]")
        self.logger.debug("="*50)
        self.logger.debug(llm_response[:200] + "..." if len(llm_response) > 200 else llm_response)
        self.logger.debug("="*50)
        
        # FastAPI 엔드포인트에서 기대하는 키로 반환값 구성
        return {
            "result": llm_response,
            "internal_docs": search_results["internal_docs"],
            "enhanced_query": search_results["enhanced_query"],
            "execution_times": {
                "total": total_elapsed_time,
                "llm": llm_elapsed_time,
                "search": search_results["search_times"]
            },
            "internal_context": search_results["internal_context_provided_to_llm"],
        }


# 싱글톤 인스턴스 관리
_structured_rag_service_instance: Optional[StructuredRAGService] = None

def get_structured_rag_service() -> StructuredRAGService:
    """구조화된 RAG 서비스 인스턴스 반환"""
    global _structured_rag_service_instance
    if _structured_rag_service_instance is None:
        logger = get_logger(__name__)
        logger.info("✨ 새로운 StructuredRAGService 인스턴스 생성 ✨")
        _structured_rag_service_instance = StructuredRAGService()
    return _structured_rag_service_instance

def get_structured_rag_answer(query: str, character_info: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """구조화된 RAG 답변 생성 함수"""
    service = get_structured_rag_service()
    return service.get_answer(query, character_info, conversation_history)
