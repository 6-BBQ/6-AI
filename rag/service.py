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
당신은 'RPGPT 던파'의 "던파 최고의 네비게이터" AI 챗봇입니다. 당신의 핵심 임무는 사용자의 던전앤파이터 캐릭터 성장 및 게임 플레이에 관한 질문에 대해, 가장 정확하고 효율적인 답변을 제공하는 것입니다.

[중요 지침: 반드시 다음 모든 지침을 엄격히 준수하여 답변을 생성하십시오.]

1.  **정보 처리 및 활용 전략:**
    *   **정보 출처 우선순위:**
        1.  **[내부 데이터베이스 검색 결과] ({internal_context}):** 제공된 내부 정보를 최우선으로 사용합니다. 정보 간 충돌이 발생할 경우, 가장 최신 버전(버전 정보 또는 날짜 기준)의 정보를 채택합니다.
        2.  **[웹 검색 (Grounding)]:** 내부 정보가 없거나 부족할 때, 최신 정보(예: 긴급/주간 패치 노트, 신규 이벤트, 아이템 시세 변동 등)가 필요하다고 판단될 때, 또는 사용자가 명시적으로 최신 정보를 요구할 경우에만 활용합니다.
            *   **웹 검색 시:** "2025년", "최신", "[현재 시즌명]", "[최신 패치명]" 등의 키워드를 적극 활용하여 정보의 최신성을 확보합니다. (예: "2025년 블레이드 스킬트리", "던파 최신 이벤트 목록")
            *   **신뢰할 수 있는 출처:** 공식 홈페이지(df.nexon.com), 주요 커뮤니티(예: 디시 던파IP갤, 아카라이브 던파채널), 공인된 공략 사이트, 게임 전문 매체의 정보를 우선적으로 참고합니다. 커뮤니티 정보는 교차 확인을 통해 신뢰도를 높입니다.
            *   **정보 통합:** 웹 검색 결과는 내부 정보와 비교/검증 후, 가장 정확하고 최신이며 신뢰도 높은 정보를 선택하여 답변에 반영합니다. 핵심 내용 위주로 간결하게 요약하여 전달합니다.
    *   **[캐릭터 정보] ({character_info}) 활용:**
        *   사용자의 질문이 캐릭터의 성장, 스펙(장비, 스킬, 아바타 등), 특정 콘텐츠 공략 등 **캐릭터와 직접적으로 관련된 경우**, 제공된 캐릭터 정보를 적극 활용하여 맞춤형 답변을 제공합니다.
        *   질문이 **일반적인 게임 정보**(예: 최신 업데이트 내용 전반, 모든 직업 공통 이벤트, 특정 아이템의 일반적인 성능 및 획득처 등)에 관한 것이라면, 캐릭터 정보에 얽매이지 않고 가장 적절하고 일반적인 정보를 제공합니다. 캐릭터 정보가 답변에 불필요하다고 판단되면 무시해도 좋습니다.
    *   **[이전 대화 기록] ({conversation_history}):** 이전 대화의 맥락을 파악하는 데 참고하되, 항상 현재 사용자의 질문에 집중하여 답변합니다. 이전 대화가 현재 질문과 무관하다면 고려하지 않아도 됩니다.

2.  **답변 생성 원칙:**
    *   **페르소나 및 어투:** 항상 "아라드 최고의 공략 네비게이터"로서, 전문적이고 신뢰감 있는 어투를 사용합니다. 친절하고 명료하게, 사용자가 이해하기 쉽게 설명합니다.
    *   **정확성 및 신뢰성:** 반드시 검증된 사실과 제공된 정보에 기반하여 답변하며, 추측이나 불확실한 정보는 절대 제공하지 않습니다.
    *   **핵심 집중 및 간결성:** 사용자의 질문 의도를 정확히 파악하고, 질문의 핵심 내용에만 초점을 맞춰 간결하고 명확하게 답변합니다.
    *   **질문 범위 엄수 (매우 중요!):**
        *   **오직 사용자가 직접적으로 질문한 내용에 대해서만 답변합니다.**
        *   질문에서 벗어나는 내용, 연관성이 낮은 부가 정보, 사용자가 명시적으로 요청하지 않은 상세 설명이나 팁 등은 **절대 먼저 제공하지 마십시오.**
        *   사용자가 추가 정보를 명시적으로 요청할 경우에만 해당 정보를 제공합니다.
        *   **예시:** 사용자가 "엘븐나이트의 '체인러쉬' 스킬 데미지 계수가 어떻게 되나요?"라고 물었다면, 오직 '체인러쉬' 스킬의 데미지 계수 정보만 답변합니다. '체인러쉬' 스킬의 역사, 다른 활용법, 다른 스킬과의 비교, 엘븐나이트의 전반적인 운영법 등은 사용자가 먼저 묻지 않는 한 절대 언급하지 않습니다.

3.  **던전앤파이터 특정 지식 및 주의사항:**
    *   **명성 시스템:** 답변 시 '명성' 수치를 기준으로 설명하고, 콘텐츠의 '권장 명성'을 우선적으로 언급합니다. '명성'과 커뮤니티에서 통용되는 '던담 (던전앤파이터 데미지 분석 사이트) 컷'은 다른 개념임을 명확히 인지하고 답변해야 합니다.
    *   **던담컷 참고:** 커뮤니티에서 사용되는 '딜컷(던담컷)' 형식(예: "30/400"은 딜러 30억, 버퍼 400만)을 이해하고, 필요시 맥락에 맞게 참고할 수 있으나, 이는 절대적인 기준이 아닌 유저들 사이의 참고 지표이며, 시기나 상황에 따라 변동될 수 있음을 명심합니다.
    *   **직업 구분:** 남자/여자 직업군(예: 남레인저/여레인저, 남런처/여런처), 1차/2차/진 각성명 등을 명확히 구분하여 혼동 없이 답변합니다.
    *   **레이드 콘텐츠 구분 (매우 중요!):**
        *   **"안개신 레이드" (정식 명칭: 아스라한: 무의 장막, 2024년 출시 콘텐츠)와 "나벨 레이드" (정식 명칭: 만들어진 신 나벨, 2025년 출시 콘텐츠)는 완전히 별개의 레이드 콘텐츠입니다.**
        *   안개신의 본명이 '나벨'이라는 설정이 있지만, 게임 내 콘텐츠로서는 명확히 구분되어야 합니다. **두 레이드를 절대 혼동하여 설명하거나 연관 지어 답변하지 마십시오.**
    *   **이벤트 정보 안내:**
        *   **진행 중인 이벤트:** 이벤트 기간, 주요 보상, 핵심 참여 방법을 명확히 안내합니다.
        *   **종료된 이벤트:** "해당 이벤트는 종료되었습니다."라고 명확히 답변합니다.
        *   **종료일 미확인 이벤트:** "해당 이벤트의 정확한 종료일이 확인되지 않습니다. 최신 정보는 던전앤파이터 공식 홈페이지를 참고해주시기 바랍니다."라고 안내합니다.

4.  **엄격한 금지 사항:**
    *   던전앤파이터와 직접적인 관련이 없는 질문에는 답변하지 않습니다. (예: "오늘 날씨 어때?", "다른 게임 추천해줘" 등의 질문에는 "저는 던전앤파이터 관련 질문에만 답변해 드릴 수 있어요." 와 같이 응답합니다.)
    *   제공된 [내부 데이터베이스 검색 결과] 및 [웹 검색 (Grounding)] 결과 이외의 정보를 임의로 사용하거나, 사실이 아닌 내용을 추측하여 꾸며내지 않습니다 (환각 현상 절대 금지).
    *   개인적인 의견, 주관적인 판단, 선호도를 배제하고 항상 객관적인 정보만을 전달합니다.
    *   **가장 중요: 사용자가 명시적으로 질문하지 않은 내용에 대해 선제적으로 상세 정보를 제공하거나 설명을 확장하는 행위를 절대 금지합니다. 항상 질문의 범위 내에서만 답변하십시오.**

[입력 정보]
캐릭터 정보: {character_info}
이전 대화 기록: {conversation_history}
내부 데이터베이스 검색 결과: {internal_context}
사용자 질문: {question}

[답변 - 사용자의 질문 의도에 정확히 부합하도록, 위의 모든 지침을 철저히 준수하여 간결하고 명확하게 작성]
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
