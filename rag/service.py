"""
분리된 RAG 서비스 - 구조화된 버전
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import torch

# LLM & 임베딩
from langchain_huggingface import HuggingFaceEmbeddings
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

load_dotenv()


class StructuredRAGService:
    """구조화된 RAG 서비스 클래스"""

    # --- 상수 정의 ---
    CACHE_DIR_NAME = "cache"
    VECTOR_DB_DIR = "vector_db/chroma"
    EMBED_MODEL_NAME = "dragonkue/bge-m3-ko"
    BM25_CACHE_FILE = "bm25_retriever.pkl"
    CROSS_ENCODER_CACHE_FILE = "cross_encoder.pkl"
    LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
    CROSS_ENCODER_MODEL_HF = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    # 그라운딩 활성화 설정 (환경변수로 제어 가능)
    ENABLE_WEB_GROUNDING = os.getenv("ENABLE_WEB_GROUNDING", "true").lower() == "true"

    CACHE_EXPIRY_SHORT = 60 * 60 * 12  # 12시간
    CACHE_EXPIRY_LONG = 60 * 60 * 24   # 24시간

    def __init__(self):
        """RAG 서비스 초기화"""
        self._setup_environment()
        self._initialize_utilities()
        self._initialize_core_components()
        self._initialize_retrievers()
        self._setup_llm_and_prompt()

    def _setup_environment(self):
        """환경 설정"""
        self.cache_dir = Path(self.CACHE_DIR_NAME)
        self.cache_dir.mkdir(exist_ok=True)
        
        # API 키 설정
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다!")
        
        print("✅ API 키 확인 완료 - Gemini LLM + 임베딩 사용")

    def _initialize_utilities(self):
        """유틸리티 클래스들 초기화"""
        self.cache_manager = CacheManager(self.cache_dir, self.CACHE_EXPIRY_SHORT, self.CACHE_EXPIRY_LONG)
        self.text_processor = TextProcessor()
        self.search_factory = SearcherFactory()

    def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        print("🚀 RAG 시스템 핵심 컴포넌트 초기화 중...")
        
        # Grounding을 위한 Google SDK 초기화
        print("Google GenAI SDK 사용 - 웹 검색 그라운딩 지원")
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        
        # 그라운딩 활성화 여부 설정 (True: 활성화, False: 비활성화)
        self.enable_grounding = self.ENABLE_WEB_GROUNDING
        
        # 임베딩 함수 변경 (한국어 성능 향상)
        print("✅ 임베딩 사용 - 한국어 성능 최적화")
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=self.EMBED_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # BGE 시리즈는 보통 정규화 필요
        )
        
        self.vectordb = Chroma(
            persist_directory=self.VECTOR_DB_DIR,
            embedding_function=self.embedding_fn
        )
        
        print("✅ 핵심 컴포넌트 초기화 완료")

    def _initialize_retrievers(self):
        """검색기 초기화"""
        print("🔄 검색기 초기화 중...")
        start_time = time.time()
        
        # 벡터 검색기 설정 (검색 개수 대폭 증가)
        self.vector_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 40, "fetch_k": 120, "lambda_mult": 0.5},
        )
        
        # BM25 검색기 생성 (캐시 사용)
        self.bm25_retriever = self._get_bm25_retriever()
        
        # 앙상블 검색기 생성 - 기본 설정
        # 동적 가중치는 rag_search에서 처리
        self.rrf_retriever = None  # 나중에 동적으로 생성
        
        # CrossEncoder 모델만 미리 로드
        self.cross_encoder_model = self._get_cross_encoder_model()
        
        # internal_retriever는 rag_search에서 동적으로 생성
        self.internal_retriever = None
        
        elapsed_time = time.time() - start_time
        print(f"🎉 검색기 초기화 완료! (소요시간: {elapsed_time:.2f}초)")

    def _setup_llm_and_prompt(self):
        """LLM 및 프롬프트 설정 (던파 전문가 버전)"""
        self.prompt = PromptTemplate(
            input_variables=["internal_context", "question", "character_info", "conversation_history"],
            template="""
당신은 던전앤파이터 전문 스펙업 가이드 챗봇입니다.  
※ 반드시 아래 제공된 정보만 활용해 답변하세요.

[캐릭터 정보]
{character_info}

[이전 대화 기록]
{conversation_history}

[내부 데이터베이스]
{internal_context}

[답변 규칙]
- 던파와 관련 없는 질문에는 대답을 거부하세요.
- 제공된 정보 외의 지식은 절대 사용하지 마세요.
- 정보가 부족하면 "제공된 정보에서 찾을 수 없습니다."라고 답변하세요.
- 대답에는 내부 데이터를 최대한 사용하여 답변하세요.
- 충돌하거나 중복되는 정보가 있다면 **가장 최신의 정보**만 사용하고 나머지는 무시하세요.
- 사용자의 질문 범위만 다루며, 관련 없는 설명은 생략하세요.
- 반드시 순서를 나열하며 설명하고, 간결하고 핵심적으로 답변하세요.

[콘텐츠 관련]
- 콘텐츠 관련 대답이 들어올 경우엔, 명성을 기준으로 대답하세요.
- 콘텐츠에는 입장 명성과 권장 명성이 있는데, 권장 명성 기준으로 얘기하세요.
- 남자/여자 직업은 별개의 직업입니다. 잘못되게 참조하지 마세요.

[이벤트 안내 기준]
- 종료된 이벤트 → "해당 이벤트는 종료되었습니다."
- 종료일이 없을 경우 → "이벤트 종료일을 확인해주세요."

[사용자 질문]
{question}

[답변 - 간결하고 명확하게]
"""
        )
        print("✅ LLM 프롬프트 설정 완료")

    def _get_bm25_retriever(self):
        """BM25 검색기 생성 (캐시 활용)"""
        def creation_func():
            docs_for_bm25 = self.search_factory.create_bm25_data_from_vectordb(self.vectordb)
            return self.search_factory.create_bm25_retriever(docs_for_bm25)
        
        return self.cache_manager.load_or_create_cached_item(
            self.BM25_CACHE_FILE, creation_func, self.CACHE_EXPIRY_SHORT, "BM25 Retriever"
        )

    def _get_cross_encoder_model(self):
        """CrossEncoder 모델 생성 (캐시 활용)"""
        def creation_func():
            return self.search_factory.create_cross_encoder_model(self.CROSS_ENCODER_MODEL_HF)
        
        return self.cache_manager.load_or_create_cached_item(
            self.CROSS_ENCODER_CACHE_FILE, creation_func, self.CACHE_EXPIRY_LONG, "CrossEncoder 모델"
        )
    
    def _determine_weights(self, query: str, character_info: Optional[Dict]) -> List[float]:
        """쿼리와 캐릭터 정보를 기반으로 앙상블 가중치 결정"""
        query_lower = query.lower()

        # 기본값: BM25 우선 (직업도, 캐릭터 정보도 이미 갖고 있음)
        weights = [0.3, 0.7]

        # “최신·업데이트” 류 키워리면 벡터 가중치로 스왑
        if any(k in query_lower for k in ["최신", "업데이트", "현재", "패치", "종결"]):
            weights = [0.7, 0.3]
            print("🔄 최신·패치 관련 키워드 감지 → 벡터 가중치 증가")

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
            print("🔄 캐시된 RAG 검색 결과 사용")
            return cached_result

        search_start_time = time.time()
        # 캐릭터 정보로 쿼리 강화
        enhanced_query = self.text_processor.enhance_query_with_character(query, character_info)
        times = {"internal_search": 0.0}
        
        # 동적 가중치 설정
        weights = self._determine_weights(query, character_info)
        print(f"🎯 앙상블 가중치: 벡터={weights[0]:.2f}, BM25={weights[1]:.2f}")
        
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
                print("🔄 내부 RAG 검색 시작...")
                docs = self.internal_retriever.get_relevant_documents(enhanced_query)
                times["internal_search"] = time.time() - start
                print(f"✅ 내부 RAG 검색 완료: {times['internal_search']:.2f}초, {len(docs)}개 문서")
                return docs
            except Exception as e:
                times["internal_search"] = time.time() - start
                print(f"❌ 내부 RAG 검색 오류 ({times['internal_search']:.2f}초): {e}")
                return []

        internal_docs = _search_internal()
        
        times["internal_search"] = time.time() - search_start_time
        print(f"🎯 내부 검색 완료 - 총 {times['internal_search']:.2f}초")

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
        
        print(f"\n[INFO] 질문 처리 시작: \"{query}\"")
        
        # 이전 대화 기록 로그 출력
        if conversation_history and len(conversation_history) > 0:
            print(f"[INFO] 이전 대화 기록: {len(conversation_history)}개 메시지")
        else:
            print("[INFO] 이전 대화 기록 없음")

        # 캐릭터 정보를 LLM용 컨텍스트로 변환
        char_context_for_llm = self.text_processor.build_character_context_for_llm(character_info)
        
        # 이전 대화 기록을 LLM용 컨텍스트로 변환
        conversation_context_for_llm = self._build_conversation_context_for_llm(conversation_history)
        
        # 검색 수행
        search_results = self.rag_search(query, character_info)
        
        # LLM 답변 생성
        llm_start_time = time.time()
        print("🔄 LLM 답변 생성 중...")
        
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
            if self.enable_grounding:
                google_search_tool = Tool(
                    google_search = GoogleSearch()
                )
                tools.append(google_search_tool)
                print("🔍 웹 검색 그라운딩 활성화됨")
            else:
                print("🚫 웹 검색 그라운딩 비활성화됨")
            
            # LLM 호출
            response = self.genai_client.models.generate_content(
                model=self.LLM_MODEL_NAME,
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
            if self.enable_grounding and hasattr(response.candidates[0], 'grounding_metadata'):
                grounding = response.candidates[0].grounding_metadata
                if hasattr(grounding, 'search_entry_point') and grounding.search_entry_point:
                    print("🌐 웹 검색 그라운딩이 실제로 사용되었습니다!")
                    # 검색된 내용 일부 출력 (디버깅용)
                    if grounding.search_entry_point.rendered_content:
                        print(f"📄 검색 결과 미리보기: {grounding.search_entry_point.rendered_content[:200]}...")
        except Exception as e:
            print(f"❌ LLM 답변 생성 오류: {e}")
            print(f"상세 에러: {str(e)}")
            print(f"에러 타입: {type(e).__name__}")
            llm_response = "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."

        llm_elapsed_time = time.time() - llm_start_time
        total_elapsed_time = time.time() - total_start_time
        
        print(f"✅ LLM 답변 생성 완료 ({llm_elapsed_time:.2f}초)")
        print(f"[INFO] 총 처리 시간: {total_elapsed_time:.2f}초")
        
        # 생성된 답변 출력
        print("\n" + "="*50)
        print("[답변]")
        print("="*50)
        print(llm_response)
        print("="*50 + "\n")
        
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
        print("✨ 새로운 StructuredRAGService 인스턴스 생성 ✨")
        _structured_rag_service_instance = StructuredRAGService()
    return _structured_rag_service_instance

def get_structured_rag_answer(query: str, character_info: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """구조화된 RAG 답변 생성 함수"""
    service = get_structured_rag_service()
    return service.get_answer(query, character_info, conversation_history)
