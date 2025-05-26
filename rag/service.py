"""
분리된 RAG 서비스 - 구조화된 버전
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# LLM & 임베딩
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Gemini 임베딩 import
from vectorstore.gemini_embeddings import GeminiEmbeddings

# 검색 관련
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate

# Google Gen AI SDK
from google import genai

# 분리된 유틸리티들
from .cache_utils import CacheManager
from .text_utils import TextProcessor
from .retrievers import MetadataAwareRetriever
from .search_factory import SearcherFactory
from .web_search import WebSearcher

load_dotenv()


class StructuredRAGService:
    """구조화된 RAG 서비스 클래스"""

    # --- 상수 정의 (기존과 동일하게 유지) ---
    CACHE_DIR_NAME = "cache"
    VECTOR_DB_DIR = "vector_db/chroma"
    EMBED_MODEL_NAME = "text-embedding-004"
    BM25_CACHE_FILE = "bm25_retriever.pkl"
    CROSS_ENCODER_CACHE_FILE = "cross_encoder.pkl"
    LLM_MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"
    CROSS_ENCODER_MODEL_HF = "cross-encoder/ms-marco-MiniLM-L6-v2"

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
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다!")
        
        print("✅ Gemini API 키 확인 완료 - LLM 및 임베딩 모두 Gemini 사용")

    def _initialize_utilities(self):
        """유틸리티 클래스들 초기화"""
        self.cache_manager = CacheManager(self.cache_dir, self.CACHE_EXPIRY_SHORT, self.CACHE_EXPIRY_LONG)
        self.text_processor = TextProcessor()
        self.search_factory = SearcherFactory()

    def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        print("🚀 RAG 시스템 핵심 컴포넌트 초기화 중...")
        
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.gemini_api_key,
            model=self.LLM_MODEL_NAME,
            temperature=0
        )
        
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.web_searcher = WebSearcher(self.gemini_client)
        
        # Gemini 임베딩 함수 초기화
        self.embed_fn = GeminiEmbeddings(
            model=self.EMBED_MODEL_NAME,
            api_key=self.gemini_api_key,
            task_type="RETRIEVAL_QUERY",  # 쿼리 검색용 최적화
            rate_limit_delay=0.05  # 배치 처리로 인해 대기시간 단축
        )
        self.vectordb = Chroma(
            persist_directory=self.VECTOR_DB_DIR,
            embedding_function=self.embed_fn
        )
        
        print("✅ 핵심 컴포넌트 초기화 완료")

    def _initialize_retrievers(self):
        """검색기 초기화"""
        print("🔄 검색기 초기화 중...")
        start_time = time.time()
        
        # 벡터 검색기 설정
        self.vector_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.8},
        )
        
        # BM25 검색기 생성 (캐시 사용)
        self.bm25_retriever = self._get_bm25_retriever()
        
        # 앙상블 검색기 생성
        self.rrf_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],
        )
        
        # CrossEncoder 재랭킹 추가
        cross_encoder_model = self._get_cross_encoder_model()
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=15)
        base_retriever = ContextualCompressionRetriever(
            base_retriever=self.rrf_retriever,
            base_compressor=compressor,
        )
        
        # 메타데이터 인식 검색기로 래핑
        self.internal_retriever = MetadataAwareRetriever(base_retriever)
        
        elapsed_time = time.time() - start_time
        print(f"🎉 검색기 초기화 완료! (소요시간: {elapsed_time:.2f}초)")

    def _setup_llm_and_prompt(self):
        """LLM 및 프롬프트 설정 (기존과 동일하게 유지)"""
        self.hybrid_prompt = PromptTemplate(
            input_variables=["internal_context", "web_context", "question", "character_info"],
            template="""
당신은 던전앤파이터 전문 스펙업 가이드 챗봇입니다.  
※ 반드시 아래 제공된 정보만 활용해 답변하세요.

[캐릭터 정보]
{character_info}

[내부 데이터베이스]
{internal_context}

[웹 검색 결과]
{web_context}

[답변 규칙]
- 제공된 정보 외의 지식은 절대 사용하지 마세요.
- 정보가 부족하면 "제공된 정보에서 찾을 수 없습니다."라고 답변하세요.
- 대답에는 내부 데이터를 최대한 사용하고, 외부 데이터로 검토를 받으세요.
- 사용자의 질문 범위만 다루며, 관련 없는 설명은 생략하세요.
- 순서를 나열하며 설명하고, 짧고 간결하게 핵심만 설명하세요.
- 답변엔 간단한 출처를 함께 작성하세요.

[콘텐츠 관련]
- 콘텐츠 관련 대답이 들어올 경우엔, 명성을 기준으로 대답하세요.
- 콘텐츠에는 입장 명성과 권장 명성이 있는데, 권장 명성 기준으로 얘기하세요.

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

    def _hybrid_search(self, query: str, character_info: Optional[Dict]) -> Dict[str, Any]:
        """하이브리드 검색 (내부 + 웹)"""
        # 캐시 확인
        cached_result = self.cache_manager.get_cached_search_result(query, 'hybrid_search', character_info)
        if cached_result:
            print("🔄 캐시된 하이브리드 검색 결과 사용")
            return cached_result

        search_start_time = time.time()
        enhanced_query = self.text_processor.enhance_query_with_character(query, character_info)
        times = {"internal_search": 0.0, "web_search": 0.0, "total_search": 0.0}

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

        def _search_web():
            start = time.time()
            try:
                print("🔄 웹 검색 (Gemini) 시작...")
                # 웹 검색은 캐시를 별도로 확인
                cached_web_result = self.cache_manager.get_cached_search_result(query, 'gemini_search', character_info)
                if cached_web_result:
                    print("🔄 캐시된 Gemini 웹 검색 결과 사용")
                    docs = cached_web_result
                else:
                    docs = self.web_searcher.search_with_grounding(query, character_info)
                    self.cache_manager.save_search_result_to_cache(query, docs, 'gemini_search', character_info)
                
                times["web_search"] = time.time() - start
                print(f"✅ 웹 검색 완료: {times['web_search']:.2f}초, {len(docs)}개 문서")
                return docs
            except Exception as e:
                times["web_search"] = time.time() - start
                print(f"❌ 웹 검색 오류 ({times['web_search']:.2f}초): {e}")
                return []

        # 병렬 검색 실행
        print("🚀 병렬 검색 시작 (내부 RAG + 웹)")
        with ThreadPoolExecutor(max_workers=2) as executor:
            internal_future = executor.submit(_search_internal)
            web_future = executor.submit(_search_web)
            internal_docs = internal_future.result()
            web_docs = web_future.result()
        
        times["total_search"] = time.time() - search_start_time
        print(f"🎯 하이브리드 검색 완료 - 총 {times['total_search']:.2f}초")

        # 검색 결과를 컨텍스트 문자열로 변환
        internal_context_str = self.text_processor.format_docs_to_context_string(internal_docs, "내부")
        web_context_str = self.text_processor.format_web_search_docs_to_context_string(web_docs)
        
        # 결과 구성
        result = {
            "all_docs": internal_docs + web_docs,
            "internal_docs": internal_docs,
            "web_docs": web_docs,
            "internal_context_provided_to_llm": internal_context_str,
            "web_context_provided_to_llm": web_context_str,
            "used_web_search": bool(web_docs),
            "enhanced_query": enhanced_query,
            "search_times": times
        }
        
        # 캐시에 저장
        self.cache_manager.save_search_result_to_cache(query, result, 'hybrid_search', character_info)
        return result

    def get_answer(self, query: str, character_info: Optional[Dict] = None) -> Dict[str, Any]:
        """RAG 답변 생성 (메인 API)"""
        total_start_time = time.time()
        
        print(f"\n[INFO] 질문 처리 시작: \"{query}\"")
        char_desc_parts = []
        if character_info:
            if class_info := character_info.get('class'):
                char_desc_parts.append(class_info)
            if fame_info := character_info.get('fame'):
                char_desc_parts.append(f"{fame_info}명성")
            if char_desc_parts:
                print(f"[INFO] 캐릭터: {' '.join(char_desc_parts)}")

        # 캐릭터 정보를 LLM용 컨텍스트로 변환
        char_context_for_llm = self.text_processor.build_character_context_for_llm(character_info)
        
        # 하이브리드 검색 수행
        search_results = self._hybrid_search(query, character_info)
        
        # LLM 답변 생성
        llm_start_time = time.time()
        print("🔄 LLM 답변 생성 중...")
        
        formatted_prompt = self.hybrid_prompt.format(
            internal_context=search_results["internal_context_provided_to_llm"],
            web_context=search_results["web_context_provided_to_llm"],
            question=query,
            character_info=char_context_for_llm
        )
        
        try:
            llm_response = self.llm.invoke(formatted_prompt).content
        except Exception as e:
            print(f"❌ LLM 답변 생성 오류: {e}")
            llm_response = "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."

        llm_elapsed_time = time.time() - llm_start_time
        total_elapsed_time = time.time() - total_start_time
        
        print(f"✅ LLM 답변 생성 완료 ({llm_elapsed_time:.2f}초)")
        print(f"[INFO] 총 처리 시간: {total_elapsed_time:.2f}초")
        
        # FastAPI 엔드포인트에서 기대하는 키로 반환값 구성
        return {
            "result": llm_response,
            "source_documents": search_results["all_docs"],
            "used_web_search": search_results["used_web_search"],
            "internal_docs": search_results["internal_docs"],
            "web_docs": search_results["web_docs"],
            "enhanced_query": search_results["enhanced_query"],
            "execution_times": {
                "total": total_elapsed_time,
                "llm": llm_elapsed_time,
                "search": search_results["search_times"]
            },
            "internal_context": search_results["internal_context_provided_to_llm"],
            "web_context": search_results["web_context_provided_to_llm"]
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

def get_structured_rag_answer(query: str, character_info: Optional[Dict] = None) -> Dict[str, Any]:
    """구조화된 RAG 답변 생성 함수"""
    service = get_structured_rag_service()
    return service.get_answer(query, character_info)
