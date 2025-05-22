from __future__ import annotations
import os
import time
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dotenv import load_dotenv

# LLM & 임베딩
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 검색 관련
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Google Gen AI SDK
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

load_dotenv()

class RAGService:
    """RAG 서비스 클래스 - Gemini 검색 그라운딩 기반 검색 강화"""

    # --- 상수 정의 ---
    CACHE_DIR_NAME = "cache"
    VECTOR_DB_DIR = "vector_db/chroma"
    EMBED_MODEL_NAME = "text-embedding-3-large"
    BM25_CACHE_FILE = "bm25_retriever.pkl"
    CROSS_ENCODER_CACHE_FILE = "cross_encoder.pkl"
    LLM_MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"
    GEMINI_SEARCH_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
    CROSS_ENCODER_MODEL_HF = "cross-encoder/ms-marco-MiniLM-L6-v2"

    CACHE_EXPIRY_SHORT = 60 * 60 * 12  # 12시간
    CACHE_EXPIRY_LONG = 60 * 60 * 24   # 24시간

    # --- 초기화 관련 메소드 ---
    def __init__(self):
        """RAG 서비스 초기화"""
        self._setup_environment()
        self._initialize_core_components()
        self._initialize_retrievers()
        self._setup_llm_and_prompt()

    def _setup_environment(self):
        self.cache_dir = Path(self.CACHE_DIR_NAME)
        self.cache_dir.mkdir(exist_ok=True)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다!")
        if not os.getenv("OPENAI_API_KEY"):
            print("경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    def _initialize_core_components(self):
        print("🚀 RAG 시스템 핵심 컴포넌트 초기화 중...")
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.gemini_api_key,
            model=self.LLM_MODEL_NAME,
            temperature=0
        )
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.embed_fn = OpenAIEmbeddings(model=self.EMBED_MODEL_NAME)
        self.vectordb = Chroma(
            persist_directory=self.VECTOR_DB_DIR,
            embedding_function=self.embed_fn
        )
        print("✅ 핵심 컴포넌트 초기화 완료")

    def _initialize_retrievers(self):
        print("🔄 검색기 초기화 중...")
        start_time = time.time()
        self.vector_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.6},
        )
        self.bm25_retriever = self._get_bm25_retriever()
        self.rrf_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],
        )
        cross_encoder_model = self._get_cross_encoder_model()
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=10)
        base_retriever = ContextualCompressionRetriever(
            base_retriever=self.rrf_retriever,
            base_compressor=compressor,
        )
        self.internal_retriever = MetadataAwareRetriever(base_retriever)
        elapsed_time = time.time() - start_time
        print(f"🎉 검색기 초기화 완료! (소요시간: {elapsed_time:.2f}초)")

    def _setup_llm_and_prompt(self):
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
- 출처는 제공하지 않아도 됩니다.

[콘텐츠 관련]
- 콘텐츠 관련 대답이 들어올 경우엔, 명성을 기준으로 대답하세요.
- 콘텐츠에는 입장 명성과 권장 명성이 있는데, 권장 명성 기준으로 얘기하세요.

[이벤트 안내 기준]
- 종료일이 2025-05-22 이후 → 참여 권장
- 종료된 이벤트 → "해당 이벤트는 종료되었습니다."
- 종료일이 없을 경우 → "이벤트 종료일을 확인해주세요."

[사용자 질문]
{question}

[답변 - 간결하고 명확하게]
"""
        )
        print("✅ LLM 프롬프트 설정 완료")

    # --- 캐싱 관련 헬퍼 메소드 ---
    def _generate_cache_key(self, base_content: str, character_info: Optional[Dict] = None) -> str:
        """캐시 키 생성 (FastAPI에서 변환된 캐릭터 정보 포함 가능)"""
        cache_input = base_content
        if character_info:
            # FastAPI에서 변환된 키들을 사용
            char_key_parts = [
                character_info.get('class', ''), # 'class_name' -> 'class'
                # character_info.get('level', ''), # level은 현재 변환 규칙에 없음. 필요시 FastAPI 변환 로직에 추가
                str(character_info.get('fame', ''))
            ]
            
            # 주요 정보만으로 키 생성 (더 간단한 방식)
            simple_char_key = "-".join(filter(None, char_key_parts))

            if simple_char_key:
                 cache_input = f"{base_content}|{simple_char_key}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def _load_or_create_cached_item(self, 
                                    cache_file_name: str, 
                                    creation_func: Callable[[], Any], 
                                    expiry_seconds: int,
                                    item_name: str = "항목") -> Any:
        cache_file = self.cache_dir / cache_file_name
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < expiry_seconds:
                try:
                    print(f"🔄 캐시된 {item_name} 로딩: {cache_file_name}")
                    with open(cache_file, 'rb') as f: item = pickle.load(f)
                    print(f"✅ {item_name} 캐시 로드 완료")
                    return item
                except Exception as e:
                    print(f"⚠️ {item_name} 캐시 로드 실패 ({cache_file_name}): {e}. 재생성합니다.")
        print(f"🔄 {item_name} 생성 중 ({cache_file_name})...")
        item = creation_func()
        try:
            with open(cache_file, 'wb') as f: pickle.dump(item, f)
            print(f"✅ {item_name} 캐시 저장 완료: {cache_file}")
        except Exception as e:
            print(f"⚠️ {item_name} 캐시 저장 실패 ({cache_file_name}): {e}")
        return item

    def _get_cached_search_result(self, query: str, cache_type: str, character_info: Optional[Dict] = None) -> Optional[Any]:
        cache_key = self._generate_cache_key(query, character_info)
        cache_file_name = f"{cache_type}_{cache_key}.pkl"
        cache_file = self.cache_dir / cache_file_name
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.CACHE_EXPIRY_SHORT:
                try:
                    with open(cache_file, 'rb') as f: return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ {cache_type} 검색 캐시 로드 실패: {e}")
        return None

    def _save_search_result_to_cache(self, query: str, result: Any, cache_type: str, character_info: Optional[Dict] = None):
        cache_key = self._generate_cache_key(query, character_info)
        cache_file_name = f"{cache_type}_{cache_key}.pkl"
        cache_file = self.cache_dir / cache_file_name
        try:
            with open(cache_file, 'wb') as f: pickle.dump(result, f)
        except Exception as e:
            print(f"⚠️ {cache_type} 검색 캐시 저장 실패: {e}")

    # --- BM25 및 CrossEncoder 로딩/생성 메소드 ---
    def _create_bm25_data_from_vectordb(self) -> List[Document]:
        print("🔄 VectorDB에서 BM25용 데이터 추출 중...")
        store_data = self.vectordb.get(include=["documents", "metadatas"])
        docs_for_bm25 = []
        for txt, meta in zip(store_data["documents"], store_data["metadatas"]):
            enhanced_content = txt
            if meta:
                if meta.get("title"):
                    enhanced_content = f"제목: {meta['title']}\n{txt}"
                if meta.get("class_name"): # VectorDB의 class_name은 그대로 유지 (BM25 인덱싱용)
                    enhanced_content += f"\n직업: {meta['class_name']}"
            docs_for_bm25.append(Document(page_content=enhanced_content, metadata=meta))
        print(f"✅ BM25용 문서 {len(docs_for_bm25)}개 준비 완료")
        return docs_for_bm25

    def _get_bm25_retriever(self) -> BM25Retriever:
        def creation_func():
            docs_for_bm25 = self._create_bm25_data_from_vectordb()
            bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            bm25_retriever.k = 15
            return bm25_retriever
        return self._load_or_create_cached_item(
            self.BM25_CACHE_FILE, creation_func, self.CACHE_EXPIRY_SHORT, "BM25 Retriever"
        )

    def _get_cross_encoder_model(self) -> HuggingFaceCrossEncoder:
        def creation_func():
            return HuggingFaceCrossEncoder(
                model_name=self.CROSS_ENCODER_MODEL_HF, model_kwargs={"device": "cpu"}
            )
        return self._load_or_create_cached_item(
            self.CROSS_ENCODER_CACHE_FILE, creation_func, self.CACHE_EXPIRY_LONG, "CrossEncoder 모델"
        )

    # --- 검색 쿼리 및 컨텍스트 처리 ---
    def _enhance_query_with_character(self, query: str, character_info: Optional[Dict]) -> str:
        """캐릭터 정보로 검색 쿼리 강화 (FastAPI에서 변환된 키 사용)"""
        if not character_info:
            return query
        
        enhancements = []
        # FastAPI에서 변환된 'class' 키 사용
        if class_info := character_info.get('class'):
            enhancements.append(class_info)
        if fame := character_info.get('fame'):
            enhancements.append(str(fame))
        
        if enhancements:
            enhanced_query = f"{' '.join(enhancements)} {query}"
            print(f"[DEBUG] 쿼리 강화: '{query}' → '{enhanced_query}'")
            return enhanced_query
        return query

    def _format_docs_to_context_string(self, docs: List[Document], context_type: str) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            content = f"[{context_type} 문서 {i+1}] {doc.page_content}"
            if doc.metadata and (url := doc.metadata.get("url")):
                content += f"\n참고 링크: {url}"
            context_parts.append(content)
        return "\n\n".join(context_parts)

    def _format_web_search_docs_to_context_string(self, web_docs: List[Document]) -> str:
        web_context_parts = []
        main_content_doc = next((doc for doc in web_docs if doc.metadata.get("source") == "gemini_search"), None)
        if main_content_doc:
            web_context_parts.append(f"[Gemini 웹 검색 결과 - 2025년 최신 정보]\n{main_content_doc.page_content}")
        source_docs = [doc for doc in web_docs if doc.metadata.get("source") in ["grounding_source", "search_suggestions"]]
        if source_docs:
            web_context_parts.append("[참고 출처]")
            for i, doc in enumerate(source_docs):
                title = doc.metadata.get("title", f"출처 {i+1}")
                url = doc.metadata.get("url", "")
                entry = f"출처 {i+1}: {title}"
                if url: entry += f" - {url}"
                web_context_parts.append(entry)
        return "\n\n".join(web_context_parts) if web_context_parts else "웹 검색 결과 없음."

    # --- 핵심 검색 로직 ---
    def _gemini_search_grounding(self, query: str, character_info: Optional[Dict]) -> List[Document]:
        cached_result = self._get_cached_search_result(query, 'gemini_search', character_info)
        if cached_result:
            print("🔄 캐시된 Gemini 웹 검색 결과 사용")
            return cached_result

        enhanced_query = self._enhance_query_with_character(query, character_info)
        
        system_instruction = """당신은 던전앤파이터 전문가입니다.
[중요한 날짜 제약사항]
- 반드시 2025년 1월 1일 이후의 최신 정보만 검색하고 사용하세요
- 2024년 12월 31일 이전의 정보는 절대 참조하지 마세요
- 검색 시 "2025" 키워드를 포함하여 최신성을 보장하세요
- 정보의 날짜를 확인할 수 없다면 해당 정보는 사용하지 마세요
[목표]
- 2025년 최신 던파 정보 제공
- 캐릭터 맞춤형 간단한 가이드 
- 핵심 정보만 간결하게 전달
[답변 형식]
- 최소한으로 대답
- 구체적인 수치나 방법 우선
- 불필요한 설명 제외
"""
        character_context_str = ""
        if character_info: # FastAPI에서 변환된 character_info 사용
            details = []
            if class_info := character_info.get('class'):
                details.append(f"- 직업: {class_info}")
            if fame_info := character_info.get('fame'):
                details.append(f"- 명성: {fame_info}")
            # Gemini 검색 프롬프트에는 핵심 정보만 포함 (필요시 추가)
            if details:
                character_context_str = "사용자 캐릭터 정보:\n" + "\n".join(details)
                character_context_str += "\n\n위 캐릭터 정보를 고려하여 맞춤형 정보를 검색하세요."
            else:
                 character_context_str = "캐릭터 정보가 제공되었으나, 세부 내용을 파악할 수 없습니다."

        final_prompt = f"{system_instruction}\n{character_context_str}\n[검색 요청]\n2025년 던전앤파이터 \"{enhanced_query}\"에 대한 간단하고 핵심적인 정보만 검색해주세요."

        try:
            print(f"[DEBUG] Gemini 검색 실행: {enhanced_query}")
            # GoogleSearch()에는 max_results 파라미터 없음
            google_search_tool = Tool(google_search=GoogleSearch())
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=final_prompt,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0.1,  # 일관성 있는 답변을 위해 낮게 설정
                    max_output_tokens=1000,  # 충분한 정보 확보를 위해 증가
                )
            )
            
            docs = []
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    main_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
                    if main_content:
                        docs.append(Document(page_content=main_content, metadata={"title": "Gemini 검색 결과", "source": "gemini_search"}))
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding = candidate.grounding_metadata
                    if hasattr(grounding, 'search_entry_point') and grounding.search_entry_point:
                        docs.append(Document(page_content="Google 검색 제안사항 및 관련 링크", metadata={"title": "검색 제안", "source": "search_suggestions"}))
                    if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                        for i, chunk in enumerate(grounding.grounding_chunks):
                            if hasattr(chunk, 'web') and chunk.web:
                                web_info = chunk.web
                                docs.append(Document(
                                    page_content=f"출처 {i+1}에서 참조된 정보",
                                    metadata={"title": getattr(web_info, 'title', f'웹 출처 {i+1}'), 
                                              "url": getattr(web_info, 'uri', ''), 
                                              "source": "grounding_source"}
                                ))
                    if hasattr(grounding, 'web_search_queries') and grounding.web_search_queries:
                        print(f"[DEBUG] 웹 검색 쿼리: {grounding.web_search_queries}")
            
            print(f"[DEBUG] Gemini 검색 결과 문서 {len(docs)}개 생성")
            self._save_search_result_to_cache(query, docs, 'gemini_search', character_info)
            return docs
        except Exception as e:
            print(f"❌ Gemini 검색 그라운딩 오류: {e}")
            return []

    def _hybrid_search(self, query: str, character_info: Optional[Dict]) -> Dict[str, Any]:
        cached_result = self._get_cached_search_result(query, 'hybrid_search', character_info)
        if cached_result:
            print("🔄 캐시된 하이브리드 검색 결과 사용")
            return cached_result

        search_start_time = time.time()
        enhanced_query = self._enhance_query_with_character(query, character_info)
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
                docs = self._gemini_search_grounding(query, character_info)
                times["web_search"] = time.time() - start
                print(f"✅ 웹 검색 완료: {times['web_search']:.2f}초, {len(docs)}개 문서")
                return docs
            except Exception as e:
                times["web_search"] = time.time() - start
                print(f"❌ 웹 검색 오류 ({times['web_search']:.2f}초): {e}")
                return []

        print("🚀 병렬 검색 시작 (내부 RAG + 웹)")
        with ThreadPoolExecutor(max_workers=2) as executor:
            internal_future = executor.submit(_search_internal)
            web_future = executor.submit(_search_web)
            internal_docs = internal_future.result()
            web_docs = web_future.result()
        
        times["total_search"] = time.time() - search_start_time
        print(f"🎯 하이브리드 검색 완료 - 총 {times['total_search']:.2f}초")

        internal_context_str = self._format_docs_to_context_string(internal_docs, "내부")
        web_context_str = self._format_web_search_docs_to_context_string(web_docs)
        
        result = {
            "all_docs": internal_docs + web_docs,
            "internal_docs": internal_docs,
            "web_docs": web_docs,
            "internal_context_provided_to_llm": internal_context_str, # FastAPI에서 사용하는 키로 변경
            "web_context_provided_to_llm": web_context_str,       # FastAPI에서 사용하는 키로 변경
            "used_web_search": bool(web_docs),
            "enhanced_query": enhanced_query,
            "search_times": times
        }
        self._save_search_result_to_cache(query, result, 'hybrid_search', character_info)
        return result

    # --- 공개 API 메소드 ---
    def get_answer(self, query: str, character_info: Optional[Dict] = None) -> Dict[str, Any]:
        total_start_time = time.time()
        
        print(f"\n[INFO] 질문 처리 시작: \"{query}\"")
        char_desc_parts = []
        if character_info: # FastAPI에서 변환된 character_info 사용
            if class_info := character_info.get('class'):
                char_desc_parts.append(class_info)
            if fame_info := character_info.get('fame'):
                char_desc_parts.append(f"{fame_info}명성")
            if char_desc_parts:
                 print(f"[INFO] 캐릭터: {' '.join(char_desc_parts)}")

        char_context_for_llm = "캐릭터 정보 없음."
        if character_info: # FastAPI에서 변환된 character_info 사용
            details = []
            if class_info := character_info.get('class'):
                details.append(f"- 직업: {class_info}")
            if fame_info := character_info.get('fame'):
                details.append(f"- 명성: {fame_info}")
            if weapon_info := character_info.get('weapon'):
                details.append(f"- 무기: {weapon_info}")
            if epic_num := character_info.get('epicNum'):
                details.append(f"- 에픽 아이템 개수: {epic_num}")
            if originality_num := character_info.get('originalityNum'):
                details.append(f"- 태초 아이템 개수: {originality_num}")
            if title_info := character_info.get('title'):
                details.append(f"- 칭호: {title_info}")
            if set_item_name := character_info.get('set_item_name'):
                set_rarity = character_info.get('set_item_rarity', '')
                details.append(f"- 세트 아이템: {set_item_name} ({set_rarity} 등급)")
            if creature_info := character_info.get('creature'):
                details.append(f"- 크리쳐: {creature_info}")
            if aura_info := character_info.get('aura'):
                details.append(f"- 오라: {aura_info}")

            if details:
                char_context_for_llm = "사용자 캐릭터 정보:\n" + "\n".join(details)
                char_context_for_llm += "\n\n위 캐릭터 정보를 고려하여 맞춤형 조언을 제공하세요."
            else:
                char_context_for_llm = "캐릭터 정보가 제공되었으나, 세부 내용을 파악할 수 없습니다."
        
        search_results = self._hybrid_search(query, character_info)
        
        llm_start_time = time.time()
        print("🔄 LLM 답변 생성 중...")
        formatted_prompt = self.hybrid_prompt.format(
            internal_context=search_results["internal_context_provided_to_llm"], # 키 일치
            web_context=search_results["web_context_provided_to_llm"],       # 키 일치
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
            "internal_docs": search_results["internal_docs"], # FastAPI에서 convert_docs_to_dict로 처리
            "web_docs": search_results["web_docs"],         # FastAPI에서 convert_docs_to_dict로 처리
            "enhanced_query": search_results["enhanced_query"],
            "execution_times": {
                "total": total_elapsed_time,
                "llm": llm_elapsed_time,
                "search": search_results["search_times"]
            },
            "internal_context": search_results["internal_context_provided_to_llm"], # 키 일치
            "web_context": search_results["web_context_provided_to_llm"]        # 키 일치
        }

class MetadataAwareRetriever:
    def __init__(self, base_retriever, top_n: int = 15):
        self.base_retriever = base_retriever
        self.top_n = top_n
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        scored_docs = []
        for doc in docs:
            score = 1.0
            meta = doc.metadata or {}
            try: views = int(meta.get("views", 0)); score += 0.2 if views > 100000 else (0.1 if views > 10000 else 0)
            except ValueError: pass
            try: likes = int(meta.get("likes", 0)); score += 0.1 if likes > 100 else (0.05 if likes > 50 else 0)
            except ValueError: pass
            try: priority = float(meta.get("priority_score", 0.0)); score += priority * 0.1
            except ValueError: pass
            try: content_s = float(meta.get("content_score", 0.0)); score += content_s * 0.01
            except ValueError: pass
            if class_name := meta.get("class_name"): # VectorDB의 class_name은 그대로 사용
                if isinstance(class_name, str) and class_name.lower() in query.lower():
                    score += 0.3
            scored_docs.append((doc, score))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:self.top_n]]

_rag_service_instance: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        print("✨ 새로운 RAGService 인스턴스 생성 ✨")
        _rag_service_instance = RAGService()
    return _rag_service_instance

def get_rag_answer(query: str, character_info: Optional[Dict] = None) -> Dict[str, Any]:
    service = get_rag_service()
    return service.get_answer(query, character_info)

if __name__ == "__main__":
    print("RAG 서비스 테스트 시작...")
    
    # FastAPI에서 변환된 형태와 유사한 테스트용 캐릭터 정보
    test_character_info_transformed = {
        "class": "아수라(남귀검사)",
        "fame": "52000",
        "weapon": "태초 무기",
        "epicNum": 5,
        "originalityNum": 1,
        "title": "세리아 칭호",
        "set_item_name": "칠흑의 정화 세트",
        "set_item_rarity": "레전더리2",
        "creature": "세리아 크리쳐",
        "aura": "세리아 오라"
    }
    
    try:
        rag_service = get_rag_service()
        test_queries = [
            "여기서 더 스펙업 하려면 어떻게 해야해?",
            "레기온 베누스 가이드라인 알려줘",
            "이번주 주요 이벤트 뭐 있어?",
        ]
        
        for i, q in enumerate(test_queries):
            print(f"\n--- 테스트 질문 {i+1} ---")
            answer_with_char = rag_service.get_answer(q, test_character_info_transformed)
            print(f"\n[답변 (캐릭터 정보 포함)]\n{answer_with_char['result']}")
            
            # internal_docs, web_docs는 Document 객체 리스트이므로, FastAPI 엔드포인트처럼 처리하려면 변환 필요
            internal_docs_count = len(answer_with_char.get("internal_docs", []))
            web_docs_count = len(answer_with_char.get("web_docs", []))

            print(f"  (웹 검색 사용: {answer_with_char['used_web_search']}, 내부 문서: {internal_docs_count}, 웹 문서: {web_docs_count})")
            print(f"  (실행 시간: 총 {answer_with_char['execution_times']['total']:.2f}s, LLM {answer_with_char['execution_times']['llm']:.2f}s)")
            
            if i < len(test_queries) - 1:
                print("\n... 다음 질문 대기 중 (1초) ...") # 테스트 시간 단축
                time.sleep(1)

    except RuntimeError as e:
        print(f"테스트 중 런타임 오류 발생: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
    print("\n--- RAG 서비스 테스트 종료 ---")