"""
RAG 서비스 모듈 - 캐릭터 정보를 활용한 개선된 검색
"""

from __future__ import annotations
import os, time, hashlib, pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
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

# 웹 검색
from openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

load_dotenv()

class RAGService:
    """RAG 서비스 클래스 - 캐릭터 정보 기반 검색 강화"""
    
    def __init__(self):
        """RAG 서비스 초기화"""
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 환경변수
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수 필요!")
        
        # 초기화
        self._initialize_rag_components()
        self._setup_llm_and_prompt()
    
    def _initialize_rag_components(self):
        """RAG 컴포넌트들 초기화"""
        print("🚀 RAG 시스템 초기화 중...")
        start_time = time.time()
        
        # 벡터 DB 설정
        chroma_dir = "vector_db/chroma"
        embed_model = "text-embedding-3-large"
        embed_fn = OpenAIEmbeddings(model=embed_model)
        
        print("🔄 벡터 DB 로딩...")
        self.vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embed_fn)
        self.vector_retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.8},
        )
        print("✅ 벡터 DB 로딩 완료")
        
        # BM25 로드
        self.bm25_retriever = self._load_bm25_index()
        
        # Ensemble 설정
        self.rrf_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5],
        )
        
        # Cross-Encoder 설정
        cross_encoder = self._load_cross_encoder()
        compressor = CrossEncoderReranker(
            model=cross_encoder,
            top_n=6
        )
        self.internal_retriever = ContextualCompressionRetriever(
            base_retriever=self.rrf_retriever,
            base_compressor=compressor,
        )
        
        elapsed_time = time.time() - start_time
        print(f"🎉 RAG 시스템 초기화 완료! (소요시간: {elapsed_time:.2f}초)")
    
    def _build_bm25_index(self):
        """BM25 인덱스를 구축하고 캐싱"""
        print("🔄 BM25 인덱스 구축 중...")
        store_data = self.vectordb.get(include=["documents", "metadatas"])
        docs_for_bm25 = []
        
        for txt, meta in zip(store_data["documents"], store_data["metadatas"]):
            enhanced_content = txt
            if meta:
                if meta.get("title"):
                    enhanced_content = f"제목: {meta['title']}\\n{txt}"
                if meta.get("class_name"):
                    enhanced_content += f"\\n직업: {meta['class_name']}"
            
            docs_for_bm25.append(Document(page_content=enhanced_content, metadata=meta))
        
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = 6
        
        # 캐싱 저장
        cache_file = self.cache_dir / "bm25_index.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print(f"✅ BM25 인덱스 캐시 저장: {cache_file}")
        except Exception as e:
            print(f"⚠️ BM25 캐시 저장 실패: {e}")
        
        return bm25_retriever
    
    def _load_bm25_index(self):
        """캐시에서 BM25 인덱스 로드하거나 새로 구축"""
        cache_file = self.cache_dir / "bm25_index.pkl"
        cache_expiry = 60 * 60 * 12  # 12시간
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < cache_expiry:
                try:
                    print("🔄 캐시된 BM25 인덱스 로딩...")
                    with open(cache_file, 'rb') as f:
                        bm25_retriever = pickle.load(f)
                    print("✅ BM25 인덱스 캐시 로드 완료")
                    return bm25_retriever
                except Exception as e:
                    print(f"⚠️ BM25 캐시 로드 실패: {e}")
        
        return self._build_bm25_index()
    
    def _load_cross_encoder(self):
        """크로스 인코더 모델 로드 (캐시 활용)"""
        cache_file = self.cache_dir / "cross_encoder.pkl"
        cache_expiry = 60 * 60 * 24  # 24시간
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < cache_expiry:
                try:
                    print("🔄 캐시된 Cross-Encoder 로딩...")
                    with open(cache_file, 'rb') as f:
                        cross_encoder = pickle.load(f)
                    print("✅ Cross-Encoder 캐시 로드 완료")
                    return cross_encoder
                except Exception as e:
                    print(f"⚠️ Cross-Encoder 캐시 로드 실패: {e}")
        
        print("🔄 Cross-Encoder 모델 로딩...")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # 캐싱 저장
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cross_encoder, f)
            print(f"✅ Cross-Encoder 캐시 저장: {cache_file}")
        except Exception as e:
            print(f"⚠️ Cross-Encoder 캐시 저장 실패: {e}")
        
        return cross_encoder
    
    def _setup_llm_and_prompt(self):
        """LLM과 프롬프트 설정"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.gemini_api_key,
            model="models/gemini-2.5-flash-preview-05-20",
            temperature=0,
        )
        
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

───────────────
답변 규칙
───────────────
- 제공된 정보 외의 지식은 절대 사용하지 마세요.
- 정보가 부족하면 "제공된 정보에서 찾을 수 없습니다."라고 답변하세요.
- 내부 데이터를 우선적으로 활용하고, 외부 데이터를 참조하는 형태로 사용하세요요.
- 내외부 모든 데이터는 2025년 데이터만 활용합니다.
- 사용자의 질문 범위만 다루며, 관련 없는 설명은 생략하세요.
- 순서를 나열하며 출처와 함께 대답해주세요.

[이벤트 안내 기준]
- 종료일이 2025-05-22 이후 → 참여 권장
- 종료된 이벤트 → "해당 이벤트는 종료되었습니다."
- 종료일이 없을 경우 → "이벤트 종료일을 확인해주세요."

[사용자 질문]
{question}

[답변]
"""
        )
    
    def _generate_cache_key(self, query, character_info=None):
        """캐시 키 생성 (캐릭터 정보 포함)"""
        cache_content = query
        if character_info:
            # 캐릭터 정보를 캐시 키에 포함
            char_key = f"{character_info.get('class_name', '')}-{character_info.get('level', '')}-{character_info.get('fame', '')}"
            cache_content = f"{query}|{char_key}"
        return hashlib.md5(cache_content.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, query, cache_type='search', character_info=None):
        """캐시에서 결과 조회"""
        cache_key = self._generate_cache_key(query, character_info)
        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        cache_expiry = 60 * 60 * 12  # 12시간
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < cache_expiry:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
        return None
    
    def _save_to_cache(self, query, result, cache_type='search', character_info=None):
        """캐시에 결과 저장"""
        cache_key = self._generate_cache_key(query, character_info)
        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"캐시 저장 중 오류: {e}")
    
    def _enhance_query_with_character(self, query: str, character_info: Optional[Dict] = None) -> str:
        """캐릭터 정보로 검색 쿼리 강화"""
        if not character_info:
            return query
        
        enhancements = []
        
        # 직업명 추가
        if character_info.get('class_name'):
            enhancements.append(character_info['class_name'])
        
        # 명성추가
        if character_info.get('fame'):
           enhancements.append(str(character_info['fame']))
        
        # 강화된 쿼리 생성
        if enhancements:
            enhanced_query = f"{' '.join(enhancements)} {query}"
            print(f"[DEBUG] 쿼리 강화: '{query}' → '{enhanced_query}'")
            return enhanced_query
        
        return query
    
    def _perplexity_web_search(self, query: str, character_info: Optional[Dict] = None, max_results=3) -> List[Document]:
        """Perplexity 웹 검색"""
        cached_result = self._get_from_cache(query, 'web_search', character_info)
        if cached_result:
            print("🔄 캐시된 웹 검색 결과 사용")
            return cached_result
        
        if not self.perplexity_api_key:
            return []
        
        try:
            client = OpenAI(
                api_key=self.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            
            # 캐릭터 정보로 강화된 쿼리 사용
            enhanced_query = self._enhance_query_with_character(query, character_info)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that provides information about "
                        "Dungeon & Fighter (DNF) game. Focus on providing the most "
                        "recent and accurate information about character progression, equipment, "
                        "and optimization guides. Be concise and direct."
                    )
                },
                {"role": "user", "content": f"2025 최신 던전앤파이터 {enhanced_query} 스펙업 가이드"}
            ]
            
            response = client.chat.completions.create(
                model="sonar",
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            
            docs = []
            content = response.choices[0].message.content
            docs.append(Document(
                page_content=content,
                metadata={"title": "Perplexity 검색 결과", "url": "", "source": "web_search"}
            ))
            
            self._save_to_cache(query, docs, 'web_search', character_info)
            return docs
            
        except Exception as e:
            print(f"❌ 웹 검색 오류: {e}")
            return []
    
    def _smart_hybrid_search(self, query, character_info: Optional[Dict] = None):
        """하이브리드 검색 (내부 + 웹)"""
        cached_result = self._get_from_cache(query, 'hybrid_search', character_info)
        if cached_result:
            return cached_result
        
        # 캐릭터 정보로 쿼리 강화
        enhanced_query = self._enhance_query_with_character(query, character_info)
        
        def get_internal_results():
            try:
                return self.internal_retriever.get_relevant_documents(enhanced_query)
            except Exception as e:
                print(f"[ERROR] 내부 검색 오류: {e}")
                return []
        
        def get_web_results():
            try:
                return self._perplexity_web_search(query, character_info)
            except Exception as e:
                print(f"[ERROR] 웹 검색 오류: {e}")
                return []
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            internal_future = executor.submit(get_internal_results)
            web_future = executor.submit(get_web_results)
            
            internal_docs = internal_future.result()
            web_docs = web_future.result()
        
        print(f"[DEBUG] 검색 결과: 내부 {len(internal_docs)}개, 웹 {len(web_docs)}개")
        
        # 컨텍스트 구성
        internal_context_parts = []
        for i, doc in enumerate(internal_docs):
            content = f"[내부 문서 {i+1}] {doc.page_content}"
            if doc.metadata and doc.metadata.get("url"):
                content += f"\\n참고 링크: {doc.metadata['url']}"
            internal_context_parts.append(content)
        
        internal_context = "\\n\\n".join(internal_context_parts)
        
        web_context = "\\n\\n".join([
            f"[웹 문서 {i+1} - 2025년 최신 정보] {doc.page_content}"
            for i, doc in enumerate(web_docs)
        ]) if web_docs else ""
        
        if not internal_docs and not web_docs:
            internal_context = "[검색 결과 없음] 질문과 관련된 정보를 찾지 못했습니다."
        
        result = {
            "all_docs": internal_docs + web_docs,
            "internal_docs": internal_docs,
            "web_docs": web_docs,
            "internal_context": internal_context,
            "web_context": web_context,
            "used_web_search": len(web_docs) > 0,
            "enhanced_query": enhanced_query  # 디버깅용
        }
        
        self._save_to_cache(query, result, 'hybrid_search', character_info)
        return result
    
    def get_answer(self, query: str, character_info: Optional[Dict] = None):
        """RAG 답변 생성"""
        total_start_time = time.time()
        
        print(f"[INFO] 질문 처리 시작: {query}")
        if character_info:
            print(f"[INFO] 캐릭터: {character_info.get('class_name', '')} {character_info.get('level', '')}레벨 {character_info.get('fame', '')}명성")
        
        # 캐릭터 정보 포맷팅 (LLM 프롬프트용)
        character_context = ""
        if character_info:
            character_context = f"""
사용자 캐릭터 정보:
- 직업: {character_info.get('class_name', '정보 없음')}
- 명성: {character_info.get('fame', '정보 없음')}

위 캐릭터 정보를 고려하여 맞춤형 조언을 제공하세요.
"""
        
        # 검색 수행 (캐릭터 정보 활용)
        search_results = self._smart_hybrid_search(query, character_info)
        
        # LLM 답변 생성
        llm_start_time = time.time()
        formatted_prompt = self.hybrid_prompt.format(
            internal_context=search_results["internal_context"],
            web_context=search_results["web_context"],
            question=query,
            character_info=character_context
        )
        
        response = self.llm.invoke(formatted_prompt).content
        
        llm_elapsed_time = time.time() - llm_start_time
        total_elapsed_time = time.time() - total_start_time
        
        print(f"[INFO] 답변 생성 완료: {total_elapsed_time:.2f}초")
        
        return {
            "result": response,
            "source_documents": search_results["all_docs"],
            "used_web_search": search_results["used_web_search"],
            "internal_docs": search_results["internal_docs"],
            "web_docs": search_results["web_docs"],
            "enhanced_query": search_results.get("enhanced_query", query),  # 디버깅용
            "execution_times": {
                "total": total_elapsed_time,
                "llm": llm_elapsed_time
            }
        }





# 전역 RAG 서비스 인스턴스
_rag_service = None

def get_rag_service() -> RAGService:
    """RAG 서비스 싱글톤 인스턴스 반환"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_rag_answer(query: str, character_info: Optional[Dict] = None):
    """RAG 답변 생성 (간편 함수)"""
    service = get_rag_service()
    return service.get_answer(query, character_info)
