from __future__ import annotations
import os, logging, time, hashlib, pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv

# LLM & 임베딩
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 내부 검색 공통
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Perplexity 웹 검색
from openai import OpenAI

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

load_dotenv()

# 캐시 디렉토리 생성
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ───────────────────────────────────────────────
# 환경 변수
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY   = os.getenv("PERPLEXITY_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수 필요!")

# ───────────────────────────────────────────────
# 내부 RAG 초기화
print("🚀 DF-RAG 시스템 초기화 중...")
init_start_time = time.time()

CHROMA_DIR   = "vector_db/chroma"
EMBED_MODEL  = "text-embedding-3-large"
embed_fn     = OpenAIEmbeddings(model=EMBED_MODEL)

print("🔄 벡터 DB 로딩...")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_fn)
vector_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.8},
)
print("✅ 벡터 DB 로딩 완료")

# BM25
def build_bm25_index():
    """BM25 인덱스를 구축하고 캐싱"""
    print("🔄 BM25 인덱스 구축 중...")
    store_data = vectordb.get(include=["documents", "metadatas"])
    docs_for_bm25 = []
    for txt, meta in zip(store_data["documents"], store_data["metadatas"]):
        # 메타데이터를 텍스트에 포함시켜 검색 품질 향상
        enhanced_content = txt
        if meta:
            # 제목이 있으면 강조
            if meta.get("title"):
                enhanced_content = f"제목: {meta['title']}\n{txt}"
            # 클래스명이 있으면 추가
            if meta.get("class_name"):
                enhanced_content += f"\n직업: {meta['class_name']}"
        
        docs_for_bm25.append(Document(page_content=enhanced_content, metadata=meta))
    
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 8
    
    # 캐싱 저장
    cache_file = CACHE_DIR / "bm25_index.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        print(f"✅ BM25 인덱스 캐시 저장: {cache_file}")
    except Exception as e:
        print(f"⚠️ BM25 캐시 저장 실패: {e}")
    
    return bm25_retriever

def load_bm25_index():
    """캐시에서 BM25 인덱스 로드하거나 새로 구축"""
    cache_file = CACHE_DIR / "bm25_index.pkl"
    
    # 캐시 만료 시간 (12시간)
    cache_expiry = 60 * 60 * 12
    
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
    
    # 캐시가 없거나 만료된 경우 새로 구축
    return build_bm25_index()

bm25_retriever = load_bm25_index()

# RRF → Ensemble
rrf_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

# Custom metadata-aware retriever wrapper
class MetadataAwareRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
    
    def get_relevant_documents(self, query):
        docs = self.base_retriever.get_relevant_documents(query)
        
        # 메타데이터 기반 점수 조정
        scored_docs = []
        for doc in docs:
            score = 1.0  # 기본 점수
            meta = doc.metadata or {}
            
            # 품질 점수 (priority_score, content_score 기준)
            if meta.get("priority_score"):
                try:
                    priority = float(meta["priority_score"])
                    score += priority * 0.1  # priority_score를 점수에 반영
                except:
                    pass
            
            if meta.get("content_score"):
                try:
                    content_score = float(meta["content_score"])
                    score += content_score * 0.01  # content_score를 점수에 반영
                except:
                    pass
            
            scored_docs.append((doc, score))
        
        # 점수순으로 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:8]]  # 상위 8개만 반환

# Cross-Encoder 재랭킹 with caching
def load_cross_encoder():
    """크로스 인코더 모델 로드 (캐시 활용)"""
    cache_file = CACHE_DIR / "cross_encoder.pkl"
    
    # 캐시 만료 시간 (24시간)
    cache_expiry = 60 * 60 * 24
    
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
    
    # 캐시가 없거나 만료된 경우 새로 로드
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

cross_encoder = load_cross_encoder()
compressor = CrossEncoderReranker(
    model=cross_encoder,
    top_n=6                             
)
base_retriever = ContextualCompressionRetriever(
    base_retriever=rrf_retriever,
    base_compressor=compressor,
)
internal_retriever = MetadataAwareRetriever(base_retriever)

init_elapsed_time = time.time() - init_start_time
print(f"🎉 시스템 초기화 완료! (소요시간: {init_elapsed_time:.2f}초)")
print("💬 질문을 입력하세요...\n")

# ───────────────────────────────────────────────
# 캐시 관련 함수
def generate_cache_key(query):
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def get_from_cache(query, cache_type='search'):
    cache_key = generate_cache_key(query)
    cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
    
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

def save_to_cache(query, result, cache_type='search'):
    cache_key = generate_cache_key(query)
    cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f"캐시 저장 중 오류: {e}")

# ───────────────────────────────────────────────
# Perplexity 웹 검색
def perplexity_web_search(query: str, max_results=3) -> List[Document]:
    cached_result = get_from_cache(query, 'web_search')
    if cached_result:
        print("🔄 캐시된 웹 검색 결과 사용")
        return cached_result
    
    if not PERPLEXITY_API_KEY:
        return []
    
    try:
        start_time = time.time()
        client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides information about "
                    "Dungeon & Fighter (DNF) game. Focus exclusively on providing the most "
                    "recent and accurate information about character progression, equipment, "
                    "and optimization guides. Extract only the key facts and essential details "
                    "from your search results. Prioritize information from 2025 sources. "
                    "Be concise and direct, omitting any unnecessary context or introduction. "
                    "Format your response as clear, actionable points whenever possible."
                )
            },
            {"role": "user", "content": f"2025 최신 던전앤파이터 {query} 명성별 스펙업 가이드"}
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
        
        citations = []
        if hasattr(response, "citations"):
            citations = response.citations
        elif hasattr(response.choices[0].message, "context") and hasattr(response.choices[0].message.context, "citations"):
            citations = response.choices[0].message.context.citations
            
        if citations:
            for citation in citations[:max_results]:
                if isinstance(citation, dict):
                    url = citation.get("url", "")
                    text = citation.get("text", "")
                else:
                    url = str(citation)
                    text = ""
                    
                # URL이 있는 경우만 추가
                if url and url != "링크 없음":
                    docs.append(Document(
                        page_content=text,
                        metadata={"title": url, "url": url, "source": "web_search"}
                    ))
        
        save_to_cache(query, docs, 'web_search')
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ 웹 검색 실행 시간: {elapsed_time:.2f}초")
        
        return docs
            
    except Exception as e:
        print(f"❌ 웹 검색 오류: {e}")
        return []

# ───────────────────────────────────────────────
# 스마트 하이브리드 검색
def smart_hybrid_search(query):
    start_time = time.time()
    
    cached_result = get_from_cache(query, 'hybrid_search')
    if cached_result:
        print("[CACHE] 캐시된 하이브리드 검색 결과 사용")
        return cached_result
    
    def get_internal_results():
        try:
            docs = internal_retriever.get_relevant_documents(query)
            print(f"[DEBUG] 내부 검색 결과: {len(docs)}개 문서 검색됨")
            return docs
        except Exception as e:
            print(f"[ERROR] 내부 검색 오류: {e}")
            return []
    
    def get_web_results():
        try:
            results = perplexity_web_search(query)
            print(f"[DEBUG] 웹 검색 결과: {len(results)}개 문서 검색됨")
            return results
        except Exception as e:
            print(f"[ERROR] 웹 검색 오류: {e}")
            return []
    
    # 병렬 실행 (항상 웹검색 포함)
    with ThreadPoolExecutor(max_workers=2) as executor:
        internal_future = executor.submit(get_internal_results)
        web_future = executor.submit(get_web_results)
        
        internal_docs = internal_future.result()
        web_docs = web_future.result()
    
    # 결과 컨텍스트 구성 (URL 정보 포함)
    internal_context_parts = []
    for i, doc in enumerate(internal_docs):
        content = f"[내부 문서 {i+1}] {doc.page_content}"
        # URL 정보가 있으면 추가
        if doc.metadata and doc.metadata.get("url"):
            content += f"\n참고 링크: {doc.metadata['url']}"
        internal_context_parts.append(content)
    
    internal_context = "\n\n".join(internal_context_parts)
    
    web_context = "\n\n".join([
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
        "used_web_search": len(web_docs) > 0
    }
    
    save_to_cache(query, result, 'hybrid_search')
    
    elapsed_time = time.time() - start_time
    print(f"[TIME] 하이브리드 검색 실행 시간: {elapsed_time:.2f}초")
    
    return result

# ───────────────────────────────────────────────
# LLM & 프롬프트 (Gemini 2.5 Flash)
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="models/gemini-2.5-flash-preview-05-20",
    temperature=0,
)

# 프롬프트 템플릿
hybrid_prompt = PromptTemplate(
    input_variables=["internal_context", "web_context", "question"],
    template="""
당신은 던전앤파이터 전문 스펙업 가이드 챗봇입니다.
중요: 반드시 제공된 정보만 사용하여 답변하세요. 제공된 정보에 없는 내용은 "해당 정보를 찾을 수 없습니다"라고 답변하세요.

다음 두 가지 정보 소스를 활용하여 사용자 질문에 답변하세요:

1. 내부 데이터베이스 정보 (주요 정보원 - 기존 가이드 및 커뮤니티 정보):
{internal_context}

2. 웹 검색 정보 (보조 정보원 - 최신 업데이트 및 추가 정보):
{web_context}

반드시 지켜야 할 규칙:
1. 제공된 정보 소스에 없는 내용은 절대 답변하지 마세요.
2. 자체 지식이나 과거 데이터를 사용하지 마세요.
3. 정보가 부족한 경우 "제공된 정보에서 해당 내용을 찾을 수 없습니다"라고 정직하게 답변하세요.
4. 중요한 정보는 어떤 소스에서 가져왔는지 표시하세요(예: "내부 문서에 따르면..." 또는 "웹 검색 결과에 따르면...").

정보 처리 지침:
- 우선순위: 내부 데이터베이스 정보를 주요 정보원으로 사용하고, 웹 검색 정보는 보조적으로 활용하세요.
- 일관성: 정보 소스 간에 충돌이 있으면 최신 정보(2025년)를 우선시하세요.
- 명확성: 확실한 정보와 불확실한 정보를 구분하여 전달하세요.
- 간결성: 불필요한 서론 없이 핵심 정보만 요약해서 간략하게 전달하세요.
- 구체성: 스펙업 추천은 구체적인 단계와 이유를 포함해야 합니다.

응답 형식 지침:
- 사용자가 묻지 않은 내용은 설명하지 마세요.
- 직업이 언급되면 해당 직업에 맞는 정보를 우선 제공하세요.
- 2025년 최신 패치 내용을 우선적으로 반영하세요.
- 던파 관련 API나 공식 정보 소스가 있다면 적절히 추천하세요.

이벤트 관련 답변 시 주의사항:
1. 이벤트 종료일이 현재 날짜(2025-05-22) 이후인 경우에만 이벤트 참여를 권장하세요
2. 이벤트가 이미 종료된 경우 "해당 이벤트는 종료되었습니다"라고 명시하세요
3. 이벤트 종료일 정보가 없는 경우 "이벤트 종료일을 확인해주세요"라고 안내하세요

사용자 질문: {question}

답변:
"""
)

# ───────────────────────────────────────────────
# 통합 응답 생성 함수
def get_answer(query):
    total_start_time = time.time()
    
    search_results = smart_hybrid_search(query)
    
    llm_start_time = time.time()
    
    # 프롬프트 템플릿 적용
    formatted_prompt = hybrid_prompt.format(
        internal_context=search_results["internal_context"],
        web_context=search_results["web_context"],
        question=query
    )
    
    response = llm.invoke(formatted_prompt).content
    
    llm_elapsed_time = time.time() - llm_start_time
    total_elapsed_time = time.time() - total_start_time
    
    print(f"⏱️ LLM 응답 생성 시간: {llm_elapsed_time:.2f}초")
    print(f"⏱️ 전체 실행 시간: {total_elapsed_time:.2f}초")
    
    return {
        "result": response,
        "source_documents": search_results["all_docs"],
        "used_web_search": search_results["used_web_search"],
        "internal_docs": search_results["internal_docs"], 
        "web_docs": search_results["web_docs"],
        "execution_times": {
            "total": total_elapsed_time,
            "llm": llm_elapsed_time
        }
    }

# ───────────────────────────────────────────────
# 8️⃣ 콘솔 채팅 함수
def ask_once(q: str):
    result = get_answer(q)
    return result["result"]

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="DF Console RAG")
    parser.add_argument("-q", "--query", type=str, help="한 번만 질문하고 종료")
    args = parser.parse_args()

    if args.query:
        print(ask_once(args.query))
        sys.exit()

    # 대화형 루프
    print("💬 DF-RAG 콘솔 챗 (exit 입력 시 종료)")
    while True:
        try:
            user_in = input("\n▶︎ 질문: ").strip()
            if not user_in or user_in.lower().startswith("exit"):
                break
            print("🧠 thinking …")
            
            # 상세 결과 출력
            result = get_answer(user_in)
            
            # 검색 소스 정보 출력 (항상 웹검색 포함)
            print("\n✅ 내부 DB + 웹 검색 사용")
            
            print("\n답변:")
            print(result["result"])
            
            print(f"\n소요 시간: {result['execution_times']['total']:.2f}초 (LLM: {result['execution_times']['llm']:.2f}초)")
            
            # 출처 정보 출력
            print("\n출처:")
            
            for doc in result["web_docs"]:
                title = doc.metadata.get('title', '')
                if title == "Perplexity 검색 결과":
                    print(f"🌐 {title}")
                elif title.startswith('http'):  # URL인 경우
                    print(f"🌐 {title}")
                else:
                    print(f"🌐 {title}")
                
            for doc in result["internal_docs"][:3]:
                title = doc.metadata.get('title', '제목 없음')
                url = doc.metadata.get('url', '') or doc.metadata.get('source', '')
                if url:
                    print(f"🗄️ {title} - {url}")
                else:
                    print(f"🗄️ {title}")
                
        except (KeyboardInterrupt, EOFError):
            break
