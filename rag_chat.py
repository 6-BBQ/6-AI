import os
import requests
import json
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from openai import OpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 환경 변수에서 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not GEMINI_API_KEY:
    print("오류: GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    sys.exit(1)

if not PERPLEXITY_API_KEY:
    print("경고: PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다. 웹 검색 기능이 작동하지 않을 수 있습니다.")

# 벡터 DB 경로
CHROMA_DIR = "vector_db/chroma"

# 1. 벡터 DB 로드
embedding_model = GoogleGenerativeAIEmbeddings(
        google_api_key=GEMINI_API_KEY,
        model="models/text-embedding-004"
    )
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

# 2. LLM 설정
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="models/gemini-2.5-flash-preview-05-20", temperature=0)

# 3. 내부 RAG 검색기 설정
internal_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 12,
        "lambda_mult": 0.7
    }
)

# 4. Perplexity API를 사용한 웹 검색 함수
def perplexity_web_search(query, max_results=3):
    api_key = PERPLEXITY_API_KEY
    
    if not api_key:
        return []
    
    try:
        client = OpenAI(
            api_key=api_key,
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
            {"role": "user", "content": f"2025 최신 던전앤파이터 {query} 스펙업 가이드"}
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
                    title = citation.get("title", "제목 없음")
                    url = citation.get("url", "링크 없음")
                    text = citation.get("text", "")
                else:
                    title = str(citation)
                    url = "링크 없음"
                    text = ""
                docs.append(Document(
                    page_content=f"{title}\n{text}",
                    metadata={"title": title, "url": url, "source": "web_search"}
                ))
                
        return docs
            
    except Exception as e:
        # 오류 발생 시 조용히 빈 리스트 반환
        return []

# 5. 하이브리드 검색 함수
def hybrid_search(query):
    """내부 RAG와 웹 검색을 항상 모두 사용하는 하이브리드 검색"""
    # 내부 RAG에서 정보 검색
    internal_docs = internal_retriever.get_relevant_documents(query)
    
    # 웹 검색 실행
    web_docs = perplexity_web_search(query)
    
    # 문맥 구성
    internal_context = "\n\n".join([doc.page_content for doc in internal_docs])
    web_context = "\n\n".join([doc.page_content for doc in web_docs]) if web_docs else ""
    
    return {
        "all_docs": internal_docs + web_docs,
        "internal_docs": internal_docs,
        "web_docs": web_docs,
        "internal_context": internal_context,
        "web_context": web_context,
        "used_web_search": len(web_docs) > 0
    }

# 6. 하이브리드 프롬프트 템플릿
hybrid_prompt = PromptTemplate(
    input_variables=["internal_context", "web_context", "question"],
    template="""
당신은 던전앤파이터 전문 스펙업 가이드 챗봇입니다.
다음 두 가지 정보 소스를 활용하여 사용자 질문에 답변하세요.

1. 내부 데이터베이스 정보 (기존 가이드 및 커뮤니티 정보):
{internal_context}

2. 웹 검색 정보 (최신 업데이트 및 추가 정보):
{web_context}

정보 가중치 지침:
- 웹 검색 정보에 50%, 내부 데이터베이스 정보에 50%의 가중치를 두고 응답을 구성하세요.
- 두 정보 소스가 상충할 경우 최신 정보를 우선하세요.
- 웹 검색 결과가 부족하거나 없는 경우에도 최선을 다해 내부 데이터베이스 정보를 활용하세요.

응답 형식 지침:
- 2025년 이전의 데이터는 의미가 없으니 참조하지 마세요.
- 불필요한 서론이나 배경 설명 없이 핵심 정보만 전달하세요.
- 스펙업 순서나 우선순위를 제시할 때는 단계별로 명확하게 안내하세요.
- 1,2,3 이런식으로 순서를 통해 우선순위를 제시하세요.
- 신뢰성이 높은 정보는 더 상세하게 제공하세요.
- 신뢰성이 떨어지는 정보는 명칭을 생략하는 식으로 추상화하여 짧게 제공하세요.
- 캐릭터 직업을 말하지 않는 경우엔 공통적인 내용만 설명하세요.
- 상황에 따라 던파 API 사이트를 적절히 추천해주는 방식을 채용하세요.

사용자 질문: {question}

답변:
"""
)

# 7. 하이브리드 체인 설정
hybrid_chain = LLMChain(llm=llm, prompt=hybrid_prompt)

# 8. 통합 응답 생성 함수
def get_answer(query):
    # 하이브리드 검색 실행
    search_results = hybrid_search(query)
    
    # 하이브리드 방식 사용
    response = hybrid_chain.run(
        internal_context=search_results["internal_context"],
        web_context=search_results["web_context"],
        question=query
    )
    
    return {
        "result": response,
        "source_documents": search_results["all_docs"],
        "used_web_search": search_results["used_web_search"],
        "internal_docs": search_results["internal_docs"], 
        "web_docs": search_results["web_docs"]
    }

# 9. 명령행 인수로 질문을 받아 한 번만 답변하는 메인 함수
def main():
    # 명령행 인수로 질문을 받음
    if len(sys.argv) > 1:
        # 명령행 인수로 전달된 질문 사용
        query = " ".join(sys.argv[1:])
    else:
        # 명령행 인수가 없으면 표준 입력에서 한 줄 읽기
        print("던파 스펙업 가이드 챗봇\n")
        query = input("질문: ")
    
    if not query or query.lower() in ['exit', 'quit', '종료']:
        print("질문이 없습니다.")
        return
    
    # 응답 생성
    result = get_answer(query)
    
    # 웹 검색 성공/실패 여부 출력
    if result["used_web_search"]:
        print("\n✅ 웹 검색 및 내부 DB 사용")
    else:
        print("\n⚠️ 웹 검색 실패 - 내부 DB만 사용")
    
    # 결과 출력
    print("\n답변:")
    print(result["result"])
    
    # 출처 정보 출력
    print("\n출처:")
    
    # 출처 문서 출력 (웹 검색 결과 먼저, 그 다음 내부 DB)
    for doc in result["web_docs"]:
        print(f"🌐 {doc.metadata.get('title', '제목 없음')} ({doc.metadata.get('url', '링크 없음')})")
        
    for doc in result["internal_docs"][:3]:  # 내부 DB는 상위 3개만 표시
        print(f"🗄️ {doc.metadata.get('title', '제목 없음')} ({doc.metadata.get('url', '링크 없음')})")

if __name__ == "__main__":
    main()
