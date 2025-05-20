import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer


# 1. 환경 변수 로드 및 API 키 설정
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# LangSmith 설정
os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# 트레이서 및 콜백 매니저 생성
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)

# 2. 웹 페이지 로드
loader = PlaywrightURLLoader(["https://lilianweng.github.io/posts/2023-06-23-agent/"])
docs = loader.load()

# 3. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 4. vectorstore 생성할 때
vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())

# 5. 사용자 질문
user_question = "Agent 시스템이란 무엇인가요?"

# 6. 관련 문서 검색
retrieved_docs = vectorstore.similarity_search(user_question, k=3)
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 7. 프롬프트 생성
template = PromptTemplate.from_template(
    """당신은 인공지능 전문가입니다. 아래는 참고 문서 일부입니다. 
이 내용을 바탕으로 질문에 대해 정확하고 전문적인 답변을 제공하세요. 
단, 참고 내용에 얽매이지 말고 전문적 지식으로 보완해 주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
"""
)
final_prompt = template.format(context=context_text, question=user_question)

# 8. ChatCompletion 호출
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 인공지능 시스템에 대한 전문적인 지식을 갖춘 도우미입니다."},
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.7,
    max_tokens=100
)

# 9. 응답 출력
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
