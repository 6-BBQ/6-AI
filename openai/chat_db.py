import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
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

# 2. 벡터 DB 로드 (사전에 저장된 Chroma DB)
persist_directory = "./chroma_db"  # 이미 크롤링+임베딩된 결과가 저장된 폴더
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

# 3. 사용자 질문
user_question = "버서커 캐릭터 스펙업을 어떻게 하면 좋을까요?"

# 4. 유사도 기반 문서 검색
retrieved_docs = vectorstore.similarity_search(user_question, k=3)
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 5. 프롬프트 생성
template = PromptTemplate.from_template(
    """당신은 던파 캐릭터 스펙 분석 전문가입니다. 아래는 해당 캐릭터의 정보와 참고 문서 일부입니다. 
이 내용을 바탕으로 사용자의 질문에 대해 정확하고 전문적인 답변을 제공하세요. 
단, 참고 내용에 얽매이지 말고 당신의 지식을 활용해 보완해 주세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
"""
)
final_prompt = template.format(context=context_text, question=user_question)

# 6. ChatCompletion 호출
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 던전앤파이터 캐릭터 분석과 스펙업 조언을 잘하는 AI 도우미입니다."},
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.7,
    max_tokens=100
)

# 7. 응답 출력
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
