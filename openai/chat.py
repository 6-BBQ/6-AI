import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv(find_dotenv())

# 환경변수에서 API 키 읽기
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 응답 생성
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "한글로 정확히 10글자만 생성해줘. 다른 설명이나 문장은 포함하지 마."
        }
    ],
    temperature=0.7,
    max_tokens=30
)

# JSON 형태로 전체 응답 출력
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))