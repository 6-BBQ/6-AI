import os
import socketio
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from database import save_message
import asyncio

# .env 파일 로드
load_dotenv(find_dotenv())

api_key = os.getenv("OPEN_API_KEY")
if not api_key:
    raise RuntimeError("OPEN_API_KEY not set in .env")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
chat_app = socketio.ASGIApp(sio, other_asgi_app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

system_prompt = (
    """당신은 던파 캐릭터 스펙 분석 전문가입니다. 아래는 해당 캐릭터의 정보와 참고 문서 일부입니다. 
이 내용을 바탕으로 사용자의 질문에 대해 정확하고 전문적인 답변을 제공하세요. 
단, 참고 내용에 얽매이지 말고 당신의 지식을 활용해 보완해 주세요."""
)

SERVICES = [
    "딜러 스펙 분석", "버퍼 스펙 분석", "에픽 장비 추천", "던전 추천", "골드 수급 방법",
    "캐릭터 육성 루트", "세팅 시뮬레이션", "레이드 공략 요약"
]

client = OpenAI(api_key=api_key)

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    sessions[sid] = {"step": "intent", "history": []}

@sio.event
async def disconnect(sid, environ):
    print(f"Client disconnected: {sid}")
    sessions.pop(sid, None)

async def classify_and_detect_service(message: str) -> str:
    classification_prompt = (
        "당신은 사용자가 요청한 서비스 유형을 식별하는 AI 어시스턴트입니다. "
        f"사용자의 요청을 다음 중 하나로 분류하세요: {', '.join(SERVICES)}."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=50,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "none"

async def analyze_character(character_name: str, question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"캐릭터명: {character_name}\n질문: {question}"},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI 분석 오류: {e}")
        return "분석 중 오류가 발생했습니다. 다시 시도해 주세요."

@sio.event
async def message(sid, data):
    print("Received message from client:", sid, data)
    session = sessions.get(sid, {})
    step = session.get("step", "intent")
    history = session.setdefault("history", [])

    # 저장: 사용자 메시지
    save_message(data, role="user", session=session)

    END_KEYWORDS = ["끝", "그만", "고마워", "아니요", "아닙니다", "괜찮아요", "종료"]
    if any(kw in data.lower() for kw in END_KEYWORDS):
        await sio.emit("bot_message", "대화를 종료합니다. 감사합니다.", room=sid)
        sessions.pop(sid, None)
        return

    if step == "intent":
        service = await classify_and_detect_service(data)
        if service != "none":
            session["service"] = service
            session["step"] = "collect_name"
            bot_reply = f"좋습니다! '{service}'에 대해 도와드릴 수 있습니다. 캐릭터명을 알려주세요."
        else:
            bot_reply = "죄송해요. 요청하신 내용을 이해하지 못했어요. 다시 한 번 말씀해 주세요."

    elif step == "collect_name":
        session["character_name"] = data
        session["step"] = "await_question"
        bot_reply = f"'{data}' 캐릭터에 대해 어떤 점이 궁금하신가요?"

    elif step == "await_question":
        character_name = session.get("character_name")
        service = session.get("service")
        question = data

        bot_reply_intro = f"'{character_name}' 캐릭터의 '{service}'에 대해 분석 중입니다. 잠시만 기다려주세요!"
        await sio.emit("bot_message", bot_reply_intro, room=sid)

        analysis = await analyze_character(character_name, question)
        save_message(analysis, role="assistant", session=session)
        await sio.emit("bot_message", analysis, room=sid)
        return

    await sio.emit("bot_message", bot_reply, room=sid)
    save_message(bot_reply, role="assistant", session=session)

    sessions[sid] = session
