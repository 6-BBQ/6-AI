import os
import base64
import jwt
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv

def create_jwt_token(target_file: str = "test.py"):
    """유효한 JWT 토큰을 생성하고 test.py에 삽입"""

    load_dotenv()

    # 1. 시크릿 키 로드 및 디코딩
    b64_key = os.getenv("JWT_SECRET_KEY")
    if not b64_key:
        raise ValueError("❌ .env에 JWT_SECRET_KEY가 없습니다")

    try:
        secret_key = base64.b64decode(b64_key).decode("utf-8")
    except Exception:
        secret_key = b64_key

    # 2. 현재 시간 기준 토큰 생성
    now = datetime.now(UTC)
    exp = now + timedelta(hours=2)

    payload = {
        "sub": "test",
        "auth": "USER",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp())
    }

    token = jwt.encode(payload, secret_key, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    print("✅ 생성된 토큰:", token)

    return token

