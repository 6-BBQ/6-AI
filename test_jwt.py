import jwt
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

def create_test_jwt_token(user_id: str = "test_user", username: str = "테스트유저", expires_in_hours: int = 24):
    """
    테스트용 JWT 토큰 생성
    
    Args:
        user_id: 사용자 ID
        username: 사용자명
        expires_in_hours: 만료 시간(시간)
        
    Returns:
        JWT 토큰 문자열
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-12345")
    
    # 토큰 페이로드
    payload = {
        "sub": user_id,  # subject (사용자 ID)
        "user_id": user_id,
        "username": username,
        "email": f"{user_id}@test.com",
        "roles": ["USER"],
        "iat": datetime.now(timezone.utc),  # issued at
        "exp": datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)  # expires at
    }
    
    # JWT 토큰 생성
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

if __name__ == "__main__":
    # 테스트 토큰 생성
    token = create_test_jwt_token()
    print("🔑 테스트용 JWT 토큰:")
    print(token)
    print()
    
    # 토큰 정보 확인
    import jwt as jwt_lib
    decoded = jwt_lib.decode(token, options={"verify_signature": False})
    print("📋 토큰 정보:")
    for key, value in decoded.items():
        if key in ['iat', 'exp']:
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
            print(f"  {key}: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            print(f"  {key}: {value}")