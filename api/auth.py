import os
import jwt
import base64
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import HTTPException, status
from dotenv import load_dotenv

load_dotenv()

class JWTAuth:
    """JWT 인증 클래스"""
    
    def __init__(self):
        # JWT 시크릿 키 (스프링과 동일한 키 사용)
        secret_key_b64 = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
        # Base64 디코딩
        try:
            self.secret_key = base64.b64decode(secret_key_b64).decode('utf-8')
        except:
            self.secret_key = secret_key_b64  # 디코딩 실패시 원본 사용
        self.algorithm = "HS256"
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        JWT 토큰 검증
        
        Args:
            token: JWT 토큰 문자열
            
        Returns:
            토큰에서 추출한 페이로드
            
        Raises:
            HTTPException: 토큰이 유효하지 않은 경우
        """
        try:
            # Bearer 접두사 제거
            if token.startswith("Bearer "):
                token = token[7:]
            
            # 토큰 디코딩
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 만료 시간 확인
            exp = payload.get("exp")
            if exp:
                exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="토큰이 만료되었습니다"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰이 만료되었습니다"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 토큰입니다"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"토큰 검증 실패: {str(e)}"
            )
    
    def get_user_id(self, token: str) -> str:
        """
        토큰에서 사용자 ID 추출
        
        Args:
            token: JWT 토큰
            
        Returns:
            사용자 ID
        """
        payload = self.verify_token(token)
        user_id = payload.get("sub") or payload.get("user_id") or payload.get("userId")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="토큰에서 사용자 ID를 찾을 수 없습니다"
            )
        
        return str(user_id)
    
    def get_user_info(self, token: str) -> Dict[str, Any]:
        """
        토큰에서 사용자 정보 추출
        
        Args:
            token: JWT 토큰
            
        Returns:
            사용자 정보 딕셔너리
        """
        payload = self.verify_token(token)
        
        return {
            "user_id": self.get_user_id(token),
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "issued_at": payload.get("iat"),
            "expires_at": payload.get("exp")
        }

# 전역 JWT 인증 인스턴스
jwt_auth = JWTAuth()

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    JWT 토큰 검증 함수 (의존성 주입용)
    
    Args:
        token: JWT 토큰
        
    Returns:
        사용자 정보
    """
    return jwt_auth.get_user_info(token)
