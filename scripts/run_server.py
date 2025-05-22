#!/usr/bin/env python3
"""
FastAPI 서버 실행 스크립트
"""
import uvicorn
import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 FastAPI 서버 시작...")
    print(f"📁 프로젝트 루트: {project_root}")
    print("🌐 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print()
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # 개발 모드: 코드 변경시 자동 재시작
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 서버를 종료합니다.")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
