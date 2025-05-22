#!/usr/bin/env python3
"""
API 테스트 실행 스크립트
"""
import sys
import time  
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🧪 API 테스트 시작...")
    print("=" * 60)
    
    try:
        # test.py 실행
        print("📋 테스트 파일 실행 중...")
        import test
        
    except ModuleNotFoundError as e:
        print(f"❌ 모듈을 찾을 수 없습니다: {e}")
        print("💡 먼저 필요한 패키지를 설치하세요:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)
    print("✅ 테스트 완료!")

if __name__ == "__main__":
    main()
