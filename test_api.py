"""
FastAPI 테스트 코드
"""

import requests
import json
import time
from test_jwt import create_test_jwt_token

# API 서버 주소
BASE_URL = "http://localhost:8000"

def test_full_chat():
    """채팅 테스트 (캐릭터 정보 포함)"""
    print("\\n=== 전체 채팅 테스트 ===")
    
    # JWT 토큰 생성
    jwt_token = create_test_jwt_token()
    
    try:
        # 요청 데이터
        request_data = {
            "query": "내 명성에서 나벨 레이드에 가려면 어떻게 스펙업을 해야해?",
            "jwt_token": jwt_token,
            "character_summary": {
                "character_id": "12345",
                "character_name": "테스트캐릭터",
                "class_name": "인파이터",
                "fame": 40000
            }
        }
        
        print("캐릭터 정보:")
        print(f"  - 이름: {request_data['character_summary']['character_name']}")
        print(f"  - 직업: {request_data['character_summary']['class_name']}")
        print(f"  - 명성: {request_data['character_summary']['fame']}")
        
        print("\\n요청 전송 중...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        elapsed_time = time.time() - start_time
        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\\n성공: {result['success']}")
            print(f"답변: {result['answer']}")
            print(f"RAG 처리 시간: {result['execution_time']:.2f}초")
            print(f"웹 검색 사용: {result['used_web_search']}")
            
            print(f"\\n참고 출처 ({len(result['sources'])}개):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source['title']}")
                if source['url']:
                    print(f"     URL: {source['url']}")
            
            return True
        else:
            print(f"에러: {response.text}")
            return False
            
    except Exception as e:
        print(f"전체 채팅 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 FastAPI 테스트 시작\\n")
    
    # 서버가 실행되고 있는지 확인
    try:
        requests.get(BASE_URL, timeout=10)
    except:
        print("❌ API 서버가 실행되지 않았습니다!")
        print("다음 명령어로 서버를 먼저 실행하세요:")
        print("cd C:\\Develop\\Project\\6-AI")
        print("python api/main.py")
        return
    
    # 테스트 실행
    tests = [
        ("전체 채팅", test_full_chat)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # 테스트 간 간격
    
    # 결과 요약
    print("\\n" + "="*50)
    print("📋 테스트 결과 요약")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{test_name}: {status}")
    
    total_success = sum(1 for _, success in results if success)
    print(f"\\n총 {len(results)}개 테스트 중 {total_success}개 성공")

if __name__ == "__main__":
    main()
