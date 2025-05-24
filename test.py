import requests
from test_jwt import create_test_jwt_token # 이 함수가 있다고 가정

# 서버 주소
API_URL = "http://localhost:8000/api/df/chat" # main.py의 prefix와 endpoints.py 라우터 경로

# 테스트용 JWT 토큰
jwt_token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0IiwiYXV0aCI6IlVTRVIiLCJleHAiOjE3NDc5Mzc5MjR9.qMN9sGnxst0Z6bWPWBhY9NGADCOucriWAdmutbuLLjM"

# 테스트용 캐릭터 정보 (파이썬 딕셔너리)
character_info = {
  "jobGrowName": "眞 레인저",
  "jobName": "거너(여)",
  "fame": "73800",
  "weaponEquip": {
    "slotName": "무기",
    "itemRarity": "태초"
  },
  "epicNum": 9,
  "originalityNum": 1,
  "titleName": "순백의 눈꽃 결정[35Lv]",
  "setItemInfoAI": [
    {
      "setItemName": "칠흑의 정화 : 균형 세트",
      "setItemRarityName": "태초"
    }
  ],
  "creatureName": "SD 흰 구름 전령 에를리히",
  "auraName": "삼신기의 불꽃"
}

# 테스트용 쿼리
query = "가장 최근 이벤트에는 뭐가 있어?."

# 요청 데이터 구성 (character_data로 변경)
payload = {
    "query": query,
    "jwt_token": jwt_token,
    "character_data": character_info
}

# POST 요청 보내기
print("🚀 API 테스트 시작...")
print(f"📡 요청 URL: {API_URL}")
print(f"❓ 질문: {query}")
print(f"👤 캐릭터: {character_info.get('jobGrowName', 'N/A')} ({character_info.get('fame', 'N/A')}명성)")
print()

# requests.post가 payload를 올바른 JSON으로 변환하여 전송합니다.
headers = {'Content-Type': 'application/json'} # 명시적으로 헤더 추가 (선택 사항이지만 권장)

try:
    response = requests.post(API_URL, json=payload, headers=headers, timeout=120)  # 2분 타임아웃
    
    # 결과 출력
    print("📊 응답 결과:")
    print(f"📡 상태 코드: {response.status_code}")
    
    if response.status_code == 200:
        try:
            result = response.json()
            print("✅ 성공!")
            print()
            print("📝 RAG 답변:")
            print("-" * 50)
            print(result.get('answer', '답변을 찾을 수 없습니다.'))
            print("-" * 50)
            print()
            
            # 추가 정보 출력
            if result.get('execution_time'):
                print(f"⏱️  실행 시간: {result['execution_time']:.2f}초")
            if result.get('used_web_search'):
                print("🌐 웹 검색 사용됨")
            if result.get('sources'):
                print(f"📚 참고 출처: {len(result['sources'])}개")
                
            # 디버깅 정보 (있는 경우만 출력)
            if result.get('enhanced_query'):
                print(f"🔍 강화된 쿼리: {result['enhanced_query']}")
                
        except requests.exceptions.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            print("📥 원본 응답:")
            print(response.text)
    else:
        print(f"❌ API 오류 (상태 코드: {response.status_code})")
        print("📥 오류 내용:")
        try:
            error_data = response.json()
            print(error_data)
        except:
            print(response.text)
            
except requests.exceptions.Timeout:
    print("⏰ 요청 시간 초과 (2분)")
except requests.exceptions.ConnectionError:
    print("🔌 서버 연결 실패 - 서버가 실행 중인지 확인하세요.")
except Exception as e:
    print(f"💥 예상치 못한 오류: {e}")