from database import get_all_messages, get_recent_conversations

def main():
    print("🔍 저장된 전체 대화 메시지 조회:\n")
    messages = get_all_messages()

    if not messages:
        print("⚠️ 저장된 메시지가 없습니다.")
        return

    print("===== 전체 대화 메시지 =====")
    for msg in messages:
        print(f"ID: {msg['id']}")
        print(f"역할: {msg.get('role', '없음')}")
        print(f"내용: {msg['message']}")
        print(f"캐릭터명: {msg.get('character_name', '없음')}")
        print(f"서비스: {msg.get('service', '없음')}")
        print(f"타임스탬프: {msg.get('timestamp', '없음')}")
        print("-" * 40)

    print("\n===== 캐릭터명 & 서비스별 최근 대화 목록 =====")
    recent_convs = get_recent_conversations(limit=5)
    if not recent_convs:
        print("최근 대화 목록이 없습니다.")
        return

    for conv in recent_convs:
        print(f"캐릭터명: {conv['character_name']} / 서비스: {conv['service']}")
        for msg in conv['messages']:
            print(f"  - [{msg['timestamp']}] ({msg.get('role', '없음')}) {msg['message']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
