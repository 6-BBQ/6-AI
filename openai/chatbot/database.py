from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# ChromaDB 클라이언트 초기화
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="chatbot_leads")
    print("ChromaDB connected successfully!")
except Exception as e:
    print(f"ChromaDB connection error: {e}")
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="chatbot_leads")
    print("ChromaDB using in-memory fallback")

# 리드 저장 함수
def save_lead(character_name: str, user_message: str, service: str) -> bool:
    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        doc_id = f"{character_name}_{timestamp}"

        collection.add(
            documents=[user_message],
            metadatas=[{
                "character_name": character_name,
                "service": service,
                "timestamp": timestamp
            }],
            ids=[doc_id]
        )
        print(f"Lead saved: {doc_id}")
        return True
    except Exception as e:
        print(f"Error saving lead: {e}")
        return False

# 최근 대화 불러오기
def get_recent_conversations(limit: int = 10) -> List[Dict[str, Any]]:
    try:
        data = collection.peek(limit * 10)
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        conversations: Dict[tuple, List[Dict[str, Any]]] = {}
        for doc_id, doc, meta in zip(ids, docs, metadatas):
            key = (meta.get("character_name", ""), meta.get("service", ""))
            conversations.setdefault(key, []).append({
                "id": doc_id,
                "message": doc,
                "timestamp": meta.get("timestamp", ""),
                "role": meta.get("role", "없음")  # 역할 포함
            })

        # 메시지 정렬
        for msgs in conversations.values():
            msgs.sort(key=lambda x: x["timestamp"])

        # 대화 정리 후 최근순 정렬
        conv_list = [{
            "character_name": character_name,
            "service": service,
            "messages": msgs
        } for (character_name, service), msgs in conversations.items()]

        conv_list.sort(key=lambda x: x["messages"][-1]["timestamp"], reverse=True)
        return conv_list[:limit]
    except Exception as e:
        print(f"Error retrieving conversations: {e}")
        return []

# 전체 메시지 불러오기
def get_all_messages() -> List[Dict[str, Any]]:
    try:
        data = collection.peek(10000)
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        messages = [{
            "id": doc_id,
            "message": doc,
            "character_name": meta.get("character_name", ""),
            "service": meta.get("service", ""),
            "timestamp": meta.get("timestamp", ""),
            "role": meta.get("role", "없음")  # 역할 포함
        } for doc_id, doc, meta in zip(ids, docs, metadatas)]

        messages.sort(key=lambda x: x["timestamp"])
        return messages
    except Exception as e:
        print(f"Error retrieving all messages: {e}")
        return []

# 메시지 저장 함수
def save_message(message: str, role: str = "user", session: Optional[dict] = None) -> None:
    """
    message: 저장할 메시지 내용
    role: 'user' 또는 'assistant'
    session: 현재 세션 정보 (character_name, service 등이 포함됨)
    """
    if session is None:
        print("Session 정보 없음으로 메시지 저장하지 않음")
        return

    character_name = session.get("character_name")
    service = session.get("service")

    if not character_name or not service:
        print(f"Session 정보 부족으로 저장하지 않음: character_name={character_name}, service={service}")
        return

    try:
        timestamp = datetime.now().isoformat(timespec="seconds")
        doc_id = f"{character_name}_{timestamp}_{role}"

        collection.add(
            documents=[message],
            metadatas=[{
                "character_name": character_name,
                "service": service,
                "timestamp": timestamp,
                "role": role
            }],
            ids=[doc_id]
        )
        print(f"Message saved: {doc_id}")
    except Exception as e:
        print(f"Error saving message: {e}")
