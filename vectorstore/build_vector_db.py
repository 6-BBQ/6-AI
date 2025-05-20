import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import json
from pathlib import Path

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키를 환경 변수에서 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    sys.exit(1)

PROCESSED_JSON_PATH = "data/processed/processed_docs.json"
CHROMA_DIR = "vector_db/chroma"
BATCH_SIZE = 500  # 한번에 처리할 문서 개수

def load_documents_from_json(json_path):
    """JSON 파일에서 문서 로드"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=doc["text"], metadata=doc["metadata"]) for doc in data]
    except FileNotFoundError:
        print(f"오류: {json_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"오류: {json_path} 파일이 올바른 JSON 형식이 아닙니다.")
        sys.exit(1)

def build_chroma_db():
    """문서에서 벡터 데이터베이스 구축"""
    print("📄 전처리된 문서 로드 중...")
    documents = load_documents_from_json(PROCESSED_JSON_PATH)
    print(f"✅ {len(documents)}개 문서 로드 완료. 임베딩 생성 중...")

    # 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings()
    db = None

    # 배치 단위로 처리
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]
        print(f"▶️ {i} ~ {i+len(batch)-1}번 문서 임베딩 중...")

        try:
            if db is None:
                # 첫 번째 배치로 DB 초기화
                db = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_model,
                    persist_directory=CHROMA_DIR
                )
            else:
                # 이후 배치 추가
                db.add_documents(batch)
            
            # 각 배치 처리 후 저장
            db.persist()
            print(f"  ✓ 배치 저장 완료")
            
        except Exception as e:
            print(f"❌ 임베딩 생성 중 오류 발생: {e}")
            sys.exit(1)

    print(f"\n✅ Chroma DB 저장 완료 → {CHROMA_DIR}")

if __name__ == "__main__":
    build_chroma_db()