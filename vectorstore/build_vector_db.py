from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1️⃣ 설정
PROCESSED_PATH = Path("data/processed_docs.jsonl")
CHROMA_DIR = "vector_db/chroma"        # persist 디렉터리
BATCH_SIZE = 200                       # 임베딩 배치 처리와 맞춰서 증가
MODEL_NAME = "text-embedding-3-large"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_vector_db")

# ─────────────────────────────────────────────────────────────
# 2️⃣ 임베딩 & DB 초기화
log.info("🚀 임베딩 기반 벡터 DB 구축 시작 (한국어 성능 최적화)")

# 임베딩 함수 초기화
embedding_fn = OpenAIEmbeddings(
    model=MODEL_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 먼저 기존 폴더 제거
if Path(CHROMA_DIR).exists():
    log.info("🧹 기존 Chroma 폴더 삭제 후 재생성")
    import shutil
    shutil.rmtree(CHROMA_DIR)

# 이후 새로 생성
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_fn,
)

# ─────────────────────────────────────────────────────────────
# 3️⃣ JSONL → Document 리스트
def load_docs(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw: Dict = json.loads(line)
            docs.append(
                Document(
                    page_content=raw["content"],
                    metadata={
                        **raw["metadata"],
                        "doc_id": raw["id"],
                    },
                )
            )
    return docs


all_docs = load_docs(PROCESSED_PATH)
total = len(all_docs)
log.info("📄 %d개 청크 로드 완료 → OpenAI 임베딩/DB 저장", total)

# ─────────────────────────────────────────────────────────────
# 4️⃣ 배치 임베딩 & 업로드
log.info("🔄 배치 임베딩 및 업로드 시작")

for i in range(0, total, BATCH_SIZE):
    batch = all_docs[i : i + BATCH_SIZE]
    
    log.info(f"📦 배치 {i//BATCH_SIZE + 1}: {len(batch)}개 문서 처리 중...")
    
    try:
        vectordb.add_documents(batch)
        log.info("✅ 배치 %d 완료: %d / %d 업로드", i//BATCH_SIZE + 1, min(i + BATCH_SIZE, total), total)
        
        # 배치 간 간단한 지연
        if i + BATCH_SIZE < total:
            import time
            time.sleep(0.5)
            
    except Exception as e:
        log.error(f"❌ 배치 {i//BATCH_SIZE + 1} 처리 실패: {e}")
        log.info("⏸️ 10초 대기 후 재시도...")
        import time
        time.sleep(10.0)
        
        try:
            vectordb.add_documents(batch)
            log.info("✅ 재시도 성공: 배치 %d 완료", i//BATCH_SIZE + 1)
        except Exception as e2:
            log.error(f"❌ 재시도도 실패: {e2}")
            continue

# Vector DB 저장
vectordb.persist()
log.info("🎉 임베딩 기반 Vector DB 저장 완료 → %s", CHROMA_DIR)
log.info("📊 모델: %s", MODEL_NAME)
log.info("📈 총 문서 수: %d개", total)
