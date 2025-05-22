from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1️⃣ 설정
PROCESSED_PATH = Path("data/processed_docs.jsonl")
CHROMA_DIR = "vector_db/chroma"        # persist 디렉터리
BATCH_SIZE = 200                       # GPU/CPU 상황에 맞게 조절
EMBEDDING_MODEL = "text-embedding-3-large"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_vector_db")

# ─────────────────────────────────────────────────────────────
# 2️⃣ 임베딩 & DB 초기화
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY 환경변수를 먼저 설정하세요!")

embedding_fn = OpenAIEmbeddings(model=EMBEDDING_MODEL)

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
log.info("📄 %d개 청크 로드 완료 → 임베딩/DB 저장", total)

# ─────────────────────────────────────────────────────────────
# 4️⃣ 배치 임베딩 & 업로드
for i in range(0, total, BATCH_SIZE):
    batch = all_docs[i : i + BATCH_SIZE]
    vectordb.add_documents(batch)
    log.info("✅ %d / %d 업로드", min(i + BATCH_SIZE, total), total)

vectordb.persist()
log.info("🎉 Vector DB 저장 완료 → %s", CHROMA_DIR)
