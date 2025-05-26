from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from kiwipiepy import Kiwi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# 1️⃣ 설정
MERGED_DIR = Path("data/merged")      # 통합된 크롤링 결과 폴더
SAVE_PATH = Path("data/processed_docs.jsonl")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# 약어 → 정식 용어 매핑
DNF_TERMS = {
}
DNF_CLASSES = {
    # … 필요한 만큼 추가
}

# 로깅
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("preprocess")

# ─────────────────────────────────────────────────────────────
# 2️⃣ 헬퍼
kiwi = Kiwi()

RE_HTML_TAG = re.compile(r"<[^>]*>")        # very naive stripper


def clean_html(text: str) -> str:
    """태그 제거 + 라인 정리"""
    text = RE_HTML_TAG.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_abbr(text: str) -> str:
    """DNF 약어/직업명 통일"""
    for k, v in {**DNF_TERMS, **DNF_CLASSES}.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)
    return text


def sent_tokenize(text: str) -> List[str]:
    """Korean sentence splitter (Kiwi)"""
    spans = kiwi.split_into_sents(text)
    return [s.text for s in spans]


def load_raw_files() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for path in MERGED_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            with path.open(encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    docs.extend(data if isinstance(data, list) else [data])
                except json.JSONDecodeError:
                    log.warning("JSON decode failed: %s", path)
        else:
            # html / txt
            with path.open(encoding="utf-8") as f:
                docs.append(
                    {"title": path.stem, "body": f.read(), "source": str(path)}
                )
    return docs


def extract_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """문서에서 메타데이터 추출"""
    metadata = {}
    
    # 기본 필드들
    if "url" in doc:
        metadata["url"] = doc["url"]
    if "source" in doc:
        metadata["source"] = doc["source"]
    if "date" in doc:
        metadata["date"] = doc["date"]
    if "timestamp" in doc:
        metadata["timestamp"] = doc["timestamp"]
    
    # 수치 정보
    if "views" in doc:
        metadata["views"] = doc["views"]
    if "likes" in doc:
        metadata["likes"] = doc["likes"]
    if "quality_score" in doc:
        metadata["quality_score"] = doc["quality_score"]
    
    # 카테고리/클래스 정보
    if "class_name" in doc:
        metadata["class_name"] = doc["class_name"]
    
    return metadata


# ─────────────────────────────────────────────────────────────
# 3️⃣ 메인 파이프라인
def main() -> None:
    log.info("🔍 raw 문서 로드 중…")
    raw_docs = load_raw_files()
    log.info("✅ %d개 로드 완료", len(raw_docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    out_f = SAVE_PATH.open("w", encoding="utf-8")

    processed = 0
    for idx, doc in enumerate(raw_docs):
        # 제목과 본문 추출
        title = doc.get("title", "").strip()
        # 'body' 키를 우선 확인하고, 없으면 'content' 확인
        content = doc.get("body", "") or doc.get("content", "")
        content = clean_html(content)
        
        # title과 content가 동일하거나 content가 비어있으면 content만 사용
        if not content or content.strip() == title:
            merged = title
        else:
            merged = f"{title}\n{content}" if title else content
        merged = normalize_abbr(merged)

        # 긴 문서는 sentence 단위로 split 후 chunk
        sentences = sent_tokenize(merged)
        merged_clean = "\n".join(sentences)

        chunks = splitter.split_text(merged_clean)
        
        # 메타데이터 추출
        metadata = extract_metadata(doc)
        metadata.update({
            "title": title,
            "id_raw": idx,
        })

        for n, chunk in enumerate(chunks):
            rec = {
                "id": f"{idx}_{n}",
                "content": chunk,
                "metadata": metadata.copy(),  # 각 청크마다 복사본 생성
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1

    out_f.close()
    log.info("🚀 %d개 청크 저장 → %s", processed, SAVE_PATH)
    
    # 처리 통계 출력
    log.info("📊 처리 통계:")
    log.info(f"   - 원본 문서: {len(raw_docs)}개")
    log.info(f"   - 생성된 청크: {processed}개")
    log.info(f"   - 평균 청크/문서: {processed/len(raw_docs):.1f}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()