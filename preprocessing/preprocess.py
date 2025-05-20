import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from textwrap import wrap

# ─────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST = ["선계", "커스텀", "미스트 기어", "아칸", "출혈"]
PRIORITY_TERMS = ["스펙업", "파밍", "명성", "레기온", "레이드", "뉴비", "가이드"]

CHUNK_SIZE = 500
OUTPUT_FILE = PROCESSED_DIR / "processed_docs.json"
# ─────────────────────────────────────────────────────

def is_valid_doc(text: str) -> bool:
    if any(bad in text for bad in BLACKLIST):
        return False
    return any(term in text for term in PRIORITY_TERMS)


def clean_text(text: str) -> str:
    """HTML 태그 및 불필요한 구문 제거"""
    text = BeautifulSoup(text, "html.parser").get_text()
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        if re.match(r"^(제목|작성자|※|▶|바로가기)", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def chunk_document(item: dict, max_chunk_size: int = CHUNK_SIZE):
    """하나의 게시글 JSON 항목을 여러 문서로 분할"""
    text = clean_text(item["content"])
    chunks = wrap(text, max_chunk_size, break_long_words=False, replace_whitespace=False)

    return [
        {
            "text": chunk,
            "metadata": {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "date": item.get("date", "")
            }
        }
        for chunk in chunks
    ]


def process_all_json_files():
    """data/raw 내의 모든 JSON 파일 전처리"""
    documents = []
    for json_file in RAW_DATA_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                raw_text = item["content"]
                # 블랙리스트·우선 키워드 검사
                if not is_valid_doc(raw_text):
                    continue
                documents.extend(chunk_document(item))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✅ 전처리 완료: {len(documents)}개 문서 저장됨 → {OUTPUT_FILE}")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    process_all_json_files()
