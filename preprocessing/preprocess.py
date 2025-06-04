from __future__ import annotations

import json
import logging
import re
import sys
import hashlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from typing import Any, Dict, List, Set
from datetime import datetime

from kiwipiepy import Kiwi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# 1️⃣ 설정
MERGED_DIR           = Path(config.MERGED_DIR)
RAW_DIR              = Path(config.RAW_DIR)
PROCESSED_SAVE_PATH  = Path(config.PROCESSED_SAVE_PATH)
PROCESSED_CACHE_PATH = Path(config.PROCESSED_CACHE_PATH)
CHUNK_SIZE           = config.CHUNK_SIZE
CHUNK_OVERLAP        = config.CHUNK_OVERLAP

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
# 2️⃣ 증분 처리 도구

def get_file_hash(file_path: Path) -> str:
    """파일의 MD5 해시 계산"""
    hasher = hashlib.md5()
    try:
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""

def load_processed_cache() -> Dict[str, str]:
    """처리된 파일 캐시 로드 (file_path -> file_hash)"""
    try:
        if PROCESSED_CACHE_PATH.exists():
            with PROCESSED_CACHE_PATH.open('r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"캐시 로드 실패: {e}")
    return {}

def save_processed_cache(cache: Dict[str, str]) -> None:
    """처리된 파일 캐시 저장"""
    try:
        PROCESSED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROCESSED_CACHE_PATH.open('w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"캐시 저장 실패: {e}")

def get_new_and_updated_files(processed_cache: Dict[str, str]) -> List[Path]:
    """새로운 파일과 업데이트된 파일 탐지"""
    new_files = []
    
    # MERGED_DIR과 RAW_DIR 모두 검사
    search_dirs = [MERGED_DIR, RAW_DIR]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for path in search_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
                
            # 상대 경로를 계산할 때 어느 디렉토리에서 온 파일인지 구분
            try:
                if search_dir == MERGED_DIR:
                    file_path_str = f"merged/{path.relative_to(MERGED_DIR)}"
                else:
                    file_path_str = f"raw/{path.relative_to(RAW_DIR)}"
            except ValueError:
                continue
                
            current_hash = get_file_hash(path)
            
            # 새 파일이거나 해시가 변경된 경우
            if file_path_str not in processed_cache or processed_cache[file_path_str] != current_hash:
                new_files.append(path)
                processed_cache[file_path_str] = current_hash
    
    return new_files

def load_existing_processed_docs() -> Set[str]:
    """기존 처리된 문서 ID 집합 로드"""
    existing_ids = set()
    try:
        if PROCESSED_SAVE_PATH.exists():
            with PROCESSED_SAVE_PATH.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        existing_ids.add(data.get('id', ''))
    except Exception as e:
        log.warning(f"기존 문서 로드 실패: {e}")
    return existing_ids

def generate_document_id(doc: Dict[str, Any], chunk_index: int = 0) -> str:
    """문서의 고유한 특성을 기반으로 안전한 ID 생성"""
    # 고유 식별자 생성을 위한 요소들
    id_components = []
    
    # 1. URL이 있으면 최우선 사용
    if doc.get('url'):
        id_components.append(doc['url'])
    
    # 2. 제목 사용
    title = doc.get('title', '').strip()
    if title:
        id_components.append(title)
    
    # 3. 내용 일부 사용 (첫 100자)
    content = doc.get('body', '') or doc.get('content', '')
    if content:
        content_preview = clean_html(content)[:100]
        id_components.append(content_preview)
    
    # 4. 파일 소스 정보
    if doc.get('_file_source'):
        id_components.append(doc['_file_source'])
    
    # 5. 타임스탬프 (날짜 정보)
    if doc.get('date'):
        id_components.append(str(doc['date']))
    elif doc.get('timestamp'):
        id_components.append(str(doc['timestamp']))
    
    # 조합된 문자열을 해시화
    combined = '|'.join(id_components)
    doc_hash = hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]  # 12자리로 축약
    
    # 청크 인덱스와 조합하여 최종 ID 생성
    return f"doc_{doc_hash}_chunk_{chunk_index}"

def check_id_uniqueness(doc_id: str, existing_ids: Set[str]) -> str:
    """ID 중복 검사 및 고유 ID 보장"""
    original_id = doc_id
    counter = 1
    
    while doc_id in existing_ids:
        # 중복이면 suffix 추가
        if "_chunk_" in original_id:
            base_part, chunk_part = original_id.rsplit('_chunk_', 1)
            doc_id = f"{base_part}_dup{counter}_chunk_{chunk_part}"
        else:
            doc_id = f"{original_id}_dup{counter}"
        counter += 1
    
    return doc_id

# ─────────────────────────────────────────────────────────────
# 3️⃣ 헬퍼
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


def load_raw_files(incremental: bool = False) -> List[Dict[str, Any]]:
    """로 파일 로드 (증분 처리 지원)"""
    docs: List[Dict[str, Any]] = []
    
    if incremental:
        # 증분 모드: 새로운/업데이트된 파일만 처리
        processed_cache = load_processed_cache()
        new_files = get_new_and_updated_files(processed_cache)
        
        if not new_files:
            log.info("🔄 증분 모드: 처리할 새 파일이 없습니다")
            return []
        
        log.info(f"🔄 증분 모드: {len(new_files)}개 파일 처리 예정")
        files_to_process = new_files
        
        # 캐시 업데이트
        save_processed_cache(processed_cache)
    else:
        # 전체 모드: 모든 파일 처리
        files_to_process = []
        
        # MERGED_DIR과 RAW_DIR 모두 처리
        search_dirs = [MERGED_DIR, RAW_DIR]
        for search_dir in search_dirs:
            if search_dir.exists():
                files_to_process.extend(list(search_dir.rglob("*")))
        
        files_to_process = [p for p in files_to_process if p.is_file()]
        log.info(f"📋 전체 모드: {len(files_to_process)}개 파일 처리 예정")
    
    for path in files_to_process:
        if path.suffix.lower() in {".json", ".jsonl"}:
            with path.open(encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        # 새로운 타임스탬프 추가
                        for item in data:
                            if isinstance(item, dict):
                                # 파일이 어느 디렉토리에서 왔는지 결정
                                try:
                                    if RAW_DIR in path.parents or path.parent == RAW_DIR:
                                        item['_file_source'] = f"raw/{path.relative_to(RAW_DIR)}"
                                    else:
                                        item['_file_source'] = f"merged/{path.relative_to(MERGED_DIR)}"
                                except ValueError:
                                    item['_file_source'] = str(path.name)
                                item['_processed_at'] = datetime.now().isoformat()
                        docs.extend(data)
                    else:
                        # 파일이 어느 디렉토리에서 왔는지 결정
                        try:
                            if RAW_DIR in path.parents or path.parent == RAW_DIR:
                                data['_file_source'] = f"raw/{path.relative_to(RAW_DIR)}"
                            else:
                                data['_file_source'] = f"merged/{path.relative_to(MERGED_DIR)}"
                        except ValueError:
                            data['_file_source'] = str(path.name)
                        data['_processed_at'] = datetime.now().isoformat()
                        docs.append(data)
                except json.JSONDecodeError:
                    log.warning("JSON decode failed: %s", path)
        else:
            # html / txt
            with path.open(encoding="utf-8") as f:
                # 파일이 어느 디렉토리에서 왔는지 결정
                try:
                    if RAW_DIR in path.parents or path.parent == RAW_DIR:
                        file_source = f"raw/{path.relative_to(RAW_DIR)}"
                    else:
                        file_source = f"merged/{path.relative_to(MERGED_DIR)}"
                except ValueError:
                    file_source = str(path.name)
                    
                doc_data = {
                    "title": path.stem, 
                    "body": f.read(), 
                    "source": str(path),
                    '_file_source': file_source,
                    '_processed_at': datetime.now().isoformat()
                }
                docs.append(doc_data)
    
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
# 4️⃣ 개선된 메인 파이프라인
def main(incremental: bool = False) -> None:
    log.info("🔍 raw 문서 로드 중…")
    raw_docs = load_raw_files(incremental=incremental)
    log.info("✅ %d개 로드 완료", len(raw_docs))

    # 증분 모드에서 처리할 새 파일이 없으면 종료
    if incremental and not raw_docs:
        log.info("✅ 전처리 완료! (처리할 새 파일 없음)")
        return

    # 증분 모드일 때만 기존 ID 로드 (중복 방지)
    existing_ids = set()
    if incremental:
        existing_ids = load_existing_processed_docs()
        log.info(f"📋 기존 처리된 문서 ID: {len(existing_ids)}개")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    # 증분 모드는 append, 전체 모드는 새로 생성
    mode = "a" if incremental and PROCESSED_SAVE_PATH.exists() else "w"
    out_f = PROCESSED_SAVE_PATH.open(mode, encoding="utf-8")

    processed = 0
    skipped_duplicates = 0
    
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
            "doc_index": idx,  # id_raw 대신 doc_index 사용
            "_file_source": doc.get('_file_source', ''),
            "_processed_at": doc.get('_processed_at', '')
        })

        # 증분 모드에서 이미 처리된 문서인지 확인
        if incremental:
            # 기본 문서ID로 중복 체크 (청크 0 기준)
            base_doc_id = generate_document_id(doc, 0)
            base_prefix = base_doc_id.replace('_chunk_0', '')
            
            # 이미 처리된 문서인지 확인
            already_processed = any(existing_id.startswith(base_prefix) for existing_id in existing_ids)
            
            if already_processed:
                skipped_duplicates += len(chunks)
                log.debug(f"문서 건너뛰기 (이미 처리됨): {title[:50]}...")
                continue

        for chunk_idx, chunk in enumerate(chunks):
            # 새로운 ID 생성 로직 사용
            doc_id = generate_document_id(doc, chunk_idx)
            
            # 증분 모드에서 ID 고유성 보장
            if incremental:
                doc_id = check_id_uniqueness(doc_id, existing_ids)
                existing_ids.add(doc_id)
            
            rec = {
                "id": doc_id,
                "content": chunk,
                "metadata": metadata.copy(),  # 각 청크마다 복사본 생성
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1

    out_f.close()
    log.info("🚀 %d개 청크 저장 → %s", processed, PROCESSED_SAVE_PATH)
    
    # 처리 통계 출력
    log.info("📊 처리 통계:")
    log.info(f"   - 원본 문서: {len(raw_docs)}개")
    log.info(f"   - 생성된 청크: {processed}개")
    if skipped_duplicates > 0:
        log.info(f"   - 중복 건너뛰기: {skipped_duplicates}개")
    if len(raw_docs) > 0:
        log.info(f"   - 평균 청크/문서: {processed/len(raw_docs):.1f}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="던파 데이터 전처리 스크립트 (개선된 증분 처리)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 전처리 (기본 모드)
  python preprocess.py
  
  # 증분 전처리 (새로운 데이터만)
  python preprocess.py --incremental
  
  # 증분 전처리 + 상세 로그
  python preprocess.py --incremental --verbose
        """
    )
    
    parser.add_argument(
        "--incremental", 
        action="store_true", 
        default=True,
        help="증분 전처리 모드 (기본값)"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="전체 전처리 모드 (증분 무시)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="상세한 로그 출력"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"청크 크기 (기본: {CHUNK_SIZE})"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int, 
        default=CHUNK_OVERLAP,
        help=f"청크 겹침 크기 (기본: {CHUNK_OVERLAP})"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 처리 결과 강제 덮어쓰기"
    )
    
    args = parser.parse_args()
    
    # 전체 모드 검사
    if args.full:
        args.incremental = False
    
    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 청크 설정 업데이트
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    
    # 강제 모드 처리
    if args.force:
        if PROCESSED_SAVE_PATH.exists():
            PROCESSED_SAVE_PATH.unlink()
            log.info(f"🗑️ 기존 결과 파일 삭제: {PROCESSED_SAVE_PATH}")
        if PROCESSED_CACHE_PATH.exists():
            PROCESSED_CACHE_PATH.unlink() 
            log.info(f"🗑️ 기존 캐시 파일 삭제: {PROCESSED_CACHE_PATH}")
    
    # 시작 메시지
    mode_emoji = "🔄" if args.incremental else "📋"
    mode_name = "증분" if args.incremental else "전체"
    log.info(f"{mode_emoji} {mode_name} 전처리 모드 시작")
    log.info(f"   - 청크 크기: {CHUNK_SIZE}")
    log.info(f"   - 청크 겹침: {CHUNK_OVERLAP}")
    log.info(f"   - 입력 디렉토리: {MERGED_DIR}")
    log.info(f"   - 출력 파일: {PROCESSED_SAVE_PATH}")
    
    try:
        main(incremental=args.incremental)
        log.info("✅ 전처리 완료!")
    except KeyboardInterrupt:
        log.info("⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        log.error(f"❌ 전처리 실패: {e}")
        raise
