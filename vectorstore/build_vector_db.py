from __future__ import annotations

import json
import logging
import os
import shutil
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

from dotenv import load_dotenv
import torch

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ───────────────────────────────────────────────
# 1️⃣ 기본 설정
load_dotenv()
PROCESSED_PATH = Path("data/processed/processed_docs.jsonl")
CHROMA_DIR = "vector_db/chroma"
VECTORDB_CACHE = Path("vector_db/vectordb_cache.json")  # 벡터DB 캐시
BATCH_SIZE = 200
MODEL_NAME = "dragonkue/bge-m3-ko"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_vector_db")

# ───────────────────────────────────────────────
# 2️⃣ 증분 처리 도구

def load_vectordb_cache() -> Set[str]:
    """벡터DB에 이미 추가된 문서 ID 집합 로드"""
    try:
        if VECTORDB_CACHE.exists():
            with VECTORDB_CACHE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_doc_ids', []))
    except Exception as e:
        log.warning(f"캐시 로드 실패: {e}")
    return set()

def save_vectordb_cache(processed_ids: Set[str]) -> None:
    """벡터DB 캐시 저장"""
    try:
        VECTORDB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'processed_doc_ids': list(processed_ids),
            'last_updated': datetime.now().isoformat(),
            'model_name': MODEL_NAME,
            'total_docs': len(processed_ids)
        }
        with VECTORDB_CACHE.open('w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"캐시 저장 실패: {e}")

def get_existing_doc_ids_from_db(vectordb) -> Set[str]:
    """기존 벡터DB에서 문서 ID 집합 추출"""
    try:
        # Chroma에서 모든 문서의 메타데이터 가져오기
        collection = vectordb.get()
        existing_ids = set()
        
        if collection and 'metadatas' in collection:
            for metadata in collection['metadatas']:
                if metadata and 'doc_id' in metadata:
                    existing_ids.add(metadata['doc_id'])
        
        log.info(f"📊 기존 벡터DB에서 {len(existing_ids)}개 문서 ID 발견")
        return existing_ids
    except Exception as e:
        log.warning(f"기존 DB 문서 ID 추출 실패: {e}")
        return set()

# ───────────────────────────────────────────────
# 3️⃣ JSONL → Document 리스트 변환 함수

def load_docs(path: Path, incremental: bool = False, existing_ids: Set[str] = None) -> List[Document]:
    """문서 로드 (증분 처리 지원)"""
    docs = []
    skipped = 0
    
    if existing_ids is None:
        existing_ids = set()
    
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw: Dict = json.loads(line)
                doc_id = raw.get("id", "")
                
                # 증분 모드에서 이미 처리된 문서 건너뛰기
                if incremental and doc_id in existing_ids:
                    skipped += 1
                    continue
                
                doc = Document(
                    page_content=raw["content"],
                    metadata={**raw["metadata"], "doc_id": doc_id}
                )
                docs.append(doc)
                
            except json.JSONDecodeError as e:
                log.warning(f"줄 {line_num}에서 JSON 파싱 실패: {e}")
                continue
            except Exception as e:
                log.warning(f"줄 {line_num}에서 문서 처리 실패: {e}")
                continue
    
    if incremental and skipped > 0:
        log.info(f"🔄 증분 모드: {skipped}개 문서 건너뛰기 (이미 처리됨)")
    
    return docs

# ───────────────────────────────────────────────
# 4️⃣ 메인 함수

def main(incremental: bool = False, force: bool = False):
    """벡터 DB 구축 메인 함수"""
    
    mode_name = "증분" if incremental else "전체"
    log.info(f"🚀 {mode_name} 모드 - 한국어 BGE-m3-ko 기반 임베딩 시작")
    
    # 임베딩 함수 정의
    embedding_fn = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 강제 모드일 때 기존 데이터 삭제
    if force and Path(CHROMA_DIR).exists():
        log.info("🗑️ 강제 모드: 기존 Chroma 폴더 삭제")
        shutil.rmtree(CHROMA_DIR)
        if VECTORDB_CACHE.exists():
            VECTORDB_CACHE.unlink()
            log.info("🗑️ 강제 모드: 기존 캐시 파일 삭제")
    
    # 벡터DB 초기화
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_fn,
    )
    
    # 기존 처리된 문서 ID 로드
    existing_ids = set()
    if incremental:
        # 캐시에서 로드 (빠름)
        cached_ids = load_vectordb_cache()
        
        # 실제 DB에서 로드 (정확함)
        db_ids = get_existing_doc_ids_from_db(vectordb)
        
        # 두 결과를 합집합으로 처리 (안전함)
        existing_ids = cached_ids.union(db_ids)
        log.info(f"🔄 증분 모드: 기존 처리된 문서 {len(existing_ids)}개")
    elif not force and Path(CHROMA_DIR).exists():
        # 전체 모드이지만 force가 아닌 경우, 기존 DB 유지하고 추가
        log.info("📋 전체 모드: 기존 DB에 새 문서 추가")
        existing_ids = get_existing_doc_ids_from_db(vectordb)
    
    # 문서 로드
    if not PROCESSED_PATH.exists():
        log.error(f"❌ 전처리된 문서 파일이 없습니다: {PROCESSED_PATH}")
        return
    
    all_docs = load_docs(PROCESSED_PATH, incremental, existing_ids)
    
    if not all_docs:
        log.info("✅ 처리할 새 문서가 없습니다")
        return
    
    total = len(all_docs)
    log.info(f"📄 {total}개 새 문서 로딩 완료")
    
    # 배치 임베딩 & 업로드
    log.info("🔄 배치 임베딩 시작")
    processed_count = 0
    new_doc_ids = set()
    
    for i in range(0, total, BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log.info(f"📦 배치 {batch_num}: {len(batch)}개 문서 처리 중...")
        
        try:
            vectordb.add_documents(batch)
            processed_count += len(batch)
            
            # 처리된 문서 ID 기록
            for doc in batch:
                doc_id = doc.metadata.get('doc_id', '')
                if doc_id:
                    new_doc_ids.add(doc_id)
            
            log.info(f"✅ 배치 {batch_num} 완료 ({processed_count}/{total})")
            
            # 배치 간 딜레이
            if i + BATCH_SIZE < total:
                import time
                time.sleep(0.5)
                
        except Exception as e:
            log.error(f"❌ 배치 {batch_num} 실패: {e}")
            import time
            time.sleep(10.0)
            
            # 재시도
            try:
                vectordb.add_documents(batch)
                processed_count += len(batch)
                
                for doc in batch:
                    doc_id = doc.metadata.get('doc_id', '')
                    if doc_id:
                        new_doc_ids.add(doc_id)
                        
                log.info(f"✅ 재시도 성공: 배치 {batch_num} 완료")
            except Exception as e2:
                log.error(f"❌ 재시도 실패: {e2}")
                continue
    
    # 캐시 업데이트 (증분 모드일 때)
    if incremental or new_doc_ids:
        updated_ids = existing_ids.union(new_doc_ids)
        save_vectordb_cache(updated_ids)
        log.info(f"💾 캐시 업데이트: 총 {len(updated_ids)}개 문서 ID 저장")
    
    # 완료 메시지
    log.info(f"🎉 벡터 DB 저장 완료 → {CHROMA_DIR}")
    log.info(f"📈 새로 추가된 문서: {processed_count}개")
    log.info(f"📊 전체 문서 수: {len(existing_ids) + len(new_doc_ids)}개")
    log.info(f"🧠 사용 모델: {MODEL_NAME}")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="던파 벡터 DB 구축 스크립트 (증분 처리 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 벡터 DB 구축 (기본 모드)
  python build_vector_db.py
  
  # 증분 벡터 DB 구축 (새로운 문서만)
  python build_vector_db.py --incremental
  
  # 강제 전체 재구축
  python build_vector_db.py --force
  
  # 증분 + 상세 로그
  python build_vector_db.py --incremental --verbose
        """
    )
    
    parser.add_argument(
        "--incremental", 
        action="store_true", 
        default=True,
        help="증분 벡터 DB 구축 (기본값)"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="전체 벡터 DB 구축 (증분 무시)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="기존 벡터 DB 강제 삭제 후 전체 재구축"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="상세한 로그 출력"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"배치 크기 (기본: {BATCH_SIZE})"
    )
    
    args = parser.parse_args()
    
    # 전체 모드 검사
    if args.full:
        args.incremental = False
    
    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 배치 크기 설정
    BATCH_SIZE = args.batch_size
    
    # 모드 충돌 검사
    if args.incremental and args.force:
        log.warning("⚠️ --incremental과 --force를 함께 사용하면 --force가 우선됩니다")
        args.incremental = False
    elif args.full and args.force:
        # --full과 --force는 동일한 효과
        pass
    
    # 시작 메시지
    mode_emoji = "🔄" if args.incremental else "🗑️" if args.force else "📋"
    mode_name = "증분" if args.incremental else "강제 재구축" if args.force else "전체"
    
    log.info(f"{mode_emoji} {mode_name} 벡터 DB 구축 시작")
    log.info(f"   - 배치 크기: {BATCH_SIZE}")
    log.info(f"   - 모델: {MODEL_NAME}")
    log.info(f"   - 입력 파일: {PROCESSED_PATH}")
    log.info(f"   - 출력 디렉토리: {CHROMA_DIR}")
    
    try:
        main(incremental=args.incremental, force=args.force)
        log.info("✅ 벡터 DB 구축 완료!")
    except KeyboardInterrupt:
        log.info("⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        log.error(f"❌ 벡터 DB 구축 실패: {e}")
        raise
