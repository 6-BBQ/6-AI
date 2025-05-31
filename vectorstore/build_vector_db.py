from __future__ import annotations

import json
import logging
import sys
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from typing import Dict, List, Set, Tuple
from datetime import datetime

import torch

from langchain.docstore.document import Document
from langchain_chroma import Chroma
# 임베딩 함수는 config.create_embedding_function()을 사용

# ───────────────────────────────────────────────
# 1️⃣ 기본 설정
PROCESSED_DOCS_PATH       = Path(config.PROCESSED_DOCS_PATH)
VECTOR_DB_DIR             = config.VECTOR_DB_DIR
VECTORDB_CACHE_PATH       = Path(config.VECTORDB_CACHE_PATH)
JOB_EMBEDDINGS_PATH       = Path(config.JOB_EMBEDDINGS_PATH)
JOB_NAMES_PATH            = Path(config.JOB_NAMES_PATH)
EMBED_MODEL_NAME          = config.EMBED_MODEL_NAME
EMBED_BATCH_SIZE          = config.EMBED_BATCH_SIZE
JOB_SIMILARITY_THRESHOLD  = config.JOB_SIMILARITY_THRESHOLD

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
        if VECTORDB_CACHE_PATH.exists():
            with VECTORDB_CACHE_PATH.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_doc_ids', []))
    except Exception as e:
        log.warning(f"캐시 로드 실패: {e}")
    return set()

def save_vectordb_cache(processed_ids: Set[str]) -> None:
    """벡터DB 캐시 저장"""
    try:
        VECTORDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'processed_doc_ids': list(processed_ids),
            'last_updated': datetime.now().isoformat(),
            'model_name': EMBED_MODEL_NAME,
            'total_docs': len(processed_ids)
        }
        with VECTORDB_CACHE_PATH.open('w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"캐시 저장 실패: {e}")

def get_existing_doc_ids_from_db(vectordb) -> Set[str]:
    """기존 벡터DB에서 문서 ID 집합 추출"""
    try:
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
# 3️⃣ 직업별 임베딩 관리 함수

def load_job_names() -> List[str]:
    """직업명 목록 로드"""
    try:
        if JOB_NAMES_PATH.exists():
            with JOB_NAMES_PATH.open('r', encoding='utf-8') as f:
                job_names = json.load(f)
                log.info(f"📋 직업명 {len(job_names)}개 로드 완료")
                return job_names
    except Exception as e:
        log.warning(f"직업명 로드 실패: {e}")
    return []

def build_job_embeddings(embedding_fn, job_names: List[str]) -> Dict[str, List[float]]:
    """각 직업명에 대한 임베딩 벡터 생성"""
    log.info(f"🧠 {len(job_names)}개 직업에 대한 임베딩 생성 중...")
    
    job_embeddings = {}
    
    for job_name in job_names:
        job_context = f"던전앤파이터 {job_name} 직업 가이드 공략 스킬 장비 세팅"
        
        try:
            embedding_vector = embedding_fn.embed_query(job_context)
            job_embeddings[job_name] = embedding_vector
        except Exception as e:
            log.warning(f"직업 '{job_name}' 임베딩 생성 실패: {e}")
            continue
    
    log.info(f"✅ {len(job_embeddings)}개 직업 임베딩 생성 완료")
    return job_embeddings

def save_job_embeddings(job_embeddings: Dict[str, List[float]]) -> None:
    """직업별 임베딩 저장"""
    try:
        JOB_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_embeddings = {}
        for job_name, embedding in job_embeddings.items():
            if isinstance(embedding, np.ndarray):
                serializable_embeddings[job_name] = embedding.tolist()
            else:
                serializable_embeddings[job_name] = embedding
        
        save_data = {
            'job_embeddings': serializable_embeddings,
            'model_name': EMBED_MODEL_NAME,
            'threshold': JOB_SIMILARITY_THRESHOLD,
            'created_at': datetime.now().isoformat(),
            'total_jobs': len(serializable_embeddings)
        }
        
        with JOB_EMBEDDINGS_PATH.open('w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        log.info(f"💾 직업 임베딩 저장 완료: {JOB_EMBEDDINGS_PATH}")
        
    except Exception as e:
        log.error(f"직업 임베딩 저장 실패: {e}")

def load_job_embeddings() -> Tuple[Dict[str, np.ndarray], str]:
    """저장된 직업별 임베딩 로드"""
    try:
        if JOB_EMBEDDINGS_PATH.exists():
            with JOB_EMBEDDINGS_PATH.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
            job_embeddings = {}
            for job_name, embedding_list in data['job_embeddings'].items():
                job_embeddings[job_name] = np.array(embedding_list)
                
            model_name = data.get('model_name', EMBED_MODEL_NAME)
            log.info(f"📂 저장된 직업 임베딩 로드: {len(job_embeddings)}개 직업")
            return job_embeddings, model_name
            
    except Exception as e:
        log.warning(f"직업 임베딩 로드 실패: {e}")
    
    return {}, EMBED_MODEL_NAME

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    try:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    except Exception:
        return 0.0

def classify_document_job(doc_content: str, embedding_fn, job_embeddings: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """문서 내용을 기반으로 직업 분류"""
    try:
        doc_embedding = np.array(embedding_fn.embed_query(doc_content))
        
        best_job = None
        best_score = 0.0
        
        for job_name, job_embedding in job_embeddings.items():
            similarity = cosine_similarity(doc_embedding, job_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_job = job_name
        
        if best_score >= JOB_SIMILARITY_THRESHOLD:
            return best_job, best_score
        else:
            return None, best_score
            
    except Exception as e:
        log.warning(f"문서 직업 분류 실패: {e}")
        return None, 0.0

def classify_existing_documents():
    """기존 문서들에 대해 직업 분류 수행 (후처리)"""
    log.info("🎯 기존 문서들에 대한 직업 분류 시작")
    
    job_embeddings, model_name = load_job_embeddings()
    if not job_embeddings:
        log.error("❌ 직업 임베딩을 찾을 수 없습니다. 먼저 벡터DB를 구축하세요.")
        return
    
    # config에서 임베딩 함수 생성
    embedding_fn = config.create_embedding_function()
    
    if not PROCESSED_DOCS_PATH.exists():
        log.error(f"❌ 전처리된 문서 파일이 없습니다: {PROCESSED_DOCS_PATH}")
        return
    
    classified_docs = []
    total_classified = 0
    total_processed = 0
    
    with PROCESSED_DOCS_PATH.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw: Dict = json.loads(line)
                total_processed += 1
                
                if raw["metadata"].get("class_name"):
                    classified_docs.append(raw)
                    continue
                
                title = raw["metadata"].get("title", "")
                content = raw.get("content", "")
                combined_text = f"{title} {content}".strip()
                
                if not combined_text:
                    classified_docs.append(raw)
                    continue
                
                classified_job, similarity_score = classify_document_job(
                    combined_text, embedding_fn, job_embeddings
                )
                
                if classified_job:
                    raw["metadata"]["class_name"] = classified_job
                    raw["metadata"]["job_similarity_score"] = round(similarity_score, 3)
                    total_classified += 1
                
                classified_docs.append(raw)
                
                if total_processed % 1000 == 0:
                    log.info(f"🔄 진행상황: {total_processed}개 처리 중, {total_classified}개 분류 완료")
                
            except json.JSONDecodeError as e:
                log.warning(f"줄 {line_num}에서 JSON 파싱 실패: {e}")
                continue
            except Exception as e:
                log.warning(f"줄 {line_num}에서 문서 처리 실패: {e}")
                continue
    
    if classified_docs:
        backup_path = PROCESSED_DOCS_PATH.with_suffix('.backup.jsonl')
        if PROCESSED_DOCS_PATH.exists():
            shutil.copy2(PROCESSED_DOCS_PATH, backup_path)
            log.info(f"💾 기존 파일 백업: {backup_path}")
        
        with PROCESSED_DOCS_PATH.open('w', encoding='utf-8') as f:
            for doc in classified_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\\n')
        
        log.info(f"✅ 직업 분류 완료!")
        log.info(f"   - 전체 문서: {total_processed}개")
        log.info(f"   - 분류된 문서: {total_classified}개")
        log.info(f"   - 분류율: {(total_classified/total_processed*100):.1f}%")

# ───────────────────────────────────────────────
# 4️⃣ 메인 함수

def load_docs(path: Path, existing_ids: Set[str] = None) -> List[Document]:
    """문서 로드 (증분 처리)"""
    docs = []
    skipped = 0
    
    if existing_ids is None:
        existing_ids = set()
    
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw: Dict = json.loads(line)
                doc_id = raw.get("id", "")
                
                if doc_id in existing_ids:
                    skipped += 1
                    continue
                
                title = raw["metadata"].get("title", "")
                page_content = f"[TITLE] {title}\\n{raw['content']}"

                doc = Document(
                    page_content=page_content,
                    metadata={**raw["metadata"], "doc_id": doc_id}
                )
                docs.append(doc)
                
            except json.JSONDecodeError as e:
                log.warning(f"줄 {line_num}에서 JSON 파싱 실패: {e}")
                continue
            except Exception as e:
                log.warning(f"줄 {line_num}에서 문서 처리 실패: {e}")
                continue
    
    if skipped > 0:
        log.info(f"🔄 증분 모드: {skipped}개 문서 건너뛰기 (이미 처리됨)")
    
    return docs

def main():
    """벡터 DB 구축 메인 함수 (증분 모드)"""
    log.info(f"🚀 증분 모드 - {config.EMBEDDING_TYPE} {EMBED_MODEL_NAME} 기반 임베딩 시작")
    
    # config에서 임베딩 함수 생성
    embedding_fn = config.create_embedding_function()
    
    # 직업별 임베딩 구축 및 저장
    job_names = load_job_names()
    if job_names:
        existing_job_embeddings, existing_model = load_job_embeddings()
        
        if (not existing_job_embeddings or 
            existing_model != EMBED_MODEL_NAME or
            set(job_names) != set(existing_job_embeddings.keys())):
            
            log.info("🔄 직업별 임베딩 새로 구축 중...")
            new_job_embeddings = build_job_embeddings(embedding_fn, job_names)
            if new_job_embeddings:
                save_job_embeddings(new_job_embeddings)
        else:
            log.info("📂 기존 직업 임베딩 사용")
    else:
        log.warning("⚠️ 직업명 목록을 로드할 수 없어 직업 임베딩을 건너뜍니다")
    
    # 벡터DB 초기화
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_fn,
    )
    
    # 기존 처리된 문서 ID 로드 (증분 처리)
    cached_ids = load_vectordb_cache()
    db_ids = get_existing_doc_ids_from_db(vectordb)
    existing_ids = cached_ids.union(db_ids)
    log.info(f"🔄 증분 모드: 기존 처리된 문서 {len(existing_ids)}개")
    
    # 문서 로드
    if not PROCESSED_DOCS_PATH.exists():
        log.error(f"❌ 전처리된 문서 파일이 없습니다: {PROCESSED_DOCS_PATH}")
        return
    
    all_docs = load_docs(PROCESSED_DOCS_PATH, existing_ids)
    
    if not all_docs:
        log.info("✅ 처리할 새 문서가 없습니다")
        return
    
    total = len(all_docs)
    log.info(f"📄 {total}개 새 문서 로딩 완료")
    
    # 배치 임베딩 & 업로드
    log.info("🔄 배치 임베딩 시작")
    processed_count = 0
    new_doc_ids = set()
    
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = all_docs[i:i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        log.info(f"📦 배치 {batch_num}: {len(batch)}개 문서 처리 중...")
        
        try:
            vectordb.add_documents(batch)
            processed_count += len(batch)
            
            for doc in batch:
                doc_id = doc.metadata.get('doc_id', '')
                if doc_id:
                    new_doc_ids.add(doc_id)
            
            log.info(f"✅ 배치 {batch_num} 완료 ({processed_count}/{total})")
            
            if i + EMBED_BATCH_SIZE < total:
                import time
                time.sleep(0.5)
                
        except Exception as e:
            log.error(f"❌ 배치 {batch_num} 실패: {e}")
            continue
    
    # 캐시 업데이트
    if new_doc_ids:
        updated_ids = existing_ids.union(new_doc_ids)
        save_vectordb_cache(updated_ids)
        log.info(f"💾 캐시 업데이트: 총 {len(updated_ids)}개 문서 ID 저장")
    
    # 완료 메시지
    log.info(f"🎉 벡터 DB 저장 완료 → {VECTOR_DB_DIR}")
    log.info(f"📈 새로 추가된 문서: {processed_count}개")
    log.info(f"📊 전체 문서 수: {len(existing_ids) + len(new_doc_ids)}개")
    log.info(f"🧠 사용 모델: {EMBED_MODEL_NAME}")

# ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--classify-jobs":
        log.info("🎯 직업 분류 모드")
        try:
            classify_existing_documents()
            log.info("✅ 직업 분류 완료!")
        except Exception as e:
            log.error(f"❌ 직업 분류 실패: {e}")
    else:
        log.info("🔄 증분 벡터 DB 구축 시작")
        try:
            main()
            log.info("✅ 벡터 DB 구축 완료!")
        except Exception as e:
            log.error(f"❌ 벡터 DB 구축 실패: {e}")