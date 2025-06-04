from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import List, Dict, Any, Set

from config import config
from utils import get_logger

# 설정
ETC_RAW_PATH = Path(config.RAW_DIR) / "etc_raw.json"

def crawl_etc_manual(
    pages: int = 1,
    depth: int = 1,
    visited_urls: Set[str] = None,
    incremental: bool = True
) -> List[Dict[str, Any]]:
    """
    etc_raw.json에서 수동으로 작성된 가이드 데이터를 로드
    
    Args:
        pages: 사용하지 않음 (호환성을 위해 유지)
        depth: 사용하지 않음 (호환성을 위해 유지)  
        visited_urls: 이미 처리된 URL 집합
        incremental: 증분 처리 여부
        
    Returns:
        List[Dict]: 가이드 데이터 리스트
    """
    logger = get_logger("etc_crawler")
    
    if visited_urls is None:
        visited_urls = set()
    
    result_data = []
    
    try:
        if not ETC_RAW_PATH.exists():
            logger.warning(f"etc_raw.json 파일이 없습니다: {ETC_RAW_PATH}")
            return []
        
        with open(ETC_RAW_PATH, "r", encoding="utf-8") as f:
            manual_data = json.load(f)
        
        if not isinstance(manual_data, list):
            logger.warning("etc_raw.json이 배열 형태가 아닙니다")
            return []
        
        for item in manual_data:
            if not isinstance(item, dict):
                continue
                
            # 필수 필드 확인
            url = item.get("url", "")
            title = item.get("title", "")
            body = item.get("body", "")
            
            if not url or not title or not body:
                logger.warning(f"필수 필드가 누락된 항목을 건너뜁니다: {item}")
                continue
            
            # 증분 처리: 이미 방문한 URL 건너뛰기
            if incremental and url in visited_urls:
                continue
            
            # 기본값 설정
            processed_item = {
                "url": url,
                "title": title,
                "date": item.get("date", datetime.now().strftime("%Y-%m-%d")),
                "views": item.get("views", 0),
                "likes": item.get("likes", 0),
                "class_name": item.get("class_name"),
                "source": item.get("source", "manual"),
                "quality_score": item.get("quality_score", 5.0),
                "body": body,
                "timestamp": item.get("timestamp", datetime.now().isoformat() + "+00:00")
            }
            
            result_data.append(processed_item)
            visited_urls.add(url)
        
        logger.info(f"✅ etc_raw.json에서 {len(result_data)}개 수동 로드 완료")
                
        return result_data
        
    except Exception as e:
        logger.error(f"etc_raw.json 처리 중 오류 발생: {e}")
        return []


if __name__ == "__main__":
    # 테스트용 실행
    result = crawl_etc_manual()
    print(f"수동 {len(result)}개 로드됨")
    for item in result:
        print(f"- {item['title']}")
