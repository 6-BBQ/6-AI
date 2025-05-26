"""
RAG 시스템을 위한 캐싱 유틸리티
"""
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class CacheManager:
    """캐싱 관련 기능을 담당하는 클래스"""
    
    def __init__(self, cache_dir: Path, expiry_short: int = 43200, expiry_long: int = 86400):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            expiry_short: 단기 캐시 만료 시간 (초, 기본 12시간)
            expiry_long: 장기 캐시 만료 시간 (초, 기본 24시간)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.expiry_short = expiry_short
        self.expiry_long = expiry_long
    
    def generate_cache_key(self, base_content: str, character_info: Optional[Dict] = None) -> str:
        """캐시 키 생성 (FastAPI에서 변환된 캐릭터 정보 포함 가능)"""
        cache_input = base_content
        if character_info:
            # FastAPI에서 변환된 키들을 사용
            char_key_parts = [
                character_info.get('job', ''),
                str(character_info.get('fame', ''))
            ]
            
            # 주요 정보만으로 키 생성
            simple_char_key = "-".join(filter(None, char_key_parts))
            if simple_char_key:
                cache_input = f"{base_content}|{simple_char_key}"
        
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()
    
    def load_or_create_cached_item(self, 
                                   cache_file_name: str, 
                                   creation_func: Callable[[], Any], 
                                   expiry_seconds: int,
                                   item_name: str = "항목") -> Any:
        """캐시된 항목을 로드하거나 새로 생성"""
        cache_file = self.cache_dir / cache_file_name
        
        # 캐시 파일이 존재하고 만료되지 않았으면 로드
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < expiry_seconds:
                try:
                    print(f"🔄 캐시된 {item_name} 로딩: {cache_file_name}")
                    with open(cache_file, 'rb') as f: 
                        item = pickle.load(f)
                    print(f"✅ {item_name} 캐시 로드 완료")
                    return item
                except Exception as e:
                    print(f"⚠️ {item_name} 캐시 로드 실패 ({cache_file_name}): {e}. 재생성합니다.")
        
        # 캐시가 없거나 만료되었으면 새로 생성
        print(f"🔄 {item_name} 생성 중 ({cache_file_name})...")
        item = creation_func()
        
        # 생성된 항목을 캐시에 저장
        try:
            with open(cache_file, 'wb') as f: 
                pickle.dump(item, f)
            print(f"✅ {item_name} 캐시 저장 완료: {cache_file}")
        except Exception as e:
            print(f"⚠️ {item_name} 캐시 저장 실패 ({cache_file_name}): {e}")
        
        return item
    
    def get_cached_search_result(self, query: str, cache_type: str, character_info: Optional[Dict] = None) -> Optional[Any]:
        """캐시된 검색 결과 조회"""
        cache_key = self.generate_cache_key(query, character_info)
        cache_file_name = f"{cache_type}_{cache_key}.pkl"
        cache_file = self.cache_dir / cache_file_name
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.expiry_short:
                try:
                    with open(cache_file, 'rb') as f: 
                        return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ {cache_type} 검색 캐시 로드 실패: {e}")
        
        return None
    
    def save_search_result_to_cache(self, query: str, result: Any, cache_type: str, character_info: Optional[Dict] = None):
        """검색 결과를 캐시에 저장"""
        cache_key = self.generate_cache_key(query, character_info)
        cache_file_name = f"{cache_type}_{cache_key}.pkl"
        cache_file = self.cache_dir / cache_file_name
        
        try:
            with open(cache_file, 'wb') as f: 
                pickle.dump(result, f)
        except Exception as e:
            print(f"⚠️ {cache_type} 검색 캐시 저장 실패: {e}")
