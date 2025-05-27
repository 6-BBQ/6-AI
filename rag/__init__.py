"""
RAG 패키지 - 구조화된 RAG 시스템

기존 rag_service.py와 새로운 구조화된 서비스를 모두 제공합니다.
"""

# 새로운 구조화된 RAG 서비스
from .service import (
    StructuredRAGService,
    get_structured_rag_service,
    get_structured_rag_answer
)

# 유틸리티 클래스들 (선택적 사용)
from .cache_utils import CacheManager
from .text_utils import TextProcessor
from .retrievers import MetadataAwareRetriever as NewMetadataAwareRetriever
from .search_factory import SearcherFactory

__all__ = [
    # 새로운 구조화된 RAG 서비스
    'StructuredRAGService',
    'get_structured_rag_service', 
    'get_structured_rag_answer',
    
    # 유틸리티 클래스들
    'CacheManager',
    'TextProcessor',
    'NewMetadataAwareRetriever',
    'SearcherFactory'
]

# 기존 함수들을 그대로 유지 (하위 호환성)
__version__ = "2.0.0"
