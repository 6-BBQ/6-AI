"""
RAG 시스템을 위한 검색기(Retriever) 클래스들
"""
from typing import List, Optional
from langchain.docstore.document import Document
from job_utils import find_job_in_text  


class MetadataAwareRetriever:
    """메타데이터를 고려한 지능형 검색기"""
    
    def __init__(self, base_retriever, top_n: int = 40):
        """
        Args:
            base_retriever: 기본 검색기 (ContextualCompressionRetriever 등)
            top_n: 반환할 문서 수
        """
        self.base_retriever = base_retriever
        self.top_n = top_n
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """메타데이터 기반 스코어링으로 문서 검색 (쿼리 관련성 우선)"""
        # 기본 검색기로 문서 검색
        docs = self.base_retriever.get_relevant_documents(query)
        
        # 쿼리 전처리
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # query 안에서 직업명 탐색
        character_job = find_job_in_text(query_lower)
        
        # 메타데이터 기반 스코어링
        scored_docs = []
        for doc in docs:
            score = 1.0
            meta = doc.metadata or {}
            
            # 1. 쿼리 관련성 점수 (최우선)
            relevance_score = 0.0
            
            # 직업명 정확 매칭 - 최고 우선순위
            if class_name := meta.get("class_name", ""):
                class_name_lower = class_name.lower()
                # 정확한 직업명 매칭
                if character_job and character_job in class_name_lower:
                    relevance_score += 10.0  # 매우 높은 보너스
                # 부분 매칭
                elif any(word in class_name_lower for word in query_words if len(word) > 2):
                    relevance_score += 3.0
            
            # 제목 매칭
            if title := meta.get("title", ""):
                title_lower = title.lower()
                # 직업명이 제목에 있는 경우
                if character_job and character_job in title_lower:
                    relevance_score += 5.0
                # 일반 쿼리 단어 매칭
                matching_words = sum(1 for word in query_words if word in title_lower and len(word) > 1)
                relevance_score += matching_words * 1.0
            
            # 내용 매칭 (보조적)
            content_lower = doc.page_content[:500].lower()  # 앞부분만 확인
            if character_job and character_job in content_lower:
                relevance_score += 2.0
            
            # 2. 품질 점수 (보조적으로만 사용)
            try:
                quality = float(meta.get("quality_score", 0.0))
                # 품질 점수는 최대 1.0까지만 영향
                quality_boost = min(quality * 0.2, 1.0)
                score += quality_boost
            except ValueError:
                pass
            
            # 최종 점수 = 기본 점수 + 관련성 점수 + 품질 보너스
            final_score = score + relevance_score
            scored_docs.append((doc, final_score))
        
        # 스코어 순으로 정렬하여 상위 N개 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:self.top_n]]
