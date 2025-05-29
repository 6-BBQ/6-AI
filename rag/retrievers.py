"""
RAG 시스템을 위한 검색기(Retriever) 클래스들
"""
from typing import List, Optional, Dict
from langchain.docstore.document import Document


class MetadataAwareRetriever:
    """메타데이터를 고려한 지능형 검색기"""
    
    def __init__(self, base_retriever, top_n: int = 25):
        """
        Args:
            base_retriever: 기본 검색기 (ContextualCompressionRetriever 등)
            top_n: 반환할 문서 수
        """
        self.base_retriever = base_retriever
        self.top_n = top_n
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """메타데이터 기반 스코어링으로 문서 검색 (품질 점수 기반)"""
        # 기본 검색기로 문서 검색
        docs = self.base_retriever.get_relevant_documents(query)
        
        # 메타데이터 기반 스코어링 (품질 점수 활용)
        scored_docs = []
        for doc in docs:
            score = 1.0
            meta = doc.metadata or {}
            
            # 통합된 콘텐츠 품질 점수 활용 (사이트별 정규화+신선도+인기도 반영)
            try: 
                quality = float(meta.get("quality_score", 0.0))
                score += quality * 0.2  # 크롤링 시점 품질 점수 반영
            except ValueError: 
                pass
            
            scored_docs.append((doc, score))
        
        # 스코어 순으로 정렬하여 상위 N개 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:self.top_n]]
