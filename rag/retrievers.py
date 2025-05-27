"""
RAG 시스템을 위한 검색기(Retriever) 클래스들
"""
from typing import List
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
        """메타데이터 기반 스코어링으로 문서 검색 (통합 품질 점수 + 쿼리 관련성)"""
        # 기본 검색기로 문서 검색
        docs = self.base_retriever.get_relevant_documents(query)
        
        # 메타데이터 기반 스코어링 (크롤링 통합 점수 + RAG 관련성 점수)
        scored_docs = []
        for doc in docs:
            score = 1.0
            meta = doc.metadata or {}
            
            # 통합된 콘텐츠 품질 점수 활용 (이미 사이트별 정규화+신선도+인기도 반영)
            try: 
                quality = float(meta.get("quality_score", 0.0))
                score += quality * 0.2  # 크롤링 시점 품질 점수 반영
            except ValueError: 
                pass
            
            # 쿼리 관련성: 직업명 일치 보너스 (가장 중요한 요소)
            if class_name := meta.get("class_name"):
                if isinstance(class_name, str):
                    query_lower = query.lower()
                    class_lower = class_name.lower()
                    
                    # 직업 매칭 정확도에 따른 차등 점수
                    if class_lower == query_lower:  # 완전 일치
                        score += 0.1
                    elif class_lower in query_lower or query_lower in class_lower:  # 부분 일치
                        score += 0.1
                    elif any(part.strip().lower() in query_lower for part in class_name.split('(') if part.strip()):  # 기본 직업명 매칭
                        score += 0.1
            
            scored_docs.append((doc, score))
        
        # 스코어 순으로 정렬하여 상위 N개 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:self.top_n]]
