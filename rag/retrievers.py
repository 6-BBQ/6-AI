"""
RAG 시스템을 위한 검색기(Retriever) 클래스들
"""
from typing import List
from langchain.docstore.document import Document


class MetadataAwareRetriever:
    """메타데이터를 고려한 지능형 검색기"""
    
    def __init__(self, base_retriever, top_n: int = 15):
        """
        Args:
            base_retriever: 기본 검색기 (ContextualCompressionRetriever 등)
            top_n: 반환할 문서 수
        """
        self.base_retriever = base_retriever
        self.top_n = top_n
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """메타데이터 기반 스코어링으로 문서 검색"""
        # 기본 검색기로 문서 검색
        docs = self.base_retriever.get_relevant_documents(query)
        
        # 메타데이터 기반 스코어링
        scored_docs = []
        for doc in docs:
            score = 1.0
            meta = doc.metadata or {}
            
            # 조회수 기반 스코어링
            try: 
                views = int(meta.get("views", 0))
                score += 0.2 if views > 100000 else (0.1 if views > 10000 else 0)
            except ValueError: 
                pass
            
            # 좋아요 기반 스코어링
            try: 
                likes = int(meta.get("likes", 0))
                score += 0.1 if likes > 100 else (0.05 if likes > 50 else 0)
            except ValueError: 
                pass
            
            # 우선순위 점수 반영
            try: 
                priority = float(meta.get("priority_score", 0.0))
                score += priority * 0.1
            except ValueError: 
                pass
            
            # 콘텐츠 점수 반영
            try: 
                content_score = float(meta.get("content_score", 0.0))
                score += content_score * 0.01
            except ValueError: 
                pass
            
            # 직업명 일치 보너스 (JSON 데이터에서는 class_name으로 저장됨)
            if class_name := meta.get("class_name"):
                # query에서 job 정보를 추출하여 비교 (예: '레인저(여)' 또는 '레인저' 등)
                if isinstance(class_name, str):
                    # 정확한 직업명 매칭 또는 부분 매칭 확인
                    if (class_name.lower() in query.lower()) or \
                       any(part.strip() in query.lower() for part in class_name.split('(') if part.strip()):
                        score += 0.3
            
            scored_docs.append((doc, score))
        
        # 스코어 순으로 정렬하여 상위 N개 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:self.top_n]]
