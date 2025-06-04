"""
RAG 시스템을 위한 검색기 초기화 유틸리티
"""
from typing import List
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from utils import get_logger


class SearcherFactory:
    """검색기 생성을 담당하는 팩토리 클래스"""
    
    @staticmethod
    def create_bm25_data_from_vectordb(vectordb: Chroma) -> List[Document]:
        """VectorDB에서 BM25용 데이터 추출"""
        logger = get_logger(__name__)
        logger.info("🔄 VectorDB에서 BM25용 데이터 추출 중...")
        
        store_data = vectordb.get(include=["documents", "metadatas"])
        docs_for_bm25 = []
        
        for txt, meta in zip(store_data["documents"], store_data["metadatas"]):
            enhanced_content = txt
            
            # 메타데이터로 컨텐츠 강화 (직업 정보 중심)
            if meta:
                # 직업 정보 강화 (가장 중요)
                if class_name := meta.get("class_name"):
                    # 직업명 1회만 prepend 해서 BM25 토큰에 확실히 포함시키기
                    enhanced_content = f"{class_name}\n{enhanced_content}"
                
                # 품질 점수는 보조적으로만 사용
                try:
                    quality_score = float(meta.get("quality_score", 0.0))
                    if quality_score > 3.0:  # 고품질 문서
                        enhanced_content += "\n추천문서"
                except (ValueError, TypeError):
                    pass
            
            docs_for_bm25.append(Document(page_content=enhanced_content, metadata=meta))
        
        logger.info(f"✅ BM25용 문서 {len(docs_for_bm25)}개 준비 완료")
        return docs_for_bm25
    
    @staticmethod
    def create_bm25_retriever(docs_for_bm25: List[Document], k: int = 35) -> BM25Retriever:
        """BM25 검색기 생성"""
        bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
        bm25_retriever.k = k
        return bm25_retriever
    
    @staticmethod
    def create_cross_encoder_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> HuggingFaceCrossEncoder:
        """CrossEncoder 모델 생성"""
        return HuggingFaceCrossEncoder(
            model_name=model_name, 
            model_kwargs={"device": "cpu"}
        )
