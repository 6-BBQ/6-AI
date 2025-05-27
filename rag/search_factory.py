"""
RAG 시스템을 위한 검색기 초기화 유틸리티
"""
from typing import List
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma


class SearcherFactory:
    """검색기 생성을 담당하는 팩토리 클래스"""
    
    @staticmethod
    def create_bm25_data_from_vectordb(vectordb: Chroma) -> List[Document]:
        """VectorDB에서 BM25용 데이터 추출"""
        print("🔄 VectorDB에서 BM25용 데이터 추출 중...")
        
        store_data = vectordb.get(include=["documents", "metadatas"])
        docs_for_bm25 = []
        
        for txt, meta in zip(store_data["documents"], store_data["metadatas"]):
            enhanced_content = txt
            
            # 메타데이터로 컨텐츠 강화
            if meta:
                if meta.get("title"):
                    enhanced_content = f"제목: {meta['title']}\n{txt}"
                if meta.get("class_name"):  # VectorDB의 class_name은 그대로 유지
                    enhanced_content += f"\n직업: {meta['class_name']}"
            
            docs_for_bm25.append(Document(page_content=enhanced_content, metadata=meta))
        
        print(f"✅ BM25용 문서 {len(docs_for_bm25)}개 준비 완료")
        return docs_for_bm25
    
    @staticmethod
    def create_bm25_retriever(docs_for_bm25: List[Document], k: int = 40) -> BM25Retriever:
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
