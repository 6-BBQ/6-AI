"""
Langchain 호환 Gemini 임베딩 클래스
"""
import os
from typing import List, Optional
from langchain.embeddings.base import Embeddings
from google import genai
from google.genai import types
import time
import logging

logger = logging.getLogger(__name__)

class GeminiEmbeddings(Embeddings):
    """Langchain 호환 Gemini 임베딩 클래스"""
    
    def __init__(
        self,
        model: str = "gemini-embedding-exp-03-07",
        api_key: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        rate_limit_delay: float = 1.0
    ):
        """
        Args:
            model: Gemini 임베딩 모델명
            api_key: Gemini API 키 (None이면 환경변수에서 가져옴)
            task_type: 임베딩 태스크 타입 (RAG의 경우 RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY)
            rate_limit_delay: API 요청 간 지연시간 (초)
        """
        self.model = model
        self.task_type = task_type
        self.rate_limit_delay = rate_limit_delay
        
        # API 키 설정
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다!")
        
        # Gemini 클라이언트 초기화
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"🤖 Gemini 임베딩 초기화: {model} (task_type: {task_type})")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들의 임베딩 생성 (RETRIEVAL_DOCUMENT 태스크) - 배치 처리 최적화"""
        logger.info(f"📄 문서 임베딩 배치 생성: {len(texts)}개")
        
        embeddings = []
        batch_size = 200  # 배치 크기 설정
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logger.info(f"📦 배치 {batch_num}/{total_batches}: {len(batch_texts)}개 문서 처리 중...")
            
            try:
                # 배치로 한번에 임베딩 생성
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch_texts,  # 리스트로 한번에 전달
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                # 배치 결과에서 임베딩 벡터 추출
                for embedding in result.embeddings:
                    embeddings.append(embedding.values)
                
                logger.info(f"✅ 배치 {batch_num} 완료: {len(batch_texts)}개 임베딩 생성")
                
                # 배치 간 짧은 대기 (API 안정성)
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"❌ 배치 {batch_num} 실패: {e}")
                logger.info("⏸️ 5초 대기 후 재시도...")
                time.sleep(5.0)
                
                try:
                    # 재시도
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    
                    for embedding in result.embeddings:
                        embeddings.append(embedding.values)
                    
                    logger.info(f"✅ 배치 {batch_num} 재시도 성공")
                    
                except Exception as e2:
                    logger.error(f"❌ 배치 {batch_num} 재시도도 실패: {e2}")
                    # 실패한 경우 빈 벡터로 대체
                    for _ in range(len(batch_texts)):
                        if embeddings:
                            embedding_vector = [0.0] * len(embeddings[0])
                        else:
                            embedding_vector = [0.0] * 768  # 기본 차원
                        embeddings.append(embedding_vector)
        
        logger.info(f"✅ 문서 임베딩 배치 처리 완료: {len(embeddings)}개")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리의 임베딩 생성 (RETRIEVAL_QUERY 태스크)"""
        logger.debug(f"🔍 쿼리 임베딩 생성: '{text[:50]}...'")
        
        try:
            # RETRIEVAL_QUERY 태스크 타입으로 임베딩 생성
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            
            embedding_vector = result.embeddings[0].values
            logger.debug(f"✅ 쿼리 임베딩 완료: {len(embedding_vector)}차원")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"❌ 쿼리 임베딩 생성 실패: {e}")
            # 실패한 경우 기본 차원의 빈 벡터 반환
            return [0.0] * 768
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """비동기 문서 임베딩 (현재는 동기 버전과 동일)"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """비동기 쿼리 임베딩 (현재는 동기 버전과 동일)"""
        return self.embed_query(text)
