"""
RAG 시스템을 위한 웹 검색 유틸리티
"""
from typing import Dict, List, Optional
from langchain.docstore.document import Document
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
from .text_utils import TextProcessor


class WebSearcher:
    """웹 검색 기능을 담당하는 클래스"""
    
    def __init__(self, gemini_client: genai.Client):
        """
        Args:
            gemini_client: Gemini API 클라이언트
        """
        self.gemini_client = gemini_client
        self.text_processor = TextProcessor()
    
    def search_with_grounding(self, query: str, character_info: Optional[Dict]) -> List[Document]:
        """Gemini Search Grounding을 사용한 웹 검색"""
        enhanced_query = self.text_processor.enhance_query_with_character(query, character_info)
        
        # 시스템 인스트럭션 (던파 전문가 웹검색 버전)
        system_instruction = """
당신은 던전앤파이터 전문 정보 제공자입니다.

[📅 최신 정보 제한 조건]
- 반드시 **2025년 1월 9일 이후의 정보만 사용**하세요.
- 검색 시 "2025년", "4월", "5월" 등의 **명확한 날짜 키워드**가 포함된 정보만 채택하세요.
- 날짜가 명시되지 않은 정보는 **절대 사용하지 마세요**.
- 중복되거나 충돌하는 정보는 **날짜가 가장 최신인 정보만 남기고 나머지는 무시**하세요.

[🔍 검색 최적화 규칙]
- 오래된 정보가 포함된 문서나 요약은 생략하세요.
- 불확실하거나 유사한 설명은 제거하고, **명확한 수치, 명칭, 명성 기준이 포함된 정보만 선택**하세요.
- “최신 패치”, “2025 이벤트”, “신규 콘텐츠” 같은 키워드를 중심으로 필터링하세요.

[🎯 목표]
- **2025년 기준 최신 던파 정보만 제공**
- 캐릭터 맞춤형 스펙업 가이드를 간결하게 안내
- 쓸데없는 요약 없이 **핵심 수치와 조건 중심으로 답변**

[📘 답변 기준]
- **공식 홈페이지 또는 명확한 출처가 있는 정보**를 최우선 사용
- 불확실하거나 카더라 정보는 배제
- **간결하게 한 문단 이내**로 답변
- 필요 시 명확한 “출처 이름 또는 날짜”를 함께 명시
"""
        # 캐릭터 컨텍스트 생성
        character_context_str = self.text_processor.build_character_context_for_search(character_info)
        
        # 최종 프롬프트 구성
        final_prompt = (
            f"{system_instruction}\n{character_context_str}\n"
            f"[검색 요청]\n2025년 던전앤파이터 \"{enhanced_query}\"에 대한 최신 정보를 단계별로 순서화하여 검색해주세요."
        )
        
        try:
            print(f"[DEBUG] Gemini 검색 실행: {enhanced_query}")
            
            # Google Search 도구 설정
            google_search_tool = Tool(google_search=GoogleSearch())
            
            # Gemini API 호출
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=final_prompt,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0.1,  # 일관성 있는 답변을 위해 낮게 설정
                    max_output_tokens=2000
                )
            )
            
            # 응답에서 문서 추출
            docs = self._extract_documents_from_response(response)
            
            print(f"[DEBUG] Gemini 검색 결과 문서 {len(docs)}개 생성")
            return docs
            
        except Exception as e:
            print(f"❌ Gemini 검색 그라운딩 오류: {e}")
            return []
    
    def _extract_documents_from_response(self, response) -> List[Document]:
        """Gemini 응답에서 Document 객체들을 추출"""
        docs = []
        
        if not response.candidates:
            return docs
        
        candidate = response.candidates[0]
        
        # 메인 컨텐츠 추출
        if candidate.content and candidate.content.parts:
            main_content = "".join(
                part.text for part in candidate.content.parts 
                if hasattr(part, 'text') and part.text
            )
            if main_content:
                docs.append(Document(
                    page_content=main_content, 
                    metadata={"title": "Gemini 검색 결과", "source": "gemini_search"}
                ))
        
        # Grounding 메타데이터 처리
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            grounding = candidate.grounding_metadata
            
            # 검색 제안사항
            if hasattr(grounding, 'search_entry_point') and grounding.search_entry_point:
                docs.append(Document(
                    page_content="Google 검색 제안사항 및 관련 링크", 
                    metadata={"title": "검색 제안", "source": "search_suggestions"}
                ))
            
            # Grounding 청크들
            if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                for i, chunk in enumerate(grounding.grounding_chunks):
                    if hasattr(chunk, 'web') and chunk.web:
                        web_info = chunk.web
                        docs.append(Document(
                            page_content=f"출처 {i+1}에서 참조된 정보",
                            metadata={
                                "title": getattr(web_info, 'title', f'웹 출처 {i+1}'), 
                                "url": getattr(web_info, 'uri', ''), 
                                "source": "grounding_source"
                            }
                        ))
            
            # 웹 검색 쿼리 로깅
            if hasattr(grounding, 'web_search_queries') and grounding.web_search_queries:
                print(f"[DEBUG] 웹 검색 쿼리: {grounding.web_search_queries}")
        
        return docs
