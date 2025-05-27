"""
RAG 시스템을 위한 텍스트 및 검색 처리 유틸리티
"""
from typing import Dict, List, Optional
from langchain.docstore.document import Document


class TextProcessor:
    """텍스트 처리 관련 기능을 담당하는 클래스"""
    
    @staticmethod
    def enhance_query_with_character(query: str, character_info: Optional[Dict]) -> str:
        """캐릭터 정보로 검색 쿼리 강화 (FastAPI에서 변환된 키 사용)"""
        if not character_info:
            return query
        
        enhancements = []
        # FastAPI에서 변환된 'job' 키 사용
        if job_info := character_info.get('job'):
            enhancements.append(job_info)
        if fame := character_info.get('fame'):
            enhancements.append(str(fame))
        
        if enhancements:
            enhanced_query = f"{' '.join(enhancements)} {query}"
            print(f"[DEBUG] 쿼리 강화: '{query}' → '{enhanced_query}'")
            return enhanced_query
        return query
    
    @staticmethod
    def format_docs_to_context_string(docs: List[Document], context_type: str) -> str:
        """문서 리스트를 컨텍스트 문자열로 변환"""
        context_parts = []
        for i, doc in enumerate(docs):
            content = f"[{context_type} 문서 {i+1}] {doc.page_content}"
            if doc.metadata and (url := doc.metadata.get("url")):
                content += f"\n참고 링크: {url}"
            context_parts.append(content)
        return "\n\n".join(context_parts)
    

    
    @staticmethod
    def build_character_context_for_llm(character_info: Optional[Dict]) -> str:
        """캐릭터 정보를 LLM용 컨텍스트로 변환"""
        if not character_info:
            return "캐릭터 정보 없음."
        
        details = []
        if job_info := character_info.get('job'):
            details.append(f"- 직업: {job_info}")
        if fame_info := character_info.get('fame'):
            details.append(f"- 명성: {fame_info}")
        if weapon_info := character_info.get('weapon'):
            details.append(f"- 무기: {weapon_info}")
        if epic_num := character_info.get('epicNum'):
            details.append(f"- 에픽 아이템 개수: {epic_num}")
        if originality_num := character_info.get('originalityNum'):
            details.append(f"- 태초 아이템 개수: {originality_num}")
        if title_info := character_info.get('title'):
            details.append(f"- 칭호: {title_info}")
        if set_item_name := character_info.get('set_item_name'):
            set_rarity = character_info.get('set_item_rarity', '')
            details.append(f"- 세트 아이템: {set_item_name} ({set_rarity} 등급)")
        if creature_info := character_info.get('creature'):
            details.append(f"- 크리쳐: {creature_info}")
        if aura_info := character_info.get('aura'):
            details.append(f"- 오라: {aura_info}")

        if details:
            char_context = "사용자 캐릭터 정보:\n" + "\n".join(details)
            char_context += "\n\n위 캐릭터 정보를 고려하여 맞춤형 조언을 제공하세요."
            return char_context
        else:
            return "캐릭터 정보가 제공되었으나, 세부 내용을 파악할 수 없습니다."
    
    @staticmethod
    def build_character_context_for_search(character_info: Optional[Dict]) -> str:
        """캐릭터 정보를 검색용 컨텍스트로 변환 (간략한 버전)"""
        if not character_info:
            return "캐릭터 정보가 제공되지 않았습니다."
        
        details = []
        if job_info := character_info.get('job'):
            details.append(f"- 직업: {job_info}")
        if fame_info := character_info.get('fame'):
            details.append(f"- 명성: {fame_info}")
        
        if details:
            character_context = "사용자 캐릭터 정보:\n" + "\n".join(details)
            character_context += "\n\n위 캐릭터 정보를 고려하여 맞춤형 정보를 검색하세요."
            return character_context
        else:
            return "캐릭터 정보가 제공되었으나, 세부 내용을 파악할 수 없습니다."
