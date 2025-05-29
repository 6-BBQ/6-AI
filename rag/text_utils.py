"""
RAG 시스템을 위한 텍스트 및 검색 처리 유틸리티
"""
from typing import Dict, List, Optional
from langchain.docstore.document import Document


class TextProcessor:
    """텍스트 처리 관련 기능을 담당하는 클래스"""
    
    @staticmethod
    def enhance_query_with_character(query: str, character_info: Optional[Dict]) -> str:
        """
        캐릭터 정보로 검색 쿼리 풍부하게 강화
        Korean + structured tokens  ➜  BM25 / 벡터 모두 높은 recall
        """
        if not character_info:
            return query

        parts = []
        add = parts.append          # 지역 변수로 바인딩(미세 최적화)

        # === 1) 직업
        if job := character_info.get("job"):
            add(f"직업::{job}")               # ex) 직업::레인저(여)
            # 영문 별칭(있을 때)
            if job_en := character_info.get("job_en"):
                add(f"job::{job_en}")         # ex) job::Female Ranger

        # === 2) 명성
        if fame := character_info.get("fame"):
            add(f"명성::{fame}")              # ex) 명성::12038

        # === 3) 무기
        if weapon := character_info.get("weapon"):
            add(f"무기::{weapon}")

        # === 4) 에픽·태초 개수
        if ep := character_info.get("epicNum"):
            add(f"에픽::{ep}")
        if ori := character_info.get("originalityNum"):
            add(f"태초::{ori}")

        # === 5) 세트 아이템
        if set_name := character_info.get("set_item_name"):
            rarity = character_info.get("set_item_rarity", "")
            add(f"세트::{set_name}{f'({rarity})' if rarity else ''}")

        # === 6) 칭호
        if title := character_info.get("title"):
            add(f"칭호::{title}")

        # === 7) 최종 조립
        if parts:
            enhanced = " | ".join(parts) + " | " + query
            print(f"[DEBUG] 쿼리 강화: '{query}' → '{enhanced}'")
            return enhanced
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
