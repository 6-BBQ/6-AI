#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 텍스트 파일 → official_raw.json 전처리 도구
사용법: python txt_to_official.py <txt_파일_경로> [제목] [URL]
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from crawlers.utils import build_item, save_official_data
except ImportError:
    print("⚠️ crawlers/utils.py를 찾을 수 없습니다. 기본 전처리를 사용합니다.")
    
    def build_item(*, source, url, title, body, date, views=0, likes=0):
        """기본 build_item 함수"""
        return {
            "url": url,
            "title": title,
            "date": date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date,
            "views": views,
            "likes": likes,
            "class_name": None,
            "source": source,
            "quality_score": 6.0,
            "body": clean_text(body),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def save_official_data(data, append=True):
        """기본 save_official_data 함수"""
        save_path = project_root / "data/raw/official_raw.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if append and save_path.exists():
            with open(save_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_data.extend(data)
            final_data = existing_data
        else:
            final_data = data
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

def clean_text(text):
    """텍스트 정리 함수"""
    if not text:
        return ""
    
    # HTML 태그 제거
    if '<' in text and '>' in text:
        text = BeautifulSoup(text, "html.parser").get_text(separator="\n")
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 줄바꿈 정리
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 연속 줄바꿈 제거 (3개 이상 → 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def extract_title_from_content(content):
    """내용에서 제목 추출 시도"""
    lines = content.strip().split('\n')
    
    # 첫 번째 비어있지 않은 줄을 제목으로 사용
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # 너무 긴 경우 자르기
            if len(line) > 100:
                line = line[:97] + "..."
            return line
    
    return "던파 가이드 문서"

def process_txt_file(txt_path, title=None, url=None):
    """텍스트 파일을 처리하여 official_raw.json에 추가"""
    
    txt_path = Path(txt_path)
    
    if not txt_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {txt_path}")
        return False
    
    try:
        # 파일 읽기
        print(f"📖 파일 읽는 중: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print("❌ 파일이 비어있습니다.")
            return False
        
        # 제목 설정
        if not title:
            title = extract_title_from_content(content)
        
        # 내용 정리
        cleaned_content = clean_text(content)
        
        # build_item 사용해서 아이템 생성
        item = build_item(
            source="official",
            url=url,
            title=title,
            body=cleaned_content,
            date=datetime.now(timezone.utc),
            views=0,
            likes=0
        )
        
        print(f"✅ 전처리 완료:")
        print(f"   📄 제목: {title}")
        print(f"   🔗 URL: {url}")
        print(f"   📊 내용 길이: {len(cleaned_content)}자")
        print(f"   ⭐ 품질 점수: {item.get('quality_score', 'N/A')}")
        
        # official_raw.json에 추가
        print(f"💾 저장 중...")
        save_official_data([item], append=True)
        
        print(f"🎉 완료! official_raw.json에 추가되었습니다.")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python txt_to_official.py <txt_파일_경로> [제목] [URL]")
        print("\n예시:")
        print("  python txt_to_official.py guide.txt")
        print("  python txt_to_official.py guide.txt \"던파 스펙업 가이드\"")
        print("  python txt_to_official.py guide.txt \"던파 스펙업 가이드\" \"https://df.nexon.com/guide/specup\"")
        return
    
    txt_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    url = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("🔧 던파 텍스트 파일 전처리 도구")
    print("=" * 50)
    
    success = process_txt_file(txt_path, title, url)
    
    if success:
        print("\n✅ 전처리 완료! 이제 파이프라인을 실행하세요:")
        print("   python pipeline.py --skip-crawl")
    else:
        print("\n❌ 전처리 실패")

if __name__ == "__main__":
    main()
