import re, logging
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import sys
from pathlib import Path

# 상위 디렉토리의 config 모듈 import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config

# 로깅 설정
logger = logging.getLogger("crawler")


# ────────────────── 텍스트 처리 유틸 ──────────────────
def clean_text(text):
    """텍스트 정리 (HTML 태그 제거, 연속 공백 제거 등)"""
    if not text:
        return ""
    
    # HTML 태그 제거
    if '<' in text and '>' in text:  # HTML로 보이는 경우만 처리
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 줄바꿈 표준화
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def calculate_content_score(text, title=""):
    """콘텐츠 품질 점수 계산 (0-100)"""
    # 텍스트가 없으면 0점
    if not text:
        return 0
    
    score = 0
    
    # 1. 길이 기반 점수 (최대 40점)
    length = len(text)
    if length < 100:
        length_score = length / 5  # 최대 20점
    elif length < 500:
        length_score = 20 + (length - 100) / 20  # 20-40점
    else:
        length_score = 40
    
    # 2. 구조 기반 점수 (최대 30점)
    # 줄바꿈 수 (문단 구분이 잘 된 텍스트는 점수 높음)
    newlines = text.count('\n')
    paragraphs = max(1, len([p for p in text.split('\n') if p.strip()]))
    structure_score = min(30, paragraphs + newlines/2)
    
    # 3. 키워드 기반 점수 (최대 30점)
    keywords = ["스펙업", "가이드", "공략", "추천", "팁", "노하우", "장비", "스킬", "종말의 숭배자",
                "상급 던전", "레이드", "에픽", "활용", "중천", "융합석", "뉴비", "레기온"]
    
    combined = (title + " " + text).lower()
    keyword_count = sum(1 for kw in keywords if kw.lower() in combined)
    keyword_score = min(30, keyword_count * 3)
    
    # 최종 점수 계산
    score = length_score + structure_score + keyword_score
    
    return min(100, score)

# ────────────────── 사이트별 정규화 기준 (config에서 가져옴) ──────────────────
def get_site_normalization():
    """config에서 사이트별 정규화 설정을 가져옴"""
    return config.get_site_normalization()

# ────────────────── 우선순위 계산 ──────────────────
_SRC_WEIGHT = {"official":1.0, "arca":0.9, "dcinside":0.9}

def _freshness_w(date: datetime, today: datetime) -> float:
    return max(0.0, 1 - (today - date).days/90)

def _engage_w(views: int, likes: int, source: str) -> float:
    """사이트별 정규화된 인기도 점수 계산 (현실적 비율 반영)"""
    import math
    
    # config에서 사이트별 정규화 기준 가져오기
    site_normalization = get_site_normalization()
    norm = site_normalization.get(source, {"views_base": 15000, "likes_base": 30})
    views_base = norm["views_base"]
    likes_base = norm["likes_base"]
    
    # 현실적인 좋아요/조회수 비율 체크
    if views > 0:
        actual_ratio = likes / views
        expected_range = norm.get("likes_ratio_range", [0.001, 0.010])  # JSON에서 가져오므로 list
        expected_range = tuple(expected_range)  # tuple로 변환
        
        # 비율이 현실적 범위를 벗어나면 페널티 적용
        ratio_penalty = 1.0
        if actual_ratio > expected_range[1] * 2:  # 너무 높은 비율
            ratio_penalty = 0.8
        elif actual_ratio < expected_range[0] * 0.5:  # 너무 낮은 비율
            ratio_penalty = 0.9
    else:
        ratio_penalty = 1.0
    
    # 사이트별 상대적 인기도로 정규화
    normalized_views = views / views_base
    normalized_likes = likes / likes_base
    
    # 로그 정규화 적용 (현실적 범위 고려)
    v = min(1.0, math.log1p(normalized_views) / 3.0)
    l = min(1.0, math.log1p(normalized_likes) / 2.5)
    
    # 비율 페널티 적용
    engagement = ((v * 0.6) + (l * 0.4)) * ratio_penalty
    
    return min(1.0, engagement)

def calc_quality_score(
    *, source: str, date: datetime,
    views: int = 0, likes: int = 0,
    today: datetime | None = None,
    content_score: float = 0.0
) -> float:
    """콘텐츠 품질 종합 점수 계산 (직업 점수 제외, 통합 점수)"""
    today = today or datetime.now(timezone.utc)
    
    # 콘텐츠 품질 점수 반영 (0~1 사이로 정규화)
    content_weight = min(1.0, content_score / 100)
    
    # 사이트별 정규화된 인기도 계산
    engagement_score = _engage_w(views, likes, source)
    
    # 통합 점수 계산 (최대 10점 정도)
    total_score = (
        (_SRC_WEIGHT.get(source, 0.5) * 2.5) +       # 사이트 영향 약화
        (engagement_score * 2.0) +                  # 인기도 영향 완화
        (_freshness_w(date, today) * 2.0) +         
        (content_weight * 3.5)                      # 콘텐츠 품질 강화
    )
    
    return round(total_score, 2)

# ────────────────── 크롤러별 저장 함수 ──────────────────
def save_official_data(data: list, append: bool = True):
    """공식 사이트 데이터 저장"""
    save_crawler_data(config.OFFICIAL_RAW_PATH, data, append)

def save_dc_data(data: list, append: bool = True):
    """디시인사이드 데이터 저장"""
    save_crawler_data(config.DC_RAW_PATH, data, append)

def save_arca_data(data: list, append: bool = True):
    """아카라이브 데이터 저장"""
    save_crawler_data(config.ARCA_RAW_PATH, data, append)

# ────────────────── 결과 dict 빌더 ──────────────────
def build_item(
    *, source: str, url: str, title: str, body: str,
    date: str, views: int = 0, likes: int = 0
) -> dict:
    
    # 1) 날짜 문자열 → datetime
    if isinstance(date, str):
        try:
            # "YYYY-MM-DD" 또는 "YYYY.MM.DD" 모두 대응
            clean = date.replace(".", "-")[:10]
            date_obj = datetime.strptime(clean, "%Y-%m-%d")
        except ValueError:
            # 파싱 실패 시 현재 시각으로 대체
            date_obj = datetime.now()
    else:
        date_obj = date

    # 2) tzinfo 없으면 UTC 로 부여
    if isinstance(date_obj, datetime) and date_obj.tzinfo is None:
        date_obj = date_obj.replace(tzinfo=timezone.utc)

    # 3) 본문 정리
    clean_body = clean_text(body)
    
    # 4) 콘텐츠 품질 점수 계산
    content_score = calculate_content_score(clean_body, title)
    
    # 6) 통합 품질 점수 계산 (직업 점수 제외)
    quality_score = calc_quality_score(
        source=source, date=date_obj,
        views=views, likes=likes,
        content_score=content_score
    )
    
    return {
        "url": url,
        "title": title,
        "date": date_obj.strftime("%Y-%m-%d"),
        "views": views, 
        "likes": likes,
        "class_name": None,  # 후처리에서 벡터 검색으로 분류 예정
        "source": source,
        "quality_score": quality_score,  # 통합된 단일 점수
        "body": clean_body,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ────────────────── 필터링 유틸 ──────────────────
def should_process_url(url, visited_urls=None):
    """URL 처리 여부 결정 (증분 크롤링 지원)"""
    if visited_urls is None:
        return True
    
    # 이미 방문한 URL이면 건너뛰기
    if url in visited_urls:
        return False
    
    return True

def filter_by_keywords(text, include_keywords, exclude_keywords):
    """키워드 기반 필터링"""
    if not text:
        return False
    
    # 제외 키워드 확인
    for keyword in exclude_keywords:
        if keyword in text:
            return False
    
    # 포함 키워드 확인
    for keyword in include_keywords:
        if keyword in text:
            return True
    
    return False

# ────────────────── 증분 저장 유틸 ──────────────────
def save_crawler_data(file_path: str, data: list, append: bool = True):
    """크롤링 데이터 증분 저장 함수"""
    import os
    import json
    from pathlib import Path
    
    if not data:
        logger.info(f"저장할 데이터가 없음: {file_path}")
        return
    
    # 디렉토리 생성
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if append and os.path.exists(file_path):
            # 기존 데이터 로드
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            
            # URL 중복 제거를 위한 기존 URL 집합
            existing_urls = {item.get('url') for item in existing_data if isinstance(item, dict) and 'url' in item}
            
            # 새로운 데이터 중 중복되지 않는 것만 추가
            new_data = [item for item in data if item.get('url') not in existing_urls]
            
            if new_data:
                final_data = existing_data + new_data
                logger.info(f"증분 저장: 기존 {len(existing_data)}개 + 새로운 {len(new_data)}개 = 총 {len(final_data)}개")
            else:
                final_data = existing_data
                logger.info(f"새로운 데이터 없음 (모두 중복): {file_path}")
        else:
            # 전체 저장 모드
            final_data = data
            logger.info(f"전체 저장: {len(final_data)}개 데이터")
        
        # 파일 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 데이터 저장 완료: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ 데이터 저장 실패 ({file_path}): {e}")
        # 실패 시 새 데이터만 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)