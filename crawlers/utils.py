# utils.py 수정안
import math, re, json, logging
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional
from rapidfuzz import process
from kiwipiepy import Kiwi

# 로깅 설정
logger = logging.getLogger("crawler")

_JOB_JSON = Path(__file__).with_name("job_names.json")
kiwi = Kiwi()

try:
    _JOB_LIST: list[str] = json.loads(_JOB_JSON.read_text(encoding="utf-8"))
    _JOB_SET: set[str] = set(_JOB_LIST)
except Exception as e:
    print(f"⚠️ 직업명 JSON 로딩 오류: {e}")
    _JOB_LIST = []
    _JOB_SET = set()

_BOUNDARY = r"(?<![가-힣A-Za-z0-9])(?:{names})(?![가-힣A-Za-z0-9])"
_JOB_PATTERN = re.compile(
    _BOUNDARY.format(
        names="|".join(map(re.escape, sorted(_JOB_SET, key=len, reverse=True)))
    ),
    re.IGNORECASE,
)

CONTEXT_KEYWORDS = {"공략", "가이드", "세팅", "스펙업", "뉴비", "팁"}

def detect_class_name(title: str, body: Optional[str] = None) -> Optional[str]:
    if not title:
        return None

    text = f"{title} {body or ''}".lower()

    # 1️⃣ 정규식 기반 직업 탐지
    m = _JOB_PATTERN.search(text)
    if m:
        return m.group(0)

    # 2️⃣ Kiwi 기반 토큰화
    try:
        tokens = set(word.form for word in kiwi.tokenize(text))
        hit = _JOB_SET.intersection(tokens)
        if hit:
            return next(iter(hit))
    except Exception as e:
        logger.warning(f"Kiwi 오류: {e}")

    # 3️⃣ Fuzzy Matching (조건부)
    if len(title) < 25 and any(k in text for k in CONTEXT_KEYWORDS):
        result = process.extractOne(
            title.replace(" ", ""), _JOB_LIST, score_cutoff=92
        )
        if result:  # None 체크 추가
            match, score, _ = result
            if score >= 92:
                return match

    return None

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

# ────────────────── 사이트별 정규화 기준 ──────────────────
SITE_NORMALIZATION = {
    "youtube": {"views_base": 50000, "likes_base": 500},
    "arca": {"views_base": 5000, "likes_base": 50}, 
    "dcinside": {"views_base": 1000, "likes_base": 10},
    "official": {"views_base": 10000, "likes_base": 100}  # 공홈은 중간값
}

# ────────────────── 우선순위 계산 ──────────────────
_SRC_WEIGHT = {"official":1.0, "youtube":0.9, "arca":0.75, "dcinside":0.7}

def _freshness_w(date: datetime, today: datetime) -> float:
    return max(0.0, 1 - (today - date).days/180)      # 180일 내 : 0~1

def _engage_w(views: int, likes: int, source: str) -> float:
    """사이트별 정규화된 인기도 점수 계산"""
    import math
    
    # 사이트별 정규화 기준 가져오기
    norm = SITE_NORMALIZATION.get(source, {"views_base": 10000, "likes_base": 100})
    views_base = norm["views_base"]
    likes_base = norm["likes_base"]
    
    # 사이트별 상대적 인기도로 정규화
    normalized_views = views / views_base
    normalized_likes = likes / likes_base
    
    # 로그 정규화 적용
    v = min(1.0, math.log1p(normalized_views) / 2.5)  # 0~1 사이
    l = min(1.0, math.log1p(normalized_likes) / 2.0)   # 0~1 사이
    
    return (v * 0.7) + (l * 0.3)  # 0~1

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
        (_SRC_WEIGHT.get(source, 0.5) * 3.0) +     # 출처 신뢰도 (0~3.0)
        (engagement_score * 2.5) +                  # 사이트별 정규화된 인기도 (0~2.5)
        (_freshness_w(date, today) * 2.0) +         # 신선도 (0~2.0)
        (content_weight * 2.5)                      # 콘텐츠 품질 (0~2.5)
    )
    
    return round(total_score, 2)

def load_yt_ids(path: str | Path) -> list[str]:
    if not path:
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            logger.info(f"✅ 유튜브 ID 파일 로드 성공: {len(ids)}개 ID 찾음")
            return ids
    except Exception as e:
        logger.error(f"❌ 유튜브 ID 파일 로드 실패: {e}")
        # 절대 경로로 시도
        try:
            project_root = Path(__file__).resolve().parents[1]
            abs_path = project_root / path
            logger.info(f"🔄 절대 경로로 재시도: {abs_path}")
            with open(abs_path, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                logger.info(f"✅ 유튜브 ID 파일 로드 성공: {len(ids)}개 ID 찾음")
                return ids
        except Exception as e2:
            logger.error(f"❌ 절대 경로 시도도 실패: {e2}")
            return []

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

    # 5) 직업 이름 감지 (메타데이터용만, 점수에는 미반영)
    cls = detect_class_name(title, clean_body)
    
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
        "class_name": cls,
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