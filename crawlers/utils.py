# utils.py 수정안
import math, re, json, logging
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

# 로깅 설정
logger = logging.getLogger("crawler")

# 1. 직업명/각성명 사전 로드  (job_names.json 은 같은 폴더에 두세요)
_JOB_JSON = Path(__file__).with_name("job_names.json")
if _JOB_JSON.exists():
    _JOB_SET: set[str] = set(json.loads(_JOB_JSON.read_text(encoding="utf-8")))
else:  # 혹시 파일 없을 때 최소 직업 세트
    _JOB_SET = {}
_JOB_PATTERN = re.compile("|".join(map(re.escape, sorted(_JOB_SET, key=len, reverse=True))))

def detect_class_name(title: str, body: str | None = None) -> str | None:
    """제목/본문에서 직업·각성명이 보이면 반환, 없으면 None"""
    text = f"{title} {body or ''}"
    m = _JOB_PATTERN.search(text)
    return m.group(0) if m else None

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

# ────────────────── 우선순위 계산 ──────────────────
_SRC_WEIGHT = {"official":1.0, "youtube":0.9, "arca":0.75, "dcinside":0.7}

def _freshness_w(date: datetime, today: datetime) -> float:
    return max(0.0, 1 - (today - date).days/90)      # 90일 내 : 0~1

def _engage_w(views: int, likes: int) -> float:
    import math
    v = min(1.0, math.log1p(views)    / 10)
    l = min(1.0, math.log1p(likes)    /  7)
    return (v*0.7)+(l*0.3)             # 0~1

def calc_priority(
    *, source: str, date: datetime,
    views: int = 0, likes: int = 0,
    class_name: str | None = None,
    today: datetime | None = None,
    content_score: float = 0.0
) -> float:
    today = today or datetime.now(timezone.utc)
    
    # 콘텐츠 점수 반영 (0~1 사이로 정규화)
    content_weight = min(1.0, content_score / 100)
    
    return round(
        (_SRC_WEIGHT.get(source,0.5) * 4) +
        (_engage_w(views,likes) * 3) +
        (_freshness_w(date,today) * 1.5) +
        ((1.0 if class_name is None else 0.3) * 0.5) +
        (content_weight * 2),  # 콘텐츠 점수 가중치 반영
        2
    )

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

    # 5) 직업 이름 감지
    cls = detect_class_name(title, clean_body)
    
    # 6) 우선순위 점수 계산
    score = calc_priority(
        source=source, date=date_obj,
        views=views, likes=likes,
        class_name=cls,
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
        "priority_score": score,
        "content_score": content_score,
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