import time
import requests
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# 상위 디렉토리의 config 및 crawler_utils import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from crawler_utils import (
    build_item, calculate_content_score, 
    should_process_url, filter_by_keywords
)

# ──────────────────────────────────────────────
# config에서 설정 가져오기
HEADERS = config.get_crawler_headers()
BASE_URL = config.OFFICIAL_BASE_URL
SAVE_PATH = config.OFFICIAL_RAW_PATH
FILTER_KEYWORDS = config.get_filter_keywords()
EXCLUDE_KEYWORDS = config.get_exclude_keywords()
QUALITY_THRESHOLD = config.OFFICIAL_QUALITY_THRESHOLD

# ──────────────────────────────────────────────
GUIDE_BASE   = f"{BASE_URL}/guide?no="
GUIDE_IDS    = [1512, 1508, 1515, 1479, 1478, 1475, 1483, 1480, 1484, 1516, 1510, 1486, 1487, 1490, 1485, 1489, 1488]          # ← 필요하면 여기만 늘려 주세요
GUIDE_QTHOLD = config.GUIDE_QUALITY_THRESHOLD   # guide도 저장할 최소 품질
# ──────────────────────────────────────────────

# 날짜 확인 함수
def is_valid_date(date_text):
    """날짜가 유효한지 확인 (2025년 이후만 유효)"""
    # "[날짜 없음]"인 경우 유효하지 않음
    if date_text == "[날짜 없음]":
        return False
    
    # 2025년 확인 (포맷: "2025-05-12")
    return date_text.startswith("2025")

# 📌 1. 게시글 리스트 추출 (한 페이지)
def get_post_list(page_num, session):
    """공식 사이트에서 게시글 목록 가져오기"""
    url = f"{BASE_URL}/community/dnfboard/list?category=99&page={page_num}"
    try:
        resp = session.get(url, timeout=config.CRAWLER_TIMEOUT)
        resp.raise_for_status()  # HTTP 오류 체크
        soup = BeautifulSoup(resp.text, "html.parser")
        posts = soup.select("article.board_list > ul")
        return posts
    except requests.exceptions.RequestException as e:
        return []

# 📌 2. 게시글 URL 및 제목 추출
def parse_post_info(post):
    """게시글에서 URL과 제목 추출"""
    title_li = post.select_one("li.title")
    if not title_li:
        return None, None

    link_tag = title_li.find_all("a")[-1]
    href = link_tag.get("href", "").strip()
    if href.startswith("/community/dnfboard/article/"):
        post_url = BASE_URL + href
    else:
        return None, None
        
    # 제목 추출
    title_text = link_tag.get_text(strip=True)
    
    return post_url, title_text

# 📌 3. 게시글 본문 크롤링 (본문 내 URL도 재귀 크롤링)
def crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=2):
    """게시글 내용 크롤링 및 재귀적으로 링크 탐색"""
    # 증분 크롤링: 이미 방문한 URL이면 건너뜀
    if not should_process_url(post_url, visited_urls):
        return []

    # 방문 기록에 추가
    visited_urls.add(post_url)
    results = []

    try:
        # 게시글 내용 가져오기
        resp = session.get(post_url, timeout=config.CRAWLER_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 제목 추출
        title_tag = soup.select_one("p.commu1st span")
        title_text = title_tag.get_text(strip=True) if title_tag else "[제목 없음]"
        title_text = re.sub(r"\s+", " ", title_text)

        # 날짜 추출
        date_tag = soup.select_one("ul.commu2nd span.date")
        if date_tag:
            raw = date_tag.get_text(strip=True)
            
            # 수정일 우선, 없으면 등록일 사용
            if "수정 :" in raw:
                date_part = raw.split("수정 :")[1].strip()
            else:
                date_match = re.search(r"등록 : (\d{4}\.\d{2}\.\d{2})", raw)
                date_part = date_match.group(1) if date_match else "[날짜 없음]"
            
            # 날짜만 추출하고 형식 변환 (YYYY-MM-DD)
            if date_part != "[날짜 없음]":
                date_text = date_part.split(" ")[0].replace(".", "-")
            else:
                date_text = "[날짜 없음]"
        else:
            date_text = "[날짜 없음]"

        # 2025년 게시글만 허용
        if not is_valid_date(date_text):
            return []
        
        # 조회수 추출
        hit_count = 0
        hit_tag = soup.select_one("ul.commu2nd li span.hits")
        if hit_tag:
            try:
                hit_text = hit_tag.get_text(strip=True)
                hit_count = int(hit_text.replace(',', ''))
            except ValueError:
                hit_count = 0
        
        # 좋아요 수 추출
        like_count = 0
        like_tag = soup.select_one("ul.commu2nd li span.like")
        if like_tag:
            try:
                like_count = int(like_tag.get_text(strip=True).replace(',', '') or 0)
            except ValueError:
                like_count = 0

        # 본문 추출
        content_div = soup.select_one("div.bd_viewcont")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"
        
        # 콘텐츠 품질 점수 계산 (utils.py의 calculate_content_score 사용)
        content_score = calculate_content_score(content_text, title_text)
        
        # 품질 임계값 이상의 게시글만 저장
        if content_score >= QUALITY_THRESHOLD:
            item = build_item(
                source="official",
                url=post_url,
                title=title_text,
                body=content_text,
                date=date_text,
                views=hit_count,
                likes=like_count
            )
            results.append(item)

        # 🔁 본문 내 추가 게시글 링크 (depth 제한 포함)
        if content_div and depth < max_depth:
            for a in content_div.find_all("a", href=True):
                linked_href = a["href"]
                if linked_href.startswith("/community/dnfboard/article/"):
                    # 링크 텍스트(제목) 추출
                    link_text = a.get_text(strip=True)

                    # 키워드 필터링 (utils.py의 filter_by_keywords 사용)
                    if not filter_by_keywords(link_text, FILTER_KEYWORDS, EXCLUDE_KEYWORDS):
                        continue
                    
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, session, visited_urls, depth + 1, max_depth))

        # 요청 간 딜레이
        time.sleep(config.CRAWLER_DELAY)

    except requests.exceptions.RequestException as e:
        pass
    except Exception as e:
        pass

    return results

def crawl_guide_page(guide_no, session):
    """
    단일 가이드 페이지 크롤링
    - <article class="content gg_template"> 안의 전체 텍스트를 본문으로 사용
    - 마지막 업데이트 일자를 date 필드에 저장 (YYYY-MM-DD)
    """
    url = f"{GUIDE_BASE}{guide_no}"
    try:
        resp = session.get(url, timeout=config.CRAWLER_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.select_one("article.content.gg_template")
    if not article:
        return None

    # ① 제목
    title_tag = article.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else f"[가이드] no={guide_no}"

    # ② 본문 - 이미지 ALT 포함, <br> → \n
    for br in article.find_all("br"):
        br.replace_with("\n")
    body_text = article.get_text("\n", strip=True)

    # ③ 날짜
    date_tag = article.select_one("div.last_update")
    date_text = "[날짜 없음]"
    if date_tag:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", date_tag.get_text())
        if m:
            date_text = m.group(1)

    # ④ 품질 스코어 & 필터
    score = calculate_content_score(body_text, title)
    if score < GUIDE_QTHOLD:
        return None

    return build_item(
        source="guide",
        url=url,
        title=title,
        body=body_text,
        date=date_text,
        views=0,
        likes=0
    )

# 📌 4. 전체 크롤링 실행
def crawl_df(max_pages=2, max_depth=2, visited_urls=None, is_incremental=True):
    """공식 사이트 전체 크롤링 실행"""
    # 증분 크롤링을 위한 방문 URL 관리
    if visited_urls is None:
        visited_urls = set()
    
    # 결과 및 세션 초기화
    session = requests.Session()
    session.headers.update(HEADERS)
    results = []
    notice_processed = False
    start_time = time.time()

    try:
        # 페이지별 크롤링
        for page in range(1, max_pages + 1):
            posts = get_post_list(page, session)

            # 게시글별 처리
            for post in posts:
                post_url, title_text = parse_post_info(post)
                if not post_url or not title_text:
                    continue
                    
                # 키워드 기반 필터링
                if not filter_by_keywords(title_text, FILTER_KEYWORDS, EXCLUDE_KEYWORDS):
                    continue

                # 공지글 / 일반글 구분
                is_notice = 'notice' in post.get("class", [])

                # 공지글은 한 번만 처리
                if is_notice:
                    if not notice_processed:
                        results.extend(crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=max_depth))
                    continue
                else:
                    # 일반 게시글 처리
                    results.extend(crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=max_depth))

            # 공지글 처리 상태 업데이트
            if not notice_processed:
                notice_processed = True
            
        # ── 게시판 크롤링 끝난 뒤 ───────────────────
        # ② 공식 가이드 크롤링 : 필요할 때 사용
        for gid in GUIDE_IDS:
            item = crawl_guide_page(gid, session)
            if item:
                item["quality_score"] = 9.0
                results.append(item)

        # 결과 요약
        elapsed_time = time.time() - start_time
        avg_time_per_post = elapsed_time / len(results) if results else 0

        # 결과 저장 (증분 처리 지원)
        from crawler_utils import save_official_data
        save_official_data(results, append=is_incremental)
        
    except Exception as e:
        pass
    
    return results

# 스크립트 직접 실행 시
if __name__ == "__main__":
    # 테스트 실행
    crawl_df(max_pages=1, max_depth=1)
