import time
import cloudscraper
import sys
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup, NavigableString

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
BASE_URL = config.ARCA_BASE_URL
SAVE_PATH = config.ARCA_RAW_PATH
FILTER_KEYWORDS = config.get_filter_keywords()
EXCLUDE_KEYWORDS = config.get_exclude_keywords()
QUALITY_THRESHOLD = config.ARCA_QUALITY_THRESHOLD
# ──────────────────────────────────────────────

# 날짜 확인 함수
def is_valid_date(date_text):
    """날짜가 유효한지 확인 (2025년 이후만 유효)"""
    # "[날짜 없음]"인 경우 유효하지 않음
    if date_text == "[날짜 없음]":
        return False
    
    # 2025년 확인 (포맷: "2025-05-12")
    return date_text.startswith("2025")

# 📌 Cloudflare 우회용 세션 생성
def get_new_scraper():
    """Cloudflare 보호를 우회하는 스크래퍼 생성"""
    try:
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        scraper.headers.update(HEADERS)
        return scraper
    except Exception as e:
        # 기본 스크래퍼로 대체
        return cloudscraper.create_scraper()

# 📌 1. 게시글 리스트 추출 (한 페이지)
def get_post_list(page_num):
    """아카라이브에서 게시글 목록 가져오기"""
    url = f"{BASE_URL}/b/dunfa?category=공략&p={page_num}"
    try:
        scraper = get_new_scraper()
        resp = scraper.get(url, timeout=config.ARCA_CRAWLER_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")
        posts = soup.select("a.vrow")
        return posts
    except Exception as e:
        return []

# 📌 2. 게시글 URL 및 제목 추출
def parse_post_info(post):
    """게시글에서 URL과 제목 추출"""
    href = post.get("href", "").split("?")[0]
    if not href.startswith("/b/"):
        return None, None
        
    post_url = BASE_URL + href
    
    # 제목 키워드 추출
    title_tag = post.select_one("span.title")
    if not title_tag:
        return post_url, None
        
    title_text = title_tag.get_text(strip=True)
    
    return post_url, title_text

# 📌 3. 게시글 본문 크롤링 및 제목 필터링
def crawl_post_content(post_url, visited_urls, depth=0, max_depth=2):
    """게시글 내용 크롤링 및 재귀적으로 링크 탐색"""
    # 증분 크롤링: 이미 방문한 URL이면 건너뜀
    if not should_process_url(post_url, visited_urls):
        return []

    # 방문 기록에 추가
    visited_urls.add(post_url)
    results = []
    
    try:
        # 게시글 내용 가져오기
        scraper = get_new_scraper()
        resp = scraper.get(post_url, timeout=config.ARCA_CRAWLER_TIMEOUT)
        soup = BeautifulSoup(resp.text, "html.parser")

        # 제목 추출
        title_tag = soup.select_one("div.title-row .title")
        if title_tag:
            title_text = ''.join(
                t for t in title_tag.contents if isinstance(t, NavigableString)
            ).strip()
        else:
            title_text = "[제목 없음]"
        
        # 날짜 추출
        date_tag = soup.select_one("div.article-info-section .date time")
        if date_tag:
            raw = date_tag.get("datetime")
            date_text = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d")
        else:
            date_text = "[날짜 없음]"

        # 2025년 게시글만 허용
        if not is_valid_date(date_text):
            return []

        # 조회수 추출
        hit_count = 0
        hit_tag = soup.select_one(
            "div.article-info-section span.head:-soup-contains('Views') + span.body"
        )
        if hit_tag:
            try:
                hits_text = hit_tag.get_text(strip=True)
                hit_count = int(hits_text.replace(",", ""))
            except ValueError:
                hit_count = 0

        # 추천수 추출
        like_count = 0
        like_tag = soup.select_one(
            "div.article-info-section span.head:-soup-contains('Like') + span.body"
        )
        if like_tag:
            try:
                like_text = like_tag.get_text(strip=True)
                like_count = int(like_text.replace(",", ""))
            except ValueError:
                like_count = 0

        # 본문 추출
        content_div = soup.select_one("div.fr-view.article-content")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"
        
        # 콘텐츠 품질 점수 계산
        content_score = calculate_content_score(content_text, title_text)
        
        # 품질 임계값 이상의 게시글만 저장
        if content_score >= QUALITY_THRESHOLD:
            item = build_item(
                source="arca",
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
                if linked_href.startswith("/b/dunfa"):
                    # 링크 텍스트(제목) 추출
                    link_text = a.get_text(strip=True)
                    
                    # 키워드 필터링
                    if not filter_by_keywords(link_text, FILTER_KEYWORDS, EXCLUDE_KEYWORDS):
                        continue
                    
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, visited_urls, depth + 1, max_depth))

        # 요청 간 딜레이 (아카라이브는 더 긴 딜레이 필요)
        time.sleep(config.ARCA_CRAWLER_DELAY)

    except Exception as e:
        pass

    return results

# 📌 4. 전체 크롤링 실행
def crawl_arca(max_pages=2, max_depth=2, visited_urls=None, is_incremental=True):
    """아카라이브 전체 크롤링 실행"""
    # 증분 크롤링을 위한 방문 URL 관리
    if visited_urls is None:
        visited_urls = set()
    
    results = []
    notice_processed = False
    start_time = time.time()
    
    try:
        # 페이지별 크롤링
        for page in range(1, max_pages + 1):
            posts = get_post_list(page)

            # 게시글별 처리
            for post in posts:
                post_url, title_text = parse_post_info(post)
                if not post_url or not title_text:
                    continue
                    
                # 키워드 기반 필터링
                if not filter_by_keywords(title_text, FILTER_KEYWORDS, EXCLUDE_KEYWORDS):
                    continue

                # 공지글 확인
                is_notice = 'notice' in post.get("class", [])

                # 공지글은 한 번만 처리
                if is_notice:
                    if not notice_processed:
                        results.extend(crawl_post_content(post_url, visited_urls, depth=0, max_depth=max_depth))
                    continue
                else:
                    # 일반 게시글 처리
                    results.extend(crawl_post_content(post_url, visited_urls, depth=0, max_depth=max_depth))

            # 공지글 처리 상태 업데이트
            if not notice_processed:
                notice_processed = True

        # 결과 요약
        elapsed_time = time.time() - start_time
        avg_time_per_post = elapsed_time / len(results) if results else 0

        # 결과 저장 (증분 처리 지원)
        from crawler_utils import save_arca_data
        save_arca_data(results, append=is_incremental)
        
    except Exception as e:
        pass
    
    return results

# 스크립트 직접 실행 시
if __name__ == "__main__":
    # 테스트 실행
    crawl_arca(max_pages=10, max_depth=0)
