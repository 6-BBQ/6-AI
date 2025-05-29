# dc_crawler.py (개선 버전)
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from utils import (
    build_item, clean_text, calculate_content_score,
    should_process_url, filter_by_keywords
)

# ──────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://gall.dcinside.com"
SAVE_PATH = "data/raw/dc_raw.json"

# 필터 키워드 (중요도에 따라 정렬)
FILTER_KEYWORDS = [
    "명성", "상급 던전", "스펙", "장비", "파밍", "뉴비", "융합석", "중천", "세트",
    "가이드", "에픽", "태초", "레기온", "레이드", "현질", "세리아", "마법부여", 
    "스킬트리", "종말의 숭배자", "베누스", "나벨"
]

# 제외 키워드
EXCLUDE_KEYWORDS = [
    "이벤트", "선계", "커스텀", "카지노", "기록실", "서고", "바칼", "이스핀즈", 
    "어둑섬", "깨어난 숲", "ㅅㅂ", "ㅂㅅ", "ㅄ", "ㅗ", "시발", "씨발", "병신", "좆"
]

# 품질 점수 임계값 (이 점수 이상인 게시글만 저장)
QUALITY_THRESHOLD = 20  # 디시는 글이 짧은 경향이 있어 낮게 설정
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
    """디시인사이드에서 게시글 목록 가져오기"""
    url = f"{BASE_URL}/mgallery/board/lists/?id=dfip&sort_type=N&exception_mode=recommend&search_head=10&page={page_num}"
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        posts = soup.select("tr.ub-content.us-post")
        return posts
    except requests.exceptions.RequestException as e:
        return []

# 📌 2. 게시글 URL 및 제목 추출
def parse_post_info(post):
    """게시글에서 URL과 제목 추출"""
    link_tag = post.select_one("td.gall_tit a[href*='view']")
    if not link_tag:
        return None, None
    
    post_url = BASE_URL + link_tag["href"]
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
        resp_post = session.get(post_url, timeout=10)
        resp_post.raise_for_status()
        soup = BeautifulSoup(resp_post.text, "html.parser")

        # 제목 추출
        title_tag = soup.select_one(".title_subject")
        title_text = title_tag.get_text(strip=True) if title_tag else "[제목 없음]"

        # 날짜 추출
        date_tag = soup.select_one("span.gall_date")
        if date_tag:
            raw = date_tag.get_text(strip=True)
            date_text = raw[:10].replace(".", "-")
        else:
            date_text = "[날짜 없음]"

        # 2025년 게시글만 허용
        if not is_valid_date(date_text):
            return []
        
        # 조회수 추출
        hit_count = 0
        hit_tag = soup.select_one("span.gall_count")
        if hit_tag:
            try:
                hit_text = hit_tag.get_text(strip=True).replace('조회', '').strip()
                hit_count = int(hit_text.replace(',', ''))
            except ValueError:
                hit_count = 0
        
        # 좋아요 수 추출
        like_count = 0
        like_tag = soup.select_one("span.gall_reply_num")
        if like_tag:
            try:
                like_text = like_tag.get_text(strip=True).replace('추천', '').strip()
                like_count = int(like_text.replace(',', ''))
            except ValueError:
                like_count = 0

        # 본문 추출
        content_div = soup.select_one("div.write_div")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"
        
        # 콘텐츠 품질 점수 계산
        content_score = calculate_content_score(content_text, title_text)
        
        # 품질 임계값 이상의 게시글만 저장
        if content_score >= QUALITY_THRESHOLD:
            item = build_item(
                source="dcinside",
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
                if linked_href.startswith("/mgallery/board/lists/?id=dfip"):
                    # 링크 텍스트(제목) 추출
                    link_text = a.get_text(strip=True)
                    
                    # 키워드 필터링
                    if not filter_by_keywords(link_text, FILTER_KEYWORDS, EXCLUDE_KEYWORDS):
                        continue
                    
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, session, visited_urls, depth + 1, max_depth))

        # 요청 간 딜레이
        time.sleep(0.05)

    except requests.exceptions.RequestException as e:
        pass
    except Exception as e:
        pass

    return results

# 📌 4. 전체 크롤링 실행
def crawl_dcinside(max_pages=2, max_depth=2, visited_urls=None, is_incremental=True):
    """디시인사이드 전체 크롤링 실행"""
    # 증분 크롤링을 위한 방문 URL 관리
    if visited_urls is None:
        visited_urls = set()
    
    # 결과 및 세션 초기화
    session = requests.Session()
    session.headers.update(HEADERS)
    session.headers.update({"Referer": "https://gall.dcinside.com/mgallery/board/lists/?id=dfip"})
    
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

                # 공지글 확인
                subject_tag = post.select_one("td.gall_subject")
                is_notice = subject_tag and "공지" in subject_tag.get_text()

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

        # 결과 요약
        elapsed_time = time.time() - start_time
        avg_time_per_post = elapsed_time / len(results) if results else 0

        # 결과 저장 (증분 처리 지원)
        from utils import save_dc_data
        save_dc_data(results, append=is_incremental)
        
    except Exception as e:
        pass
    
    return results

# 스크립트 직접 실행 시
if __name__ == "__main__":
    # 테스트 실행
    crawl_dcinside(max_pages=1, max_depth=0)
