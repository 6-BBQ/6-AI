import json
import time
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://gall.dcinside.com"
SAVE_PATH = "data/raw/dc_row.json"
FILTER_KEYWORDS = ["명성", "던전", "스펙업", "장비", "파밍", "뉴비", "유입", "융합석", "중천",
                   "팁", "가이드", "공략", "세트", "에픽", "태초", "소울", "레기온", "레이드", "초보자", "현질", "준종결", "종결"]
# ──────────────────────────────────────────────

# 📌 1. 게시글 리스트 추출 (한 페이지)
def get_post_list(page_num, session):
    url = f"{BASE_URL}/mgallery/board/lists/?id=dfip&sort_type=N&exception_mode=recommend&search_head=10&page={page_num}"
    resp = session.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    posts = soup.select("tr.ub-content.us-post")
    return posts

# 📌 2. 게시글 URL 추출
def parse_post_url(post):
    link_tag = post.select_one("td.gall_tit a[href*='view']")
    if not link_tag:
        return None

    return BASE_URL + link_tag["href"]

# 📌 3. 게시글 본문 크롤링 (본문 내 URL도 재귀 크롤링)
def crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=2):
    if post_url in visited_urls:
        return []

    visited_urls.add(post_url)
    results = []
    
    try:
        resp_post = session.get(post_url)
        soup = BeautifulSoup(resp_post.text, "html.parser")

        title_tag = soup.select_one(".title_subject")
        title_text = title_tag.get_text(strip=True) if title_tag else "[제목 없음]"

        date_tag = soup.select_one("span.gall_date")
        if date_tag:
            raw = date_tag.get_text(strip=True)
            date_text = raw[:10].replace('.', '-')
        else:
            date_text = "[날짜 없음]"

        content_div = soup.select_one("div.write_div")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

        # 📌 제목 키워드 필터링
        if not any(keyword in title_text for keyword in FILTER_KEYWORDS):
            return []

        post_data = {
            "url": post_url,
            "title": title_text,
            "date": date_text,
            "content": content_text
        }

        results.append(post_data)

        # 🔁 본문 내 추가 게시글 링크 (depth 제한 포함)
        if content_div and depth < max_depth:
            for a in content_div.find_all("a", href=True):
                linked_href = a["href"]
                if linked_href.startswith("/mgallery/board/lists/?id=dfip"):
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, session, visited_urls, depth + 1, max_depth))

        time.sleep(0.01)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

    return results

# 📌 4. 전체 크롤링 실행
def crawl_dcinside(max_pages=2, max_depth=2):
    session = requests.Session()
    session.headers.update(HEADERS)

    visited_urls = set()
    results = []
    notice_processed = False

    for page in range(1, max_pages + 1):
        print(f"\n📄 페이지 {page} 크롤링 중...")
        posts = get_post_list(page, session)

        for post in posts:
            post_url = parse_post_url(post)
            if not post_url:
                continue

            subject_tag = post.select_one("td.gall_subject")
            is_notice = subject_tag and "공지" in subject_tag.get_text()

            if is_notice:
                if not notice_processed:
                    print(f"📌 공지글 수집: {post_url}")
                    results.extend(crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=max_depth))
                continue
            else:
                results.extend(crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=max_depth))
        
        if not notice_processed:
            notice_processed = True

    # 📂 결과 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(results)}개의 게시글 저장 완료: {SAVE_PATH}")