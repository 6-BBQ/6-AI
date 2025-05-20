import time
import json
import requests
import re
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://df.nexon.com"
SAVE_PATH = "data/raw/official_row.json"
FILTER_KEYWORDS = ["명성", "던전", "스펙업", "장비", "파밍", "뉴비", "유입", "융합석", "중천",
                   "팁", "가이드", "공략", "세트", "에픽", "태초", "소울", "레기온", "레이드", "초보자", "현질", "준종결", "종결"]
# ──────────────────────────────────────────────

# 📌 1. 게시글 리스트 추출 (한 페이지)
def get_post_list(page_num, session):
    url = f"{BASE_URL}/community/dnfboard/list?category=99&page={page_num}"
    resp = session.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    posts = soup.select("article.board_list > ul")
    return posts

# 📌 2. 게시글 URL 추출
def parse_post_url(post):
    title_li = post.select_one("li.title")
    if not title_li:
        return None

    link_tag = title_li.find_all("a")[-1]
    href = link_tag.get("href", "").strip()
    if href.startswith("/community/dnfboard/article/"):
        return BASE_URL + href
    return None

# 📌 3. 게시글 본문 크롤링 (본문 내 URL도 재귀 크롤링)
def crawl_post_content(post_url, session, visited_urls, depth=0, max_depth=2):
    if post_url in visited_urls:
        return []

    visited_urls.add(post_url)
    results = []

    try:
        resp = session.get(post_url)
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.select_one("p.commu1st span")
        title_text = title_tag.get_text(strip=True) if title_tag else "[제목 없음]"
        title_text = re.sub(r"\s+", " ", title_text)

        date_tag = soup.select_one("li.date")
        date_text = date_tag.get_text(strip=True) if date_tag else "[날짜 없음]"

        content_div = soup.select_one("div.bd_viewcont")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

        # ✅ 제목 키워드 필터링 (본문 무시)
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
                if linked_href.startswith("/community/dnfboard/article/"):
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, session, visited_urls, depth + 1, max_depth))

        time.sleep(0.01)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

    return results

# 📌 4. 전체 크롤링 실행
def crawl_df(max_pages=2, max_depth=2):
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

            is_notice = 'notice' in post.get("class", [])

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