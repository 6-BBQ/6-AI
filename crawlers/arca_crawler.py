import json
import time
import datetime
import cloudscraper
from bs4 import BeautifulSoup, NavigableString

# ──────────────────────────────────────────────
HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://arca.live"
SAVE_PATH = "data/raw/arca_row.json"
FILTER_KEYWORDS = ["명성", "던전", "스펙업", "장비", "파밍", "뉴비", "유입", "초보자", "융합석", "중천", "세트", "나벨", "베누스",
                   "가이드", "공략", "에픽", "태초", "소울", "레기온", "레이드", "현질", "세리아", "준종결", "종결"]
EXCLUDE_KEYWORDS = ["이벤트", "선계", "커스텀", "카지노", "기록실", "서고", "바칼", "ㅅㅂ", "ㅂㅅ", "ㅄ", "ㅗ", "시발", "씨발", "병신", "좆"]
# ──────────────────────────────────────────────

# 날짜 확인 함수 추가
def is_valid_date(date_text):
    # "[날짜 없음]"인 경우 유효하지 않음
    if date_text == "[날짜 없음]":
        return False
    
    # 2025년 확인 (포맷: "2025-05-12")
    return date_text.startswith("2025")

# 📌 Cloudflare 우회용 세션 생성
def get_new_scraper():
    return cloudscraper.create_scraper()

# 📌 1. 게시글 리스트 추출 (한 페이지)
def get_post_list(page_num):
    url = f"{BASE_URL}/b/dunfa?mode=best&category=공략&p={page_num}"
    scraper = get_new_scraper()
    resp = scraper.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.select("a.vrow")

# 📌 2. 게시글 URL 및 제목 추출
def parse_post_info(post):
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
    if post_url in visited_urls:
        return []

    visited_urls.add(post_url)
    results = []
    
    try:
        scraper = get_new_scraper()
        resp = scraper.get(post_url)
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.select_one("div.title-row .title")
        if title_tag:
            title_text = ''.join(
                t for t in title_tag.contents if isinstance(t, NavigableString)
            ).strip()
        else:
            title_text = "[제목 없음]"
        
        date_tag = soup.select_one("div.article-info-section .date time")
        if date_tag:
            raw = date_tag.get("datetime")
            date_text = datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d")
        else:
            date_text = "[날짜 없음]"

        # 2025년 게시글만 허용
        if not is_valid_date(date_text):
            return []

        content_div = soup.select_one("div.fr-view.article-content")
        content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

        # 제목은 이미 리스트 단계에서 필터링했으므로 여기서는 필터링하지 않음
        
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
                if linked_href.startswith("/b/dunfa"):
                    # 링크 텍스트(제목) 추출
                    link_text = a.get_text(strip=True)
                    
                    # 링크 제목 필터링 - 제외 키워드가 포함된 링크는 건너뛰기
                    if any(bad_word in link_text for bad_word in EXCLUDE_KEYWORDS):
                        continue
                    
                    # 포함 키워드가 하나라도 있는지 확인 - 없으면 건너뛰기
                    if not any(keyword in link_text for keyword in FILTER_KEYWORDS):
                        continue
                    
                    full_link = BASE_URL + linked_href
                    results.extend(crawl_post_content(full_link, visited_urls, depth + 1, max_depth))

        time.sleep(0.01)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

    return results

# 📌 4. 전체 크롤링 실행
def crawl_arca(max_pages=2, max_depth=2):
    visited_urls = set()
    results = []
    notice_processed = False

    for page in range(1, max_pages + 1):
        print(f"\n📄 페이지 {page} 크롤링 중...")
        posts = get_post_list(page)

        for post in posts:
            post_url, title_text = parse_post_info(post)
            if not post_url or not title_text:
                continue
                
            # 게시글 리스트에서 제목 필터링
            if not any(keyword in title_text for keyword in FILTER_KEYWORDS):
                continue
                
            if any(bad_word in title_text for bad_word in EXCLUDE_KEYWORDS):
                continue

            is_notice = 'notice' in post.get("class", [])

            if is_notice:
                if not notice_processed:
                    print(f"📌 공지글 수집: {post_url}")
                    results.extend(crawl_post_content(post_url, visited_urls, depth=0, max_depth=max_depth))
                continue
            else:
                results.extend(crawl_post_content(post_url, visited_urls, depth=0, max_depth=max_depth))

        if not notice_processed:
            notice_processed = True

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(results)}개의 게시글 저장 완료: {SAVE_PATH}")