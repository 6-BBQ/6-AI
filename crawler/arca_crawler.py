import time
import cloudscraper
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_new_scraper():
    return cloudscraper.create_scraper()

def crawl_arca():
    list_url = "https://arca.live/b/dunfa?mode=best&category=%EA%B3%B5%EB%9E%B5"
    scraper = get_new_scraper()
    resp = scraper.get(list_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = soup.select("a.vrow")
    print(f"🔍 총 {len(posts)}개의 게시글을 찾았습니다.\n")

    for post in posts:
        if 'notice' in post.get('class', []):
            continue

        scraper = get_new_scraper()
        href = post.get("href", "").split("?")[0]
        if not href.startswith("/b/"):
            continue

        post_url = "https://arca.live" + href
        print(f"\n🔗 게시글 URL: {post_url}")
        
        try:
            post_resp = scraper.get(post_url)
            post_soup = BeautifulSoup(post_resp.text, "html.parser")

            title_tag = post_soup.select_one("div.title-row .title")
            title = title_tag.get_text(strip=True) if title_tag else "[제목 없음]"

            date_tag = post_soup.select_one("div.article-info-section .date time")
            date = date_tag.get("datetime", "[날짜 없음]") if date_tag else "[날짜 없음]"

            content_div = post_soup.select_one("div.fr-view.article-content")
            content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

            print("📌 제목:", title)
            print("🕒 작성일:", date)
            print("📝 본문 일부:", content_text[:100], "...")

            time.sleep(1.5)

        except Exception as e:
            print("❌ 에러 발생:", e)

if __name__ == "__main__":
    crawl_arca()