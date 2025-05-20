import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://gall.dcinside.com"

def crawl_dcinside():
    session = requests.Session()
    session.headers.update(HEADERS)

    list_url = f"{BASE_URL}/mgallery/board/lists/?id=dfip&sort_type=N&exception_mode=recommend&search_head=10&page=1"
    resp = session.get(list_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = soup.select("tr.ub-content.us-post")
    print(f"🔍 총 {len(posts)}개의 게시글을 찾았습니다.\n")

    for post in posts:
        subject_tag = post.select_one("td.gall_subject")
        if subject_tag and "공지" in subject_tag.text:
            continue

        link_tag = post.select_one("td.gall_tit a[href*='view']")
        if not link_tag:
            continue

        post_url = BASE_URL + link_tag["href"]
        print(f"\n🔗 게시글 URL: {post_url}")

        try:
            resp_post = session.get(post_url)
            post_soup = BeautifulSoup(resp_post.text, "html.parser")

            title = post_soup.select_one(".title_subject")
            title_text = title.get_text(strip=True) if title else "[제목 없음]"

            date = post_soup.select_one("span.gall_date")
            date_text = date.get_text(strip=True) if date else "[날짜 없음]"

            content_div = post_soup.select_one("div.write_div")
            content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

            print("📌 제목:", title_text)
            print("🕒 작성일:", date_text)
            print("📝 본문 일부:", content_text[:100], "...")

            time.sleep(1.5)

        except Exception as e:
            print("❌ 에러 발생:", e)

if __name__ == "__main__":
    crawl_dcinside()