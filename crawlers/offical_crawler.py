import time
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
BASE_URL = "https://df.nexon.com"

def crawl_df():
    session = requests.Session()
    session.headers.update(HEADERS)

    list_url = f"{BASE_URL}/community/dnfboard/list?category=99"
    resp = session.get(list_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    posts = soup.select("article.board_list > ul")
    print(f"🔍 총 {len(posts)}개의 게시글을 찾았습니다.\n")

    for post in posts:
        # 공지글 제외
        if 'notice' in post.get("class", []):
            continue

        # 제목 및 링크
        title_li = post.select_one("li.title")
        if not title_li:
            continue

        # 게시글 링크
        link_tag = title_li.find_all("a")[-1]  # 마지막 a태그가 실제 링크임
        href = link_tag.get("href", "").strip()
        if not href.startswith("/community/dnfboard/article/"):
            continue
        post_url = BASE_URL + href
        print(f"\n🔗 게시글 URL: {post_url}")

        try:
            # 상세 페이지 요청
            resp_post = session.get(post_url)
            post_soup = BeautifulSoup(resp_post.text, "html.parser")

            # 제목
            title_tag = post_soup.select_one("p.commu1st span")
            title_text = title_tag.get_text(separator="", strip=True) if title_tag else "[제목 없음]"
            title_text = " ".join(title_text.split())

            # 날짜
            date_tag = post_soup.select_one("li.date")
            date_text = date_tag.get_text(strip=True) if date_tag else "[날짜 없음]"

            # 본문
            content_div = post_soup.select_one("div.bd_viewcont")
            content_text = content_div.get_text("\n", strip=True) if content_div else "[본문 없음]"

            print("📌 제목:", title_text)
            print("🕒 작성일:", date_text)
            print("📝 본문 일부:", content_text[:100], "...")

            time.sleep(1.5)

        except Exception as e:
            print("❌ 에러 발생:", e)

if __name__ == "__main__":
    crawl_df()