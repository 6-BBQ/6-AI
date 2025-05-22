# youtube_crawler.py 수정안
from __future__ import annotations
from datetime import datetime, timezone
import json, time, sys, urllib.parse
from pathlib import Path

import yt_dlp                          # pip install yt-dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript
)

from utils import build_item           # priority_score·class_name 자동 부여

# ────────────────────────────────────────────────────────────────
SAVE_PATH = Path("data/raw/youtube_raw.json")

# yt-dlp 기본 옵션
YDL_FLAT_OPTS = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": True,      # 영상 ID·제목만
    "nocheckcertificate": True,
}
YDL_META_OPTS = {
    "quiet": True,
    "skip_download": True,
    "nocheckcertificate": True,
}

# ────────────────────────────────────────────────────────────────
def search_youtube_videos(query, max_results=20):
    """
    YouTube 검색 결과에서 영상 ID 목록 가져오기
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        
    Returns:
        list[str]: 영상 ID 목록
    """
    # 검색 URL 인코딩
    search_url = f"ytsearch{max_results}:{query}"
    
    try:
        with yt_dlp.YoutubeDL(YDL_FLAT_OPTS) as ydl:
            info = ydl.extract_info(search_url, download=False)
            
            # 검색 결과에서 영상 ID 추출
            entries = info.get("entries", [])
            video_ids = []

            for entry in entries:
                # 영상 ID가 있을 경우 추출
                vid = entry.get("id") or ""
                if len(vid) == 11:  # 유튜브 video_id는 항상 11자리
                    video_ids.append(vid)

            return video_ids
    except Exception as e:
        return []

def list_channel_video_ids(channel_url: str, limit: int) -> list[str]:
    """
    yt-dlp 로 채널 → 영상 ID 목록 가져오기 (최대 limit 개)
    """
    try:
        with yt_dlp.YoutubeDL(YDL_FLAT_OPTS) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            
            # 채널이면 entries 안에 다시 video entries가 있는 경우
            entries = info.get("entries", [])
            video_ids = []

            for entry in entries[:limit]:
                # 영상 ID가 있을 경우 추출
                vid = entry.get("id") or ""
                if len(vid) == 11:  # 유튜브 video_id는 항상 11자리
                    video_ids.append(vid)

            return video_ids
    except Exception as e:
        return []

def fetch_video_meta(video_id: str) -> dict:
    """yt-dlp 로 영상 메타데이터 조회"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(YDL_META_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)
        # upload_date: 'YYYYMMDD'
        date_obj = datetime.strptime(info["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc)
        return {
            "title": info["title"],
            "date": date_obj,
            "views": info.get("view_count", 0) or 0,
            "likes": info.get("like_count", 0) or 0,
        }
    except Exception as e:
        return {
            "title": f"YOUTUBE_{video_id}",
            "date": datetime.now(timezone.utc),
            "views": 0,
            "likes": 0,
        }

def fetch_caption_text(video_id: str) -> str:
    """자막 가져오기"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("ko", "en"))
        return " ".join(seg["text"].strip() for seg in transcript if seg["text"].strip())
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as e:
        return ""
    except Exception as e:
        return ""

# ────────────────────────────────────────────────────────────────
def crawl_youtube_search(search_query: str, max_videos: int, visited_urls=None) -> list[dict]:
    """
    YouTube 검색 결과 크롤링
    
    Args:
        search_query: 검색 쿼리
        max_videos: 가져올 최대 영상 수
        visited_urls: 이미 방문한 URL 집합 (증분 크롤링 지원)
    """
    # 증분 크롤링을 위한 방문 URL 관리
    if visited_urls is None:
        visited_urls = set()
    
    # 검색 결과 영상 ID 가져오기
    video_ids = search_youtube_videos(search_query, max_videos)
    
    return process_video_ids(video_ids, visited_urls)

def crawl_youtube_channel(channel_url: str, max_videos: int, visited_urls=None) -> list[dict]:
    """
    YouTube 채널의 영상 크롤링
    
    Args:
        channel_url: YouTube 채널 URL
        max_videos: 가져올 최대 영상 수
        visited_urls: 이미 방문한 URL 집합 (증분 크롤링 지원)
    """
    # 증분 크롤링을 위한 방문 URL 관리
    if visited_urls is None:
        visited_urls = set()
    
    # 영상 ID 목록 가져오기
    video_ids = list_channel_video_ids(channel_url, max_videos)
    
    return process_video_ids(video_ids, visited_urls)

def process_video_ids(video_ids, visited_urls):
    """
    영상 ID 목록을 처리하여 자막이 있는 영상만 저장
    
    Args:
        video_ids: 영상 ID 목록
        visited_urls: 방문한 URL 집합
    """
    results = []
    start_time = time.time()

    for vid in video_ids:
        url = f"https://www.youtube.com/watch?v={vid}"
        
        # 증분 크롤링: 이미 방문한 URL이면 건너뜀
        if url in visited_urls:
            continue
            
        # 방문 기록에 추가
        visited_urls.add(url)

        # 1) 메타데이터 가져오기
        meta = fetch_video_meta(vid)

        # 2) 자막 가져오기
        caption = fetch_caption_text(vid)
        if not caption:
            continue    # 자막 없는 영상은 뉴비 가이드로 쓰기 어렵다 판단

        # 3) utils.build_item → priority_score / class_name 자동 부여
        item = build_item(
            source="youtube",
            url=url,
            title=meta["title"],
            body=caption,
            date=meta["date"],
            views=meta["views"],
            likes=meta["likes"],
        )
        results.append(item)

        # API/서버 과부하 방지
        time.sleep(2.0)

    # 결과 요약
    elapsed_time = time.time() - start_time

    # 📂 결과 저장
    save_dir = Path(SAVE_PATH).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

# 이전 함수와의 호환성을 위한 함수
def crawl_youtube(source: str, max_videos: int, visited_urls=None) -> list[dict]:
    """
    YouTube 크롤링 (채널 URL 또는 검색어)
    
    Args:
        source: 채널 URL 또는 검색어
        max_videos: 가져올 최대 영상 수
        visited_urls: 이미 방문한 URL 집합 (증분 크롤링 지원)
    """
    # source가 URL인지 검색어인지 판단
    if source.startswith(("http://", "https://", "www.")):
        # URL - 채널 크롤링
        return crawl_youtube_channel(source, max_videos, visited_urls)
    else:
        # 검색어 - 검색 결과 크롤링
        return crawl_youtube_search(source, max_videos, visited_urls)

# 직접 실행 시
if __name__ == "__main__":
    # 명령줄 인수 처리
    if len(sys.argv) > 1:
        query_or_channel = sys.argv[1]
    else:
        # 기본 검색어
        query_or_channel = "던파 가이드"
        
    # 가져올 영상 수
    max_videos = 10
    if len(sys.argv) > 2:
        try:
            max_videos = int(sys.argv[2])
        except ValueError:
            pass
    
    print(f"YouTube 크롤링 시작: {query_or_channel} (최대 {max_videos}개)")
    
    # 검색어 또는 채널 URL에 따라 다른 함수 호출
    results = crawl_youtube(query_or_channel, max_videos)
    print(f"크롤링 완료: {len(results)}개 영상")
