# youtube_crawler.py 수정안 - 하이브리드 방식 + append 모드
from __future__ import annotations
from datetime import datetime, timezone
import json, time, sys, urllib.parse, os
from pathlib import Path

import yt_dlp                          # pip install yt-dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript
)

from utils import build_item           # quality_score·class_name 자동 부여

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

            print(f"🔍 검색 결과: '{query}'에서 {len(video_ids)}개 영상 ID 수집")
            return video_ids
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
        return []

def list_channel_video_ids(channel_url: str, limit: int) -> list[str]:
    """
    yt-dlp 로 채널 → 영상 ID 목록 가져오기 (최대 limit 개)
    """
    try:
        print(f"📺 채널 처리 중: {channel_url}")
        
        # 채널 URL 정규화 (여러 형식 지원)
        if "@" in channel_url and "/videos" not in channel_url:
            channel_url = f"{channel_url}/videos"
        
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

            print(f"📺 채널 결과: {len(video_ids)}개 영상 ID 수집")
            return video_ids
    except Exception as e:
        print(f"❌ 채널 크롤링 오류: {e}")
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
    
    return process_video_ids(video_ids, visited_urls, "검색")

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
    
    return process_video_ids(video_ids, visited_urls, "채널")

def process_video_ids(video_ids, visited_urls, source_name="youtube"):
    """
    영상 ID 목록을 처리하여 자막이 있는 영상만 반환 (파일 저장 제거)
    
    Args:
        video_ids: 영상 ID 목록
        visited_urls: 방문한 URL 집합
        source_name: 소스 이름 (로그용)
    
    Returns:
        list[dict]: 처리된 영상 데이터 목록
    """
    results = []
    start_time = time.time()
    print(f"🔄 {source_name} 영상 처리 시작: {len(video_ids)}개 영상")

    for i, vid in enumerate(video_ids, 1):
        url = f"https://www.youtube.com/watch?v={vid}"
        
        # 증분 크롤링: 이미 방문한 URL이면 건너뛰기
        if url in visited_urls:
            print(f"   ⏭️ {i}/{len(video_ids)}: 이미 방문한 URL 건너뛰기")
            continue
            
        # 방문 기록에 추가
        visited_urls.add(url)

        # 1) 메타데이터 가져오기
        print(f"   📺 {i}/{len(video_ids)}: 메타데이터 수집 중...")
        meta = fetch_video_meta(vid)

        # 2) 자막 가져오기
        caption = fetch_caption_text(vid)
        if not caption:
            print(f"   ⚠️ {i}/{len(video_ids)}: 자막 없음, 건너뛰기")
            continue    # 자막 없는 영상은 뉴비 가이드로 쓰기 어렵다 판단

        # 3) utils.build_item → quality_score / class_name 자동 부여
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
        print(f"   ✅ {i}/{len(video_ids)}: '{meta['title'][:50]}...' 수집 완료")

        # API/서버 과부하 방지
        time.sleep(2.0)

    # 결과 요약
    elapsed_time = time.time() - start_time
    print(f"✅ {source_name} 처리 완료: {len(results)}개 수집 ({elapsed_time:.1f}초 소요)")
    
    return results

def save_results_append(results: list[dict], source_name: str):
    """
    결과를 append 모드로 저장 (덮어쓰기 방지)
    """
    if not results:
        return
    
    save_dir = Path(SAVE_PATH).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 기존 데이터가 있으면 로드
    existing_data = []
    if SAVE_PATH.exists():
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"⚠️ 기존 파일 로드 실패: {e}")
            existing_data = []
    
    # 새 데이터 추가
    existing_data.extend(results)
    
    # 전체 데이터 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 {source_name} 결과 저장 완료: {len(results)}개 추가, 총 {len(existing_data)}개")

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

# 직접 실행 시 - 하이브리드 방식 (검색 + 신뢰 채널)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube 던파 크롤러 (하이브리드 방식)")
    parser.add_argument("--incremental", action="store_true", default=True, help="증분 크롤링 (기본값)")
    parser.add_argument("--full", action="store_true", help="전체 크롤링 (증분 무시)")
    parser.add_argument("--max-videos", type=int, default=3, help="소스당 최대 영상 수")
    parser.add_argument("--search-query", type=str, default="던파 가이드", help="검색 쿼리")
    args = parser.parse_args()
    
    print("🎆 YouTube 하이브리드 크롤링 시작!")
    print("1️⃣ 검색 기반: 다양한 채널의 콘텐츠 수집")
    print("2️⃣ 신뢰 채널: 고품질 콘텐츠 보장")
    print(f"📋 설정: 증분={args.incremental}, 최대영상={args.max_videos}, 검색어='{args.search_query}'")
    print()
    
    # 전체 모드 검사
    if args.full:
        args.incremental = False
    
    # 증분 크롤링 지원
    visited_urls = set()
    if args.incremental:
        try:
            # visited_urls.json 로드
            sys.path.append(str(Path(__file__).parent))
            from crawler import load_visited_urls, save_visited_urls
            visited_urls = load_visited_urls()
            print(f"🔄 기존 방문 URL {len(visited_urls)}개 로드됨")
        except ImportError:
            print("⚠️ crawler.py를 찾을 수 없어 새로 시작합니다")
    else:
        # 비증분 모드: 기존 파일 초기화
        if SAVE_PATH.exists():
            print("🗑️ 기존 YouTube 크롤링 결과 초기화")
            os.remove(SAVE_PATH)
    
    all_results = []
    
    # 1️⃣ 검색 기반 크롤링 (다양성 확보)
    print(f"🔍 검색 기반 크롤링: '{args.search_query}' (최대 {args.max_videos}개)")
    try:
        search_results = crawl_youtube_search(args.search_query, args.max_videos, visited_urls)
        all_results.extend(search_results)
        save_results_append(search_results, "검색")
        print(f"   ✅ 검색 결과: {len(search_results)}개 수집")
    except Exception as e:
        print(f"   ⚠️ 검색 오류: {e}")
    
    # 2️⃣ 신뢰할 수 있는 채널들 (품질 보장)
    trusted_channels = [
        "https://www.youtube.com/@zangzidnf",  # 던파 관련 채널
        # "다른 신뢰 채널 URL들을 여기에 추가"
    ]
    
    for i, channel_url in enumerate(trusted_channels, 1):
        print(f"🎥 신뢰 채널 {i}: {channel_url.split('/')[-1]} (최대 {args.max_videos}개)")
        try:
            channel_results = crawl_youtube_channel(channel_url, args.max_videos, visited_urls)
            all_results.extend(channel_results)
            save_results_append(channel_results, f"채널{i}")
            print(f"   ✅ 채널 {i} 결과: {len(channel_results)}개 수집")
        except Exception as e:
            print(f"   ⚠️ 채널 {i} 오류: {e}")
    
    # 증분 크롤링인 경우 visited_urls 저장
    if args.incremental:
        try:
            save_visited_urls(visited_urls)
            print(f"💾 방문 URL {len(visited_urls)}개 저장됨")
        except NameError:
            print("⚠️ visited_urls 저장 실패")
    
    # 최종 결과
    print()
    print(f"🎆 하이브리드 크롤링 완료!")
    print(f"📊 총 {len(all_results)}개 영상 수집 (자막 있는 영상만)")
    print(f"🔄 방문한 URL: {len(visited_urls)}개 (증분: {args.incremental})")
    print(f"💾 결과 저장 위치: {SAVE_PATH}")
    print(f"✅ YouTube 크롤링 완료: {len(all_results)}개 수집")
