# crawler.py (간결한 버전)
from __future__ import annotations
import argparse, sys, textwrap, os, json, time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from official_crawler import crawl_df
from dc_crawler import crawl_dcinside
from arca_crawler import crawl_arca
from youtube_crawler import crawl_youtube

# 방문한 URL 저장소 (증분 크롤링 지원)
VISITED_URLS_FILE = "data/visited_urls.json"

def load_visited_urls():
    """이전에 방문한 URL 목록 로드"""
    try:
        if os.path.exists(VISITED_URLS_FILE):
            with open(VISITED_URLS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        return set()

def save_visited_urls(urls):
    """방문한 URL 목록 저장"""
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(VISITED_URLS_FILE), exist_ok=True)
        
        with open(VISITED_URLS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(urls), f, ensure_ascii=False)
    except Exception as e:
        pass

def run_crawler(crawler_func, *args, **kwargs):
    """크롤러 실행 함수 (에러 처리 포함)"""
    start_time = time.time()
    func_name = crawler_func.__name__
    
    try:
        result = crawler_func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result
    except Exception as e:
        return []

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ▶ 던파 스펙업 가이드용 통합 크롤링 스크립트
        """)
    )
    parser.add_argument("--pages", type=int, default=10, help="각 게시판 최대 페이지 수")
    parser.add_argument("--depth", type=int, default=2, help="본문 링크 재귀 depth")
    parser.add_argument("--yt-mode", type=str, default="hybrid", 
                        choices=["channel", "search", "hybrid"],
                        help="YouTube 크롤링 모드 (channel: 채널만, search: 검색만, hybrid: 둘 다)")
    parser.add_argument("--yt-channel", type=str,
                        default="https://www.youtube.com/@zangzidnf",
                        help="YouTube 채널 URL(@핸들 또는 /channel/ID)")
    parser.add_argument("--yt-query", type=str, default="던파 가이드",
                        help="YouTube 검색 쿼리")
    parser.add_argument("--yt-max", type=int, default=20,
                        help="채널에서 가져올 최신 영상 개수")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 활성화")
    parser.add_argument("--workers", type=int, default=4, help="병렬 작업자 수")
    parser.add_argument("--incremental", action="store_true", help="증분 크롤링 활성화")
    parser.add_argument("--clear-history", action="store_true", help="방문 기록 초기화")
    parser.add_argument("--sources", type=str, default="all", 
                        help="크롤링할 소스 (콤마로 구분: official,dc,arca,youtube,all)")
    parser.add_argument("--quality-threshold", type=int, default=0,
                        help="최소 품질 점수 (이 점수 미만의 항목은 최종 결과에서 제외)")
    parser.add_argument("--merge", action="store_true", help="모든 결과를 하나의 파일로 병합")

    args = parser.parse_args()
    
    # 방문 기록 초기화 옵션
    if args.clear_history:
        try:
            if os.path.exists(VISITED_URLS_FILE):
                os.remove(VISITED_URLS_FILE)
        except Exception as e:
            pass
        return

    print(f"\n🔔 통합 크롤링 시작 ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"   - pages = {args.pages}, depth = {args.depth}")
    print(f"   - YouTube 모드 = {args.yt_mode}")
    if args.yt_mode in ['hybrid', 'channel']:
        print(f"   - yt-channel = {args.yt_channel}")
    if args.yt_mode in ['hybrid', 'search']:
        print(f"   - yt-query = '{args.yt_query}'")
    print(f"   - yt-max = {args.yt_max}")
    print(f"   - 병렬 처리 = {args.parallel}, 작업자 수 = {args.workers}")
    print(f"   - 증분 크롤링 = {args.incremental}")
    
    # 증분 크롤링을 위한 방문 URL 로드
    visited_urls = load_visited_urls() if args.incremental else set()
    
    # 크롤링할 소스 결정
    sources = args.sources.lower().split(',')
    crawl_all = "all" in sources
    
    # 크롤링 작업 정의
    crawl_tasks = []
    
    if crawl_all or "official" in sources:
        crawl_tasks.append(("공홈", lambda: run_crawler(crawl_df, args.pages, args.depth, visited_urls)))
    
    if crawl_all or "dc" in sources:
        crawl_tasks.append(("디시", lambda: run_crawler(crawl_dcinside, args.pages, args.depth, visited_urls)))
    
    if crawl_all or "arca" in sources:
        crawl_tasks.append(("아카", lambda: run_crawler(crawl_arca, args.pages, args.depth, visited_urls)))
    
    if crawl_all or "youtube" in sources:
        def youtube_crawl_task():
            youtube_results = []
            if args.yt_mode in ["hybrid", "search"]:
                # 검색 기반 크롤링
                from youtube_crawler import crawl_youtube_search
                search_results = run_crawler(crawl_youtube_search, args.yt_query, args.yt_max // 2 if args.yt_mode == "hybrid" else args.yt_max, visited_urls)
                youtube_results.extend(search_results)
                
            if args.yt_mode in ["hybrid", "channel"]:
                # 채널 기반 크롤링
                from youtube_crawler import crawl_youtube_channel
                channel_results = run_crawler(crawl_youtube_channel, args.yt_channel, args.yt_max // 2 if args.yt_mode == "hybrid" else args.yt_max, visited_urls)
                youtube_results.extend(channel_results)
            
            # YouTube 결과를 개별 파일에 저장 (중복 제거 된 상태)
            from youtube_crawler import save_results_append
            if youtube_results:
                # 기존 파일 초기화 후 저장
                youtube_raw_path = "data/raw/youtube_raw.json"
                import os
                if os.path.exists(youtube_raw_path):
                    os.remove(youtube_raw_path)
                
                # 결과 저장
                import json
                from pathlib import Path
                save_dir = Path(youtube_raw_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(youtube_raw_path, "w", encoding="utf-8") as f:
                    json.dump(youtube_results, f, ensure_ascii=False, indent=2)
                print(f"💾 YouTube 통합 결과 저장: {len(youtube_results)}개")
                
            return youtube_results
            
        crawl_tasks.append(("유튜브", youtube_crawl_task))
    
    # 결과 및 통계
    results = {}
    all_results = []
    
    # 시작 시간
    start_time = time.time()
    
    # 병렬 또는 순차 실행
    if args.parallel:
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=min(args.workers, len(crawl_tasks))) as executor:
            # 작업 제출
            future_to_task = {executor.submit(task_func): task_name for task_name, task_func in crawl_tasks}
            
            # 결과 수집
            for i, future in enumerate(as_completed(future_to_task)):
                task_name = future_to_task[future]
                try:
                    task_results = future.result()
                    count = len(task_results)
                    results[task_name] = count
                    all_results.extend(task_results)
                except Exception as e:
                    results[task_name] = 0
    else:
        # 순차 실행
        for i, (task_name, task_func) in enumerate(crawl_tasks):
            try:
                task_results = task_func()
                count = len(task_results)
                results[task_name] = count
                all_results.extend(task_results)
            except Exception as e:
                results[task_name] = 0
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    
    # 품질 임계값 필터링 (content_score -> quality_score 변경)
    if args.quality_threshold > 0:
        original_count = len(all_results)
        all_results = [item for item in all_results if item.get("quality_score", 0) >= args.quality_threshold]
        filtered_count = original_count - len(all_results)
    
    # 결과 병합 저장
    if args.merge and all_results:
        merged_file = f"data/merged/crawl_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        os.makedirs(os.path.dirname(merged_file), exist_ok=True)
        
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 증분 크롤링인 경우 방문 URL 저장
    if args.incremental:
        save_visited_urls(visited_urls)
    
    # 결과 요약
    print("\n모든 크롤링 완료!")
    print(f"   총 소요 시간: {elapsed_time:.1f}초")
    
    for source, count in results.items():
        print(f"   - {source}: {count}개 항목")
    
    total_count = sum(results.values())
    print(f"\n   총 {total_count}개 항목 수집 완료!")
    
    if args.quality_threshold > 0:
        print(f"   품질 필터링 후 남은 항목: {len(all_results)}개")
    
    if args.merge:
        print(f"   병합 결과 저장: {merged_file}")

if __name__ == "__main__":
    # 패키지 밖에서 python crawler/crawler.py로도 실행 가능하도록 경로 보정
    if __package__ is None and __name__ == "__main__":
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
