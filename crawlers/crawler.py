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
from utils import get_logger

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
        print(f"⚠️ {func_name} 크롤링 오류: {e}")
        return []

def main():
    # 로거 초기화
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ▶ 던파 스펙업 가이드용 통합 크롤링 스크립트
            
            기본적으로 증분 크롤링을 수행합니다.
            전체 크롤링이 필요한 경우 --full 옵션을 사용하세요.
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
                        help="YouTube 기본 검색 쿼리 (사용되지 않음, 다중 쿼리 사용)")
    parser.add_argument("--yt-max", type=int, default=20,
                        help="채널에서 가져올 최신 영상 개수")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 활성화")
    parser.add_argument("--workers", type=int, default=4, help="병렬 작업자 수")
    parser.add_argument("--incremental", action="store_true", default=True, help="증분 크롤링 (기본값)")
    parser.add_argument("--full", action="store_true", help="전체 크롤링 (증분 무시)")
    parser.add_argument("--clear-history", action="store_true", help="방문 기록 초기화")
    parser.add_argument("--sources", type=str, default="all", 
                        help="크롤링할 소스 (콤마로 구분: official,dc,arca,youtube,all)")
    parser.add_argument("--quality-threshold", type=int, default=0,
                        help="최소 품질 점수 (이 점수 미만의 항목은 최종 결과에서 제외)")
    parser.add_argument("--merge", action="store_true", help="모든 결과를 하나의 파일로 병합")

    args = parser.parse_args()
    
    # 전체 모드 검사
    if args.full:
        args.incremental = False
    
    # 방문 기록 초기화 옵션
    if args.clear_history:
        try:
            if os.path.exists(VISITED_URLS_FILE):
                os.remove(VISITED_URLS_FILE)
        except Exception as e:
            pass
        return

    logger.info(f"\n🔔 통합 크롤링 시작 ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info(f"   - pages = {args.pages}, depth = {args.depth}")
    logger.info(f"   - YouTube 모드 = {args.yt_mode}")
    if args.yt_mode in ['hybrid', 'channel']:
        logger.info(f"   - yt-channel = {args.yt_channel}")
    if args.yt_mode in ['hybrid', 'search']:
        logger.info(f"   - yt-query = '던파 가이드(10), 현질가이드(5), 나벨공략(5)'")
    logger.info(f"   - yt-max = {args.yt_max}")
    logger.info(f"   - 병렬 처리 = {args.parallel}, 작업자 수 = {args.workers}")
    logger.info(f"   - 증분 크롤링 = {args.incremental}")
    
    # 증분 크롤링을 위한 방문 URL 로드
    if args.incremental:
        visited_urls = load_visited_urls()
    else:
        # 전체 모드일 때는 빈 집합으로 시작 (모든 URL 재처리)
        visited_urls = set()
    
    # 크롤링할 소스 결정
    sources = args.sources.lower().split(',')
    crawl_all = "all" in sources
    
    # 크롤링 작업 정의
    crawl_tasks = []
    
    # 증분 모드 설정
    is_incremental = args.incremental
    
    if crawl_all or "official" in sources:
        crawl_tasks.append(("공홈", lambda: run_crawler(crawl_df, args.pages, args.depth, visited_urls, is_incremental)))
    
    if crawl_all or "dc" in sources:
        crawl_tasks.append(("디시", lambda: run_crawler(crawl_dcinside, args.pages, args.depth, visited_urls, is_incremental)))
    
    if crawl_all or "arca" in sources:
        crawl_tasks.append(("아카", lambda: run_crawler(crawl_arca, args.pages, args.depth, visited_urls, is_incremental)))
    
    if crawl_all or "youtube" in sources:
        def youtube_crawl_task():
            youtube_results = []
            if args.yt_mode in ["hybrid", "search"]:
                # 여러 검색 기반 크롤링 (카테고리별)
                from youtube_crawler import crawl_youtube_multi_query
                
                # 던파 카테고리별 검색 쿼리 설정
                search_queries = [
                    ("던파 가이드", 10),       # 던파 가이드 10개
                    ("던파 현질 가이드", 5),    # 던파 현질 가이드 5개
                    ("던파 나벨 공략", 5),      # 던파 나벨 공략 5개
                ]
                
                search_results = run_crawler(crawl_youtube_multi_query, search_queries, visited_urls)
                youtube_results.extend(search_results)
                
            if args.yt_mode in ["hybrid", "channel"]:
                # 채널 기반 크롤링, 임시로 꺼둠
                from youtube_crawler import crawl_youtube_channel
                # 하이브리드 모드일 때는 채널 크롤링 개수를 줄여서 검색 결과와 균형 맞춤
                channel_results = run_crawler(crawl_youtube_channel, args.yt_channel, 3 if args.yt_mode == "hybrid" else args.yt_max, visited_urls)
                youtube_results.extend(channel_results)
            
            # YouTube 결과를 개별 파일에 저장 (증분 모드 지원)
            if youtube_results:
                youtube_raw_path = "data/raw/youtube_raw.json"
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(youtube_raw_path), exist_ok=True)
                
                try:
                    if args.incremental and os.path.exists(youtube_raw_path):
                        # 증분 모드: 기존 데이터 로드 후 병합
                        with open(youtube_raw_path, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                        
                        # URL 중복 제거를 위한 기존 URL 집합
                        existing_urls = {item.get('url') for item in existing_data if isinstance(item, dict) and 'url' in item}
                        
                        # 새로운 데이터 중 중복되지 않는 것만 추가
                        new_data = [item for item in youtube_results if item.get('url') not in existing_urls]
                        
                        if new_data:
                            final_data = existing_data + new_data
                            print(f"💾 YouTube 증분 저장: 기존 {len(existing_data)}개 + 새로운 {len(new_data)}개")
                        else:
                            final_data = existing_data
                            print(f"💾 YouTube: 새로운 데이터 없음 (모두 중복)")
                    else:
                        # 전체 모드 또는 파일이 없는 경우: 전체 저장
                        final_data = youtube_results
                        print(f"💾 YouTube 전체 저장: {len(youtube_results)}개")
                    
                    # 파일 저장
                    with open(youtube_raw_path, "w", encoding="utf-8") as f:
                        json.dump(final_data, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"⚠️ YouTube 데이터 저장 실패: {e}")
                    # 실패 시 그냥 새 데이터만 저장
                    with open(youtube_raw_path, "w", encoding="utf-8") as f:
                        json.dump(youtube_results, f, ensure_ascii=False, indent=2)
                
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
    
    # 증분 크롤링인 경우만 방문 URL 저장
    if args.incremental:
        save_visited_urls(visited_urls)
    
    # 결과 요약
    logger.info("\n모든 크롤링 완료!")
    logger.info(f"   총 소요 시간: {elapsed_time:.1f}초")
    
    for source, count in results.items():
        logger.info(f"   - {source}: {count}개 항목")
    
    total_count = sum(results.values())
    logger.info(f"\n   총 {total_count}개 항목 수집 완료!")
    
    if args.quality_threshold > 0:
        logger.info(f"   품질 필터링 후 남은 항목: {len(all_results)}개")
    
    if args.merge:
        logger.info(f"   병합 결과 저장: {merged_file}")

if __name__ == "__main__":
    # 패키지 밖에서 python crawler/crawler.py로도 실행 가능하도록 경로 보정
    if __package__ is None and __name__ == "__main__":
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
