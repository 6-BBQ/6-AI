from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing import Callable, Tuple

from config import config
from dc_crawler import crawl_dcinside
from official_crawler import crawl_df
from arca_crawler import crawl_arca
from utils import get_logger

# 증분 크롤링 기록 파일
VISITED_URLS_PATH = config.VISITED_URLS_PATH

# ────────────────────────────────────────────────────────────
# 방문 URL 로드 / 저장
# ────────────────────────────────────────────────────────────

def load_visited_urls() -> set[str]:
    if os.path.exists(VISITED_URLS_PATH):
        try:
            with open(VISITED_URLS_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_visited_urls(urls: set[str]) -> None:
    os.makedirs(Path(VISITED_URLS_PATH).parent, exist_ok=True)
    try:
        with open(VISITED_URLS_PATH, "w", encoding="utf-8") as f:
            json.dump(list(urls), f, ensure_ascii=False)
    except Exception:
        pass

# ────────────────────────────────────────────────────────────
# 크롤러 실행 헬퍼
# ────────────────────────────────────────────────────────────

def run_crawler(func: Callable, *args, **kwargs) -> list[dict]:
    start = time.time()
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        print(f"⚠️  {func.__name__} 오류: {e}")
        return []
    finally:
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__} 종료 — {elapsed:.1f}s")

# ────────────────────────────────────────────────────────────
# 메인 진입점
# ────────────────────────────────────────────────────────────

def main() -> None:
    logger = get_logger("crawler")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            ▶ 던파 스펙업 가이드용 통합 크롤러 (YouTube 제외)

            기본 = 증분 모드. 전체 초기화를 원하면 --full 사용.
            """
        ),
    )

    # 공통 옵션 (config 기본값 활용)
    parser.add_argument("--pages", type=int, default=config.DEFAULT_CRAWL_PAGES)
    parser.add_argument("--depth", type=int, default=config.DEFAULT_CRAWL_DEPTH)

    # 실행 방식
    parser.add_argument("--parallel", action="store_true", help="Thread 병렬 실행")
    parser.add_argument("--workers", type=int, default=4)

    # 증분 / 전체 모드
    parser.add_argument("--incremental", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--clear-history", action="store_true", help="visited_urls.json 초기화")

    # 소스 선택
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="크롤링 소스: official,dc,arca,all (콤마구분)",
    )

    # 품질 필터 및 병합
    parser.add_argument("--quality-threshold", type=int, default=0)
    parser.add_argument("--merge", action="store_true", help="결과를 data/merged/* 에 저장")

    args = parser.parse_args()

    # 전체 모드 → 증분 플래그 해제
    if args.full:
        args.incremental = False

    # 방문 기록 초기화
    if args.clear_history and os.path.exists(VISITED_URLS_PATH):
        os.remove(VISITED_URLS_PATH)
        print("🗑️  방문 기록 초기화 완료")
        if not args.full:
            return  # 초기화만 하고 종료

    logger.info("🔔 크롤링 시작 (%s)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("   pages=%s depth=%s parallel=%s workers=%s", args.pages, args.depth, args.parallel, args.workers)

    visited = load_visited_urls() if args.incremental else set()

    # 작업 목록 작성
    tasks: list[Tuple[str, Callable[[], list[dict]]]] = []
    sel = args.sources.lower().split(",")
    all_sel = "all" in sel
    if all_sel or "official" in sel:
        tasks.append(("공홈", lambda: run_crawler(crawl_df, args.pages, args.depth, visited, args.incremental)))
    if all_sel or "dc" in sel:
        tasks.append(("디시", lambda: run_crawler(crawl_dcinside, args.pages, args.depth, visited, args.incremental)))
    if all_sel or "arca" in sel:
        tasks.append(("아카", lambda: run_crawler(crawl_arca, args.pages, args.depth, visited, args.incremental)))

    results: dict[str, int] = {}
    all_items: list[dict] = []
    t0 = time.time()

    def collect(name: str, func: Callable[[], list[dict]]):
        items = func()
        results[name] = len(items)
        all_items.extend(items)

    if args.parallel and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            fut_map = {ex.submit(func): name for name, func in tasks}
            for fut in as_completed(fut_map):
                collect(fut_map[fut], lambda f=fut: f.result())
    else:
        for name, func in tasks:
            collect(name, func)

    # 품질 필터
    if args.quality_threshold > 0:
        before = len(all_items)
        all_items = [x for x in all_items if x.get("quality_score", 0) >= args.quality_threshold]
        logger.info("품질 필터링: %s → %s", before, len(all_items))

    # 결과 병합 저장
    if args.merge and all_items:
        merged_dir = Path(config.MERGED_DIR)
        merged_dir.mkdir(parents=True, exist_ok=True)
        file_path = merged_dir / f"crawl_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        logger.info("💾 병합 결과 저장: %s (%s items)", file_path, len(all_items))

    # 방문 기록 저장 (증분)
    if args.incremental:
        save_visited_urls(visited)

    # 요약 출력
    elapsed = time.time() - t0
    logger.info("🎉 크롤링 완료 — %.1fs", elapsed)
    for src, cnt in results.items():
        logger.info("   %s: %s", src, cnt)
    logger.info("   총 수집: %s", sum(results.values()))


if __name__ == "__main__":
    # 모듈 경로 보정: "python crawler.py" 실행 시
    if __package__ is None and __name__ == "__main__":
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
