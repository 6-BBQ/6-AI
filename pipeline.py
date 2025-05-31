from __future__ import annotations

import subprocess
import sys
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

from config import config  # 중앙 설정 싱글턴
from utils import get_logger

# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────

def run_script(path: str, args: list[str] | None = None) -> None:
    """하위 파이썬 스크립트를 동기로 실행하고 상태를 로깅"""
    args = args or []
    logger.info(f"🟡 실행 중: {path} {' '.join(args)}")

    try:
        subprocess.run([sys.executable, path, *args], check=True)
        logger.info(f"✅ 완료: {path}")
    except subprocess.CalledProcessError as exc:
        logger.error(f"❌ 오류 발생 ({path}): {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """엔드‑투‑엔드 파이프라인 진입점"""

    # 로거 초기화 (config.LOG_LEVEL / LOG_DIR 반영)
    global logger
    logger = get_logger(
        "pipeline",
        level=config.LOG_LEVEL,
        log_dir=config.LOG_DIR,
    )

    # CORS·로그 디렉터리 등 필수 폴더 생성
    config.create_directories()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="던파 스펙업 AI 파이프라인 (증분 처리 지원)",
        epilog=textwrap.dedent(
            """
            예시:
              python pipeline.py                # 기본 (증분)
              python pipeline.py --full         # 전체 재처리
              python pipeline.py --skip-crawl   # 크롤링 건너뛰기
            """
        ),
    )

    # 공통 플래그
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="증분 처리 모드 (기본값)",
    )
    parser.add_argument("--full", action="store_true", help="전체 처리 모드")

    # 크롤러 매개변수 (central config 기본값 활용) 🕷️
    parser.add_argument(
        "--pages",
        type=int,
        default=config.DEFAULT_CRAWL_PAGES,
        help=f"크롤링할 페이지 수 (기본: {config.DEFAULT_CRAWL_PAGES})",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=config.DEFAULT_CRAWL_DEPTH,
        help=f"크롤링 재귀 깊이 (기본: {config.DEFAULT_CRAWL_DEPTH})",
    )
    parser.add_argument(
        "--yt-mode",
        type=str,
        default="hybrid",
        choices=["channel", "search", "hybrid"],
        help="YouTube 크롤링 모드 (기본: hybrid)",
    )
    parser.add_argument(
        "--yt-max",
        type=int,
        default=20,
        help="YouTube 최대 영상 수 (기본: 20)",
    )

    # 단계 건너뛰기 옵션
    parser.add_argument("--skip-crawl", action="store_true", help="크롤링 단계 건너뛰기")
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="전처리 단계 건너뛰기"
    )
    parser.add_argument("--skip-vectordb", action="store_true", help="벡터 DB 구축 단계 건너뛰기")
    parser.add_argument("--force", action="store_true", help="기존 산출물 강제 덮어쓰기")

    args = parser.parse_args()

    # 전체 모드이면 증분 플래그 해제
    if args.full:
        args.incremental = False

    mode_name = "증분" if args.incremental else "전체"
    mode_emoji = "🔄" if args.incremental else "🚀"

    logger.info("\n" + "=" * 50)
    logger.info(f"{mode_emoji} 던파 스펙업 파이프라인 {mode_name} 실행 시작")
    logger.info(f"   📅 시작 시간: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"   🕸️  크롤 페이지: {args.pages}")
    logger.info(f"   ↪️  깊이: {args.depth}")
    logger.info(f"   🎥  YouTube 모드: {args.yt_mode} / {args.yt_max}개")

    t0 = datetime.now()

    # 1️⃣ 크롤링 단계 ----------------------------------------------------------
    if not args.skip_crawl:
        logger.info("\n" + "=" * 50)
        logger.info("1️⃣ 크롤링 단계")
        logger.info("=" * 50)

        crawl_args = [
            "--pages",
            str(args.pages),
            "--depth",
            str(args.depth),
            "--yt-mode",
            args.yt_mode,
            "--yt-max",
            str(args.yt_max),
            "--merge",
        ]
        crawl_args.append("--incremental" if args.incremental else "--full")
        run_script("crawlers/crawler.py", crawl_args)
    else:
        logger.info("\n⏭️  크롤링 단계 건너뛰기")

    # 2️⃣ 전처리 단계 ----------------------------------------------------------
    if not args.skip_preprocess:
        logger.info("\n" + "=" * 50)
        logger.info("2️⃣ 전처리 단계")
        logger.info("=" * 50)

        preprocess_args: list[str] = []
        if args.incremental:
            preprocess_args.append("--incremental")
        if args.force:
            preprocess_args.append("--force")
        run_script("preprocessing/preprocess.py", preprocess_args)
    else:
        logger.info("\n⏭️  전처리 단계 건너뛰기")

    # 3️⃣ 벡터 DB 구축 단계 ----------------------------------------------------
    if not args.skip_vectordb:
        logger.info("\n" + "=" * 50)
        logger.info("3️⃣ 벡터 DB 구축 단계")
        logger.info("=" * 50)

        vectordb_args: list[str] = []
        if args.incremental:
            vectordb_args.append("--incremental")
        if args.force:
            vectordb_args.append("--force")
        run_script("vectorstore/build_vector_db.py", vectordb_args)
    else:
        logger.info("\n⏭️  벡터 DB 구축 단계 건너뛰기")

    # 완료 ---------------------------------------------------------------
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info("\n" + "=" * 50)
    logger.info(f"🎉 전체 파이프라인 완료! ({mode_name} 모드)")
    logger.info(f"   ⏱️  총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
