from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List

from config import config
from utils import get_logger

# ────────────────────────────────────────────────────────────
# 스크립트 경로
# ────────────────────────────────────────────────────────────

CRAWLER_SCRIPT: Path = Path(config.CRAWLER_SCRIPT)
PREPROCESS_SCRIPT: Path = Path(config.PREPROCESS_SCRIPT)
BUILD_VECTORDB_SCRIPT: Path = Path(config.BUILD_VECTORDB_SCRIPT)

# ────────────────────────────────────────────────────────────
# 공통 실행 헬퍼
# ────────────────────────────────────────────────────────────

def run_script(script: Path, args: List[str]) -> None:
    """하위 Python 스크립트를 실행하고 실패 시 즉시 종료"""
    cmd = [sys.executable, str(script), *args]
    logger.info("▶ %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("❌ %s 실패 — %s", script.name, exc)
        sys.exit(exc.returncode)

# ────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────

def main() -> None:
    global logger
    logger = get_logger("pipeline")

    # 필수 디렉터리 보장
    config.create_directories()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="던파 스펙업 AI 파이프라인 (YouTube 제외)",
        epilog=textwrap.dedent(
            """
            사용 예시:
              python pipeline.py                # 증분 모드
              python pipeline.py --full         # 전체 재처리
              python pipeline.py --skip-crawl   # 전처리부터 실행
            """,
        ),
    )

    # 처리 모드 플래그
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--incremental", action="store_true", default=True, help="증분 처리(기본)")
    mode.add_argument("--full", action="store_true", help="전체 재처리")

    # 크롤러 기본 옵션 (config 값 반영)
    parser.add_argument("--pages", type=int, default=config.DEFAULT_CRAWL_PAGES)
    parser.add_argument("--depth", type=int, default=config.DEFAULT_CRAWL_DEPTH)
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="크롤링 소스 지정 (official,dc,arca,all)",
    )

    # 단계 스킵
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-vectordb", action="store_true")
    parser.add_argument("--force", action="store_true", help="기존 산출물 강제 덮어쓰기")

    args = parser.parse_args()

    # 전체 모드일 때 increment 플래그 오버라이드
    if args.full:
        args.incremental = False

    logger.info("═" * 60)
    logger.info("🚀 파이프라인 시작 — %s 모드", "증분" if args.incremental else "전체")
    logger.info("📅 %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    t_start = datetime.now()

    # 1️⃣ 크롤링 단계 ---------------------------------------------------------
    if not args.skip_crawl:
        crawl_cli = [
            "--pages", str(args.pages),
            "--depth", str(args.depth),
            "--sources", args.sources,
            "--merge",
            "--incremental" if args.incremental else "--full",
        ]
        run_script(CRAWLER_SCRIPT, crawl_cli)
    else:
        logger.info("⏭ 크롤링 단계 스킵")

    # 2️⃣ 전처리 단계 ---------------------------------------------------------
    if not args.skip_preprocess:
        pre_cli: List[str] = []
        if args.incremental:
            pre_cli.append("--incremental")
        if args.force:
            pre_cli.append("--force")
        run_script(PREPROCESS_SCRIPT, pre_cli)
    else:
        logger.info("⏭ 전처리 단계 스킵")

    # 3️⃣ 벡터 DB 구축 단계 ----------------------------------------------------
    if not args.skip_vectordb:
        vec_cli: List[str] = []
        if args.incremental:
            vec_cli.append("--incremental")
        if args.force:
            vec_cli.append("--force")
        run_script(BUILD_VECTORDB_SCRIPT, vec_cli)
    else:
        logger.info("⏭ 벡터 DB 단계 스킵")

    elapsed = (datetime.now() - t_start).total_seconds()
    logger.info("🎉 전체 완료 — %.1fs (%.1fm)", elapsed, elapsed / 60)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()