from __future__ import annotations
import argparse, sys, json, textwrap, os
from pathlib import Path
from datetime import datetime

# 1️⃣ 개별 크롤러 import
from official_crawler import crawl_df
from dc_crawler       import crawl_dcinside
from arca_crawler     import crawl_arca
from youtube_crawler  import crawl_youtube

def load_yt_ids(path: str | Path) -> list[str]:
    """텍스트 파일(한 줄 = video_id) → 리스트"""
    if not path:
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            print(f"✅ 유튜브 ID 파일 로드 성공: {len(ids)}개 ID 찾음")
            return ids
    except Exception as e:
        print(f"❌ 유튜브 ID 파일 로드 실패: {e}")
        # 절대 경로로 시도
        try:
            project_root = Path(__file__).resolve().parents[1]
            abs_path = project_root / path
            print(f"🔄 절대 경로로 재시도: {abs_path}")
            with open(abs_path, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                print(f"✅ 유튜브 ID 파일 로드 성공: {len(ids)}개 ID 찾음")
                return ids
        except Exception as e2:
            print(f"❌ 절대 경로 시도도 실패: {e2}")
            return []

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            ▶ 던파 스펙업 가이드용 통합 크롤링 스크립트
            예시:
              python -m crawler.crawler --pages 8 --depth 2 \\
                    --yt-list data/youtube_ids.txt
        """)
    )
    parser.add_argument("--pages", type=int, default=2,  help="각 게시판 최대 페이지 수")
    parser.add_argument("--depth", type=int, default=2,  help="본문 링크 재귀 depth")
    parser.add_argument("--yt-list", type=str, default="data/youtube_ids.txt", help="YouTube video_id 리스트 txt 경로")
    args = parser.parse_args()

    print(f"\n🔔 통합 크롤링 시작 ({datetime.now():%Y-%m-%d %H:%M:%S})\n"
          f"   - pages = {args.pages}, depth = {args.depth}\n"
          f"   - yt-list = {args.yt_list}")



    # 5️⃣ YouTube 자막
    # 현재 작업 디렉토리 출력
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"유튜브 ID 파일 경로: {args.yt_list}")
    
    yt_ids = load_yt_ids(args.yt_list)
    if yt_ids:
        print(f"\n🟨 [4/4] YouTube 자막 (ID {len(yt_ids)}개)")
        crawl_youtube(yt_ids)
    else:
        print("\n🟨 [4/4] YouTube 자막 ▶️ (ID 리스트 없음 → 건너뜀)")

    print("\n✅ 모든 크롤링 완료!")

if __name__ == "__main__":
    # 패키지 밖에서 python crawler/crawler.py로도 실행 가능하도록 경로 보정
    if __package__ is None and __name__ == "__main__":
        # 프로젝트 루트(rpgpt/)를 PYTHONPATH에 추가
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()