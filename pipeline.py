import subprocess
import sys
import argparse
import textwrap
from datetime import datetime

def run_script(path: str, args: list[str] = []):
    print(f"\n🟡 실행 중: {path} {' '.join(args)}")
    try:
        result = subprocess.run(
            [sys.executable, path] + args,
            check=True
        )
        print(f"✅ 완료: {path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생 ({path}): {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="던파 스펙업 AI 파이프라인 (증분 처리 지원)",
        epilog=textwrap.dedent("""
        예시:
          # 기본 실행 (증분 모드)
          python pipeline.py
          
          # 전체 파이프라인 (모든 데이터 재처리)
          python pipeline.py --full
          
          # 특정 단계만 실행
          python pipeline.py --skip-crawl
          python pipeline.py --skip-preprocess
          python pipeline.py --skip-vectordb
          
          # 강제 전체 재구축
          python pipeline.py --force
        """)
    )
    
    parser.add_argument(
        "--incremental", 
        action="store_true", 
        default=True,
        help="증분 처리 모드 (기본값)"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="전체 처리 모드 (증분 무시)"
    )
    
    parser.add_argument(
        "--pages", 
        type=int, 
        default=50, 
        help="크롤링할 페이지 수 (기본: 30)"
    )
    
    parser.add_argument(
        "--depth", 
        type=int, 
        default=3, 
        help="크롤링 재귀 깊이 (기본: 2)"
    )
    
    parser.add_argument(
        "--yt-mode", 
        type=str, 
        default="hybrid",
        choices=["channel", "search", "hybrid"],
        help="YouTube 크롤링 모드 (기본: hybrid)"
    )
    
    parser.add_argument(
        "--yt-max", 
        type=int, 
        default=20,
        help="YouTube 최대 영상 수"
    )
    
    parser.add_argument(
        "--skip-crawl", 
        action="store_true", 
        help="크롤링 단계 건너뛰기"
    )
    
    parser.add_argument(
        "--skip-preprocess", 
        action="store_true", 
        help="전처리 단계 건너뛰기"
    )
    
    parser.add_argument(
        "--skip-vectordb", 
        action="store_true", 
        help="벡터 DB 구축 단계 건너뛰기"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="모든 단계에서 기존 데이터 강제 덮어쓰기"
    )
    
    args = parser.parse_args()
    
    # 전체 모드 검사
    if args.full:
        args.incremental = False
    
    # 시작 메시지
    mode_emoji = "🔄" if args.incremental else "🚀"
    mode_name = "증분" if args.incremental else "전체"
    
    print(f"\n{mode_emoji} 던파 스펙업 파이프라인 {mode_name} 실행 시작")
    print(f"   📅 시작 시간: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"   🔧 모드: {mode_name} 처리")
    print(f"   📊 페이지 수: {args.pages}")
    print(f"   🔍 깊이: {args.depth}")
    print(f"   🎥 YouTube 모드: {args.yt_mode}")
    print(f"   📹 YouTube 최대: {args.yt_max}")
    
    pipeline_start = datetime.now()
    
    # 1️⃣ 크롤링 단계
    if not args.skip_crawl:
        print("\n" + "="*50)
        print("1️⃣ 크롤링 단계")
        print("="*50)
        
        crawl_args = [
            "--pages", str(args.pages),
            "--depth", str(args.depth),
            "--yt-mode", args.yt_mode,
            "--yt-max", str(args.yt_max),
            "--merge"
        ]
        
        if args.incremental:
            crawl_args.append("--incremental")
        else:
            crawl_args.append("--full")
            
        run_script("crawlers/crawler.py", crawl_args)
    else:
        print("\n⏭️ 크롤링 단계 건너뛰기")

    # 2️⃣ 전처리 단계
    if not args.skip_preprocess:
        print("\n" + "="*50)
        print("2️⃣ 전처리 단계")
        print("="*50)
        
        preprocess_args = []
        
        if args.incremental:
            preprocess_args.append("--incremental")
            
        if args.force:
            preprocess_args.append("--force")
            
        run_script("preprocessing/preprocess.py", preprocess_args)
    else:
        print("\n⏭️ 전처리 단계 건너뛰기")

    # 3️⃣ 벡터 DB 구축 단계
    if not args.skip_vectordb:
        print("\n" + "="*50)
        print("3️⃣ 벡터 DB 구축 단계")
        print("="*50)
        
        vectordb_args = []
        
        if args.incremental:
            vectordb_args.append("--incremental")
            
        if args.force:
            vectordb_args.append("--force")
            
        run_script("vectorstore/build_vector_db.py", vectordb_args)
    else:
        print("\n⏭️ 벡터 DB 구축 단계 건너뛰기")

    # 완료 메시지
    pipeline_end = datetime.now()
    total_time = (pipeline_end - pipeline_start).total_seconds()
    
    print("\n" + "="*50)
    print(f"🎉 전체 파이프라인 완료! ({mode_name} 모드)")
    print(f"   📅 완료 시간: {pipeline_end:%Y-%m-%d %H:%M:%S}")
    print(f"   ⏱️ 총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    print("="*50)


if __name__ == "__main__":
    main()
