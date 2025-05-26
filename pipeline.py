import subprocess
import sys

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


if __name__ == "__main__":
    print("\n🚀 던파 스펙업 파이프라인 전체 실행 시작\n")

    # 1️⃣ 크롤링 (하이브리드 모드 사용)
    run_script("crawlers/crawler.py", [
        "--pages", "30",
        "--depth", "2",
        "--yt-mode", "hybrid",
        "--yt-max", "30",
        "--merge",
        "--incremental"
    ])

    # 2️⃣ 전처리
    run_script("preprocessing/preprocess.py")

    # 3️⃣ 벡터 DB 구축
    run_script("vectorstore/build_vector_db.py")

    print("\n🎉 전체 파이프라인 완료!\n")
