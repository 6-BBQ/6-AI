import json, time
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

SAVE_PATH = Path("data/raw/youtube_row.json")
LANGS_PRIORITY = ['ko', 'ko-Hang', 'en']  # 필요시 자동번역 사용

def fetch_transcript_text(video_id: str) -> str:
    """유튜브 영상의 자막을 가져오는 함수"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=LANGS_PRIORITY)
        return " ".join(seg['text'] for seg in transcript).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"⚠️ 영상 {video_id}에 자막이 없습니다.")
        return ""
    except Exception as e:
        print(f"❌ 영상 {video_id} 자막 가져오기 실패: {e}")
        return ""

def crawl_youtube(video_ids: list[str]):
    """유튜브 영상 ID 목록을 받아 자막을 수집하는 함수"""
    results = []
    success_count = 0
    failed_ids = []
    
    print(f"🔍 총 {len(video_ids)}개 유튜브 영상 처리 시작...")
    
    for i, vid in enumerate(video_ids, 1):
        print(f"   [{i}/{len(video_ids)}] 영상 ID: {vid} 처리 중...")
        text = fetch_transcript_text(vid)
        
        if not text:
            failed_ids.append(vid)
            continue
            
        results.append({
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": f"YOUTUBE_{vid}",
            "date": time.strftime("%Y-%m-%d"),
            "content": text
        })
        success_count += 1
        
        # 너무 빠른 연속 요청 방지
        time.sleep(5.0)
    
    # 결과가 없어도 빈 파일 생성
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 유튜브 처리 결과:")
    print(f"   - 성공: {success_count}/{len(video_ids)}개")
    print(f"   - 실패: {len(failed_ids)}/{len(video_ids)}개")
    if failed_ids:
        print(f"   - 실패한 ID: {', '.join(failed_ids)}")
    
    if success_count > 0:
        print(f"✅ 유튜브 {success_count}개 자막 저장 → {SAVE_PATH}")
    else:
        print(f"❌ 유튜브 자막 저장 실패 - 성공한 영상이 없습니다.")
