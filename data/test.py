import yt_dlp

query = "던파 가이드"
search_url = f"ytsearch20:{query}"

ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
video_ids = []

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    result = ydl.extract_info(search_url, download=False)
    for entry in result['entries']:
        video_ids.append(entry['id'])

with open("data/youtube_ids.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(video_ids))

print(f"{len(video_ids)}개의 비디오 ID가 저장되었습니다.")
