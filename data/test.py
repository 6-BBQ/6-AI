import feedparser, itertools
CHANNEL_ID = "UCzps05UZ7XpX3SFoFyYIDow"
feed = feedparser.parse(f"https://www.youtube.com/feeds/videos.xml?channel_id={CHANNEL_ID}")

video_ids = [entry.yt_videoid for entry in itertools.islice(feed.entries, 20)]
with open("data/youtube_ids.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(video_ids))
