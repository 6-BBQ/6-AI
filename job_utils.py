import json
from pathlib import Path

_JSON_PATH = Path(__file__).resolve().with_name("job_names.json")

with open(_JSON_PATH, "r", encoding="utf-8") as f:
    # {lower: original} 형태로 만들어 두면 대소문자·공백 걱정 없이 매칭 가능
    JOB_NAMES = {name.lower(): name for name in json.load(f)}

def find_job_in_text(text: str) -> str | None:
    """문자열에서 직업명을 찾아 원본 표기로 돌려준다(없으면 None)."""
    text_lower = text.lower()
    for lower_job in JOB_NAMES:
        if lower_job in text_lower:
            return JOB_NAMES[lower_job]
    return None
