from io import BytesIO
import os
import requests
import time
import numpy as np
from pathlib import Path
import easyocr
from PIL import Image

# 전역 변수로 모델 로드 (1회만)
def load_ocr_model():
    try:
        # 모델 다운로드 경로 명시
        os.environ["EASYOCR_DOWNLOAD_DIR"] = str(Path("./models").absolute())
        Path("./models").mkdir(exist_ok=True)
        
        reader = easyocr.Reader(['ko', 'en'], gpu=False, download_enabled=True)
        return reader
    except Exception as e:
        print(f"[OCR 오류] 모델 로드 실패: {e}")
        return None

_READER = load_ocr_model()

def ocr_image_from_url(url, timeout=15, max_retries=3, retry_delay=2):
    """향상된 OCR 처리 함수"""
    if _READER is None:
        return ""
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            # 이미지 다운로드
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            
            # 이미지 크기 검증
            content_length = len(resp.content)
            if content_length < 1000:  # 너무 작은 파일은 이미지가 아닐 가능성 높음
                return ""
            
            # 이미지 변환: bytes → PIL Image → numpy array (EasyOCR 호환)
            try:
                pil_image = Image.open(BytesIO(resp.content))
                # 이미지가 RGBA인 경우 RGB로 변환 (일부 PNG 처리를 위해)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                # PIL Image → numpy array (EasyOCR 권장 입력 형식)
                img_array = np.array(pil_image)
                
                # numpy array로 OCR 실행 (가장 안정적인 입력 형식)
                result = _READER.readtext(img_array, detail=0, paragraph=True)
                text = "\n".join(result).strip()
                return text
            
            except Exception:
                # 대체 방법 시도 - bytes 직접 전달
                try:
                    result = _READER.readtext(resp.content, detail=0, paragraph=True)
                    text = "\n".join(result).strip()
                    return text
                except Exception:
                    pass
                    
        except Exception:
            pass
        
        # 마지막 시도가 아니면 대기
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return ""

def save_if_large(text, min_len=20):
    """텍스트 길이 기반 필터링"""
    return text if len(text) >= min_len else ""