"""
6-AI 프로젝트 공통 로깅 설정
"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """컬러 로그 포매터 (개발 환경용)"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 청록색
        'INFO': '\033[32m',     # 녹색
        'WARNING': '\033[33m',  # 노란색
        'ERROR': '\033[31m',    # 빨간색
        'CRITICAL': '\033[35m', # 자홍색
        'RESET': '\033[0m'      # 리셋
    }
    
    def format(self, record):
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str, 
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    통일된 로거 설정
    
    Args:
        name: 로거 이름 (보통 모듈명)
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: 파일 로깅 여부
        log_to_console: 콘솔 로깅 여부
        log_dir: 로그 파일 디렉토리
    
    Returns:
        설정된 로거 인스턴스
    """
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 공통 포맷터
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 파일 핸들러
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # 일반 로그 파일 (로테이션)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 에러 전용 로그 파일
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}_error.log",
            maxBytes=5*1024*1024,   # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # 콘솔 핸들러
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 개발/운영 환경별 설정
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        # 운영 환경: WARNING 이상만 콘솔 출력
        if log_to_console:
            console_handler.setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    간편한 로거 획득 함수
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
    
    Returns:
        로거 인스턴스
    """
    # config에서 로그 레벨 가져오기 (순환 임포트 방지)
    try:
        from config import config
        log_level = config.LOG_LEVEL
        log_to_file = config.LOG_TO_FILE
        log_dir = config.LOG_DIR
        env = config.ENVIRONMENT
    except ImportError:
        # config 모듈이 없으면 환경변수에서 직접 가져오기
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        log_dir = os.getenv("LOG_DIR", "logs")
        env = os.getenv("ENVIRONMENT", "development").lower()
    
    return setup_logger(
        name=name.replace(".", "_"),  # 파일명에 적합하게 변경
        level=log_level,
        log_to_file=log_to_file,
        log_to_console=True,
        log_dir=log_dir
    )


# 성능 측정 데코레이터
def log_execution_time(logger: Optional[logging.Logger] = None):
    """함수 실행 시간을 로깅하는 데코레이터 (sync/async 모두 지원)"""
    def decorator(func):
        import functools
        import asyncio
        
        @functools.wraps(func)  # 함수 메타데이터 보존
        async def async_wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = datetime.now()
            
            try:
                _logger.debug(f"시작: {func.__name__}")
                result = await func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                _logger.info(f"완료: {func.__name__} ({execution_time:.2f}초)")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                _logger.error(f"실패: {func.__name__} ({execution_time:.2f}초) - {str(e)}")
                raise
        
        @functools.wraps(func)  # 함수 메타데이터 보존
        def sync_wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = datetime.now()
            
            try:
                _logger.debug(f"시작: {func.__name__}")
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                _logger.info(f"완료: {func.__name__} ({execution_time:.2f}초)")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                _logger.error(f"실패: {func.__name__} ({execution_time:.2f}초) - {str(e)}")
                raise
        
        # async 함수인지 확인하여 적절한 래퍼 반환
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# 시스템 정보 로깅
def log_system_info(logger: logging.Logger):
    """시스템 정보를 로깅"""
    import platform
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
    
    logger.info("=== 시스템 정보 ===")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    if has_psutil:
        logger.info(f"CPU 코어: {psutil.cpu_count()}")
        logger.info(f"메모리: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # GPU 정보 (가능한 경우)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            logger.info("GPU: 사용 불가")
    except ImportError:
        logger.info("GPU: PyTorch 미설치")
