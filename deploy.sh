#!/bin/bash
# 6-AI 프로젝트 자동 배포 스크립트
# 사용법: bash deploy.sh

echo "🚀 6-AI 프로젝트 자동 배포 시작"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 에러 발생 시 스크립트 중단
set -e

# 함수 정의
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 환경 검사
log_info "1️⃣ 환경 검사 중..."

# Python 버전 확인
if ! python3 --version &> /dev/null; then
    log_error "Python3가 설치되지 않았습니다"
    exit 1
fi

# .env 파일 확인
if [ ! -f ".env" ]; then
    log_warn ".env 파일이 없습니다. .env.example을 복사합니다..."
    cp .env.example .env
    log_warn "⚠️  .env 파일을 편집하여 API 키를 설정해주세요!"
    echo "   - GEMINI_API_KEY"
    echo "   - JWT_SECRET_KEY"
    read -p "API 키 설정을 완료했나요? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "API 키 설정을 완료한 후 다시 실행해주세요"
        exit 1
    fi
fi

# 2. 가상환경 설정
log_info "2️⃣ 가상환경 설정 중..."

if [ ! -d "venv" ]; then
    log_info "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 3. 의존성 설치
log_info "3️⃣ 의존성 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. 데이터 준비 확인
log_info "4️⃣ 데이터 준비 상태 확인..."

# 벡터 DB 존재 여부 확인
if [ ! -d "vector_db/chroma" ] || [ ! "$(ls -A vector_db/chroma 2>/dev/null)" ]; then
    log_warn "벡터 DB가 없습니다. 데이터 파이프라인을 실행합니다..."
    
    # 파이프라인 실행
    log_info "📥 크롤링 및 벡터 DB 구축 중... (시간이 소요될 수 있습니다)"
    python pipeline.py --pages 5 --yt-max 10
    
    if [ $? -eq 0 ]; then
        log_info "✅ 데이터 파이프라인 완료"
    else
        log_error "❌ 데이터 파이프라인 실패"
        exit 1
    fi
else
    log_info "✅ 기존 벡터 DB 발견 - 건너뛰기"
fi

# 5. 서비스 시작
log_info "5️⃣ 서비스 시작 준비..."

# 포트 확인
PORT=${PORT:-8000}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    log_warn "포트 $PORT이 이미 사용 중입니다"
    read -p "계속 진행하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 백그라운드에서 서비스 시작
log_info "🚀 API 서버 시작 중..."
echo "📊 로그 확인: tail -f logs/api_main.log"
echo "🌐 API 문서: http://localhost:$PORT/docs"
echo "🛑 서버 종료: Ctrl+C"
echo ""

# 서버 실행
python -m api.main

echo ""
log_info "🎉 배포 완료!"
