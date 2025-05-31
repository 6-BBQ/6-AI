@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 6-AI 프로젝트 자동 배포 스크립트 (Windows)
REM 사용법: deploy.bat

echo 🚀 6-AI 프로젝트 자동 배포 시작
echo ================================
echo.

REM 1. 환경 검사
echo [INFO] 1️⃣ 환경 검사 중...

REM Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았거나 PATH에 없습니다
    pause
    exit /b 1
)

REM .env 파일 확인
if not exist ".env" (
    echo [WARN] .env 파일이 없습니다. .env.example을 복사합니다...
    copy .env.example .env >nul
    echo [WARN] ⚠️  .env 파일을 편집하여 API 키를 설정해주세요!
    echo    - GEMINI_API_KEY
    echo    - JWT_SECRET_KEY
    echo.
    set /p answer="API 키 설정을 완료했나요? (y/N): "
    if /i not "!answer!"=="y" (
        echo [ERROR] API 키 설정을 완료한 후 다시 실행해주세요
        pause
        exit /b 1
    )
)

REM 2. 가상환경 설정
echo [INFO] 2️⃣ 가상환경 설정 중...

if not exist "venv" (
    echo [INFO] 가상환경 생성 중...
    python -m venv venv
)

REM 가상환경 활성화
call venv\Scripts\activate.bat

REM 3. 의존성 설치
echo [INFO] 3️⃣ 의존성 설치 중...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. 데이터 준비 확인
echo [INFO] 4️⃣ 데이터 준비 상태 확인...

REM 벡터 DB 존재 여부 확인
if not exist "vector_db\chroma" (
    goto run_pipeline
)

dir /b "vector_db\chroma" 2>nul | findstr . >nul
if errorlevel 1 (
    goto run_pipeline
) else (
    echo [INFO] ✅ 기존 벡터 DB 발견 - 건너뛰기
    goto start_service
)

:run_pipeline
echo [WARN] 벡터 DB가 없습니다. 데이터 파이프라인을 실행합니다...
echo [INFO] 📥 크롤링 및 벡터 DB 구축 중... (시간이 소요될 수 있습니다)

python pipeline.py --pages 5 --yt-max 10
if errorlevel 1 (
    echo [ERROR] ❌ 데이터 파이프라인 실패
    pause
    exit /b 1
)
echo [INFO] ✅ 데이터 파이프라인 완료

:start_service
REM 5. 서비스 시작
echo [INFO] 5️⃣ 서비스 시작 준비...

REM 포트 확인
set PORT=8000
netstat -an | findstr :!PORT! >nul 2>&1
if not errorlevel 1 (
    echo [WARN] 포트 !PORT!이 이미 사용 중입니다
    set /p answer="계속 진행하시겠습니까? (y/N): "
    if /i not "!answer!"=="y" (
        exit /b 1
    )
)

REM 서비스 시작
echo [INFO] 🚀 API 서버 시작 중...
echo 📊 로그 확인: type logs\api_main.log
echo 🌐 API 문서: http://localhost:!PORT!/docs
echo 🛑 서버 종료: Ctrl+C
echo.

python -m api.main

echo.
echo [INFO] 🎉 배포 완료!
pause
