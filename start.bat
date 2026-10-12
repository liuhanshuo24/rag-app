@echo off
cd /d %~dp0

echo ==========================================
echo   RAG QnA Assistant - One-click Start (Windows)
echo ==========================================

rem --- 1. Create venv (prefer Python 3.12) ---
if not exist .venv (
    echo [1/4] Creating virtual env .venv ...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.12 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Please install Python 3.10-3.12.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate venv.
    pause
    exit /b 1
)

rem --- 2. Install dependencies ---
echo [2/4] Installing dependencies (first run may take 5-15 min)...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] pip install failed. Check network / pip source.
    pause
    exit /b 1
)

rem Use HF mirror for model downloads (China network)
set HF_ENDPOINT=https://hf-mirror.com
rem Load models from local cache only (ingest step above guarantees cache exists)
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

rem --- 3. Prepare config ---
if not exist .env (
    echo [3/4] First run: copy .env.example to .env and fill LLM_API_KEY
    copy .env.example .env >nul
)

rem --- 4. Build index and start ---
echo [4/4] Building KB index ...
python scripts\ingest.py
if errorlevel 1 (
    echo [ERROR] Ingest failed. Run "python scripts\ingest.py" manually for details.
    pause
    exit /b 1
)

echo.
echo Service URL: http://127.0.0.1:8000
echo Press Ctrl+C to stop
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
if errorlevel 1 (
    echo [ERROR] uvicorn failed to start. Port 8000 may be in use.
    pause
)
