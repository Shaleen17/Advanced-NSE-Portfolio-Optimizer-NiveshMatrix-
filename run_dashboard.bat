@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
    if errorlevel 1 (
        python -m streamlit run app.py
        exit /b %errorlevel%
    )
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
