@echo off
REM Quantum-Classical ML Dashboard Launcher
REM One-click script to launch the Streamlit application

echo ========================================
echo  Quantum-Classical ML Dashboard
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [!] Virtual environment not found.
    echo [*] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [*] Installing dependencies...
    pip install -r requirements.txt
    pip install -r app\requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [*] Checking Gemini API key...
if not exist ".env" (
    echo [!] .env file not found. Please create .env with GEMINI_API_KEY
)

echo [*] Launching Streamlit dashboard...
echo [*] Dashboard will open at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app\streamlit_app.py

pause
