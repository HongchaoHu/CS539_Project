@echo off
echo ====================================
echo Data Analysis Agent - Starting...
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist ".venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Installing dependencies...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    REM Activate existing virtual environment
    call .venv\Scripts\activate.bat
)

REM Start the server in a new window
echo Starting API server...
start "Data Analysis Agent API" cmd /k ".venv\Scripts\activate.bat && python -m uvicorn app:app --port 8080 --reload"

REM Wait for server to start
echo Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Open browser
echo Opening web browser...
start http://localhost:8080

echo.
echo ====================================
echo Application is running!
echo Web interface: http://localhost:8080
echo ====================================
echo.
echo Press any key to exit this window (server will keep running)...
pause >nul
