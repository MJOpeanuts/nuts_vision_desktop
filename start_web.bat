@echo off
REM Startup script for nuts_vision web interface (Windows)
REM Always uses the Python of the local virtual environment (venv), never a global one.

echo ==========================================
echo nuts_vision - Web Interface Launcher
echo ==========================================
echo.

set "VENV_PY=%~dp0venv\Scripts\python.exe"
cd /d "%~dp0"

if not exist "%VENV_PY%" (
    echo Virtual environment not found, creating it...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment. Install Python 3.12 and retry.
        pause
        exit /b 1
    )
)

echo Installing dependencies with %VENV_PY% ...
"%VENV_PY%" -m pip install -q --timeout 30 -r requirements.txt
if errorlevel 1 (
    echo WARNING: Failed to install dependencies ^(check your network connection^)
    echo Manual command: "%VENV_PY%" -m pip install -r requirements.txt
)
echo.

REM ONNX must be present before starting: no install during analysis
"%VENV_PY%" check_onnx.py
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

REM Optional .env (not required)
if exist ".env" (
    for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do set %%a
)

if not defined STREAMLIT_PORT set STREAMLIT_PORT=8501

set /a MAX_ATTEMPTS=10
set /a ATTEMPT=0

:check_port
netstat -aon 2>nul | findstr /R ":%STREAMLIT_PORT% " | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Port %STREAMLIT_PORT% is in use, trying next port...
    set /a STREAMLIT_PORT+=1
    set /a ATTEMPT+=1
    if %ATTEMPT% LSS %MAX_ATTEMPTS% goto check_port
    echo ERROR: Could not find an available port after %MAX_ATTEMPTS% attempts.
    pause
    exit /b 1
)

echo ==========================================
echo Starting nuts_vision Web Interface...
echo ==========================================
echo http://localhost:%STREAMLIT_PORT%
echo Press Ctrl+C to stop the server
echo.

"%VENV_PY%" -m streamlit run app.py --server.port %STREAMLIT_PORT% --server.address localhost

pause
