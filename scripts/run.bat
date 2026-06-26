@echo off
:: Force the working directory to the project root (one level up from scripts)
cd /d "%~dp0.."

echo ==================================================
echo Starting ADAS Mobile Companion Server
echo ==================================================
echo.

:: Call the virtual environment python directly to avoid activation issues
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\src\server.py"

pause
