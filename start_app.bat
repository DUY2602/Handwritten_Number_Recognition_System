@echo off
echo ============================================================
echo   Handwritten Number Recognition System - Setup and Run
echo ============================================================
echo.
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
echo.
echo [2/3] Checking and installing dependencies...
echo (Note: This may take a few minutes during the first run)
pip install -r requirements.txt
echo.
echo [3/3] Starting the Web Application...
echo ============================================================
python app.py
pause
