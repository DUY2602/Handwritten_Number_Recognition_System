@echo off
echo ============================================================
echo   Handwritten Number Recognition System - Dependency Setup
echo ============================================================
echo.
echo [1/2] Upgrading pip...
python -m pip install --upgrade pip
echo.
echo [2/2] Installing libraries from requirements.txt...
echo (Note: This process may take a few minutes due to large TensorFlow and PyTorch packages)
pip install -r requirements.txt
echo.
echo ============================================================
echo Setup complete! You can run the app with: python app.py
pause
