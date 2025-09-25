@echo off
echo Starting Deepfake Detection Backend...
echo.
echo Make sure you have:
echo 1. Python installed
echo 2. Dependencies installed (pip install -r requirements.txt)
echo 3. FFmpeg installed and in PATH
echo 4. Model file "deepfake_detector (1).pkl" in the same directory
echo.
echo Starting server on http://localhost:8000
echo.
python app.py
pause 