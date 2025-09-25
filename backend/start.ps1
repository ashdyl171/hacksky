Write-Host "Starting Deepfake Detection Backend..." -ForegroundColor Green
Write-Host ""
Write-Host "Make sure you have:" -ForegroundColor Yellow
Write-Host "1. Python installed" -ForegroundColor White
Write-Host "2. Dependencies installed (pip install -r requirements.txt)" -ForegroundColor White
Write-Host "3. FFmpeg installed and in PATH" -ForegroundColor White
Write-Host "4. Model file 'deepfake_detector (1).pkl' in the same directory" -ForegroundColor White
Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

try {
    python app.py
} catch {
    Write-Host "Error starting the backend: $_" -ForegroundColor Red
    Write-Host "Press any key to continue..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 