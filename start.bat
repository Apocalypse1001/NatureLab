@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=python"
set "PORT=8756"
set "URL=http://127.0.0.1:%PORT%/"

echo ============================================
echo   NatureLab
echo   start backend + open the app in a browser
echo ============================================

rem ---- 1. Python backend (background, minimized window) ----
start "NatureLab-backend" /min cmd /c "cd /d "%ROOT%backend" && %PYTHON% -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% 2>nul"

echo waiting for the backend on %URL%
powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){ try { $r=Invoke-WebRequest -Uri '%URL%api/status' -UseBasicParsing -TimeoutSec 1; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
  echo [ERROR] backend did not start in time.
  echo   Install Python deps:  pip install -r "%ROOT%backend\requirements.txt"
  pause
  exit /b 1
)

rem ---- 2. Frontend: build once if dist is missing ----
if not exist "%ROOT%frontend\dist\index.html" (
  echo [setup] building the frontend ...
  set "NPM=npm"
  if exist "%ROOT%runtime\node\npm.cmd" set "NPM=%ROOT%runtime\node\npm.cmd"
  pushd "%ROOT%frontend"
  call "%NPM%" ci --no-audit --no-fund
  call "%NPM%" run build
  popd
)

rem ---- 3. Open the app ----
start "" "%URL%"
echo NatureLab is running: %URL%
echo (close the "NatureLab-backend" window to stop the app)
endlocal
