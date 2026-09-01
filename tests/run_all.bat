@echo off
setlocal
set "ROOT=%~dp0.."
python "%~dp0test_backend.py"
if errorlevel 1 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test_launcher.ps1"
if errorlevel 1 exit /b 1
pushd "%~dp0"
if not exist node_modules call npm ci --no-audit --no-fund
call npm test
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
