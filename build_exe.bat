@echo off
rem Builds NatureLab.exe (a thin launcher) with PyInstaller.
rem The backend/frontend are NOT bundled into the exe - they are launched
rem from the NatureLab folder, exactly as start.bat does.
setlocal
pip install pyinstaller==6.22.2
pyinstaller --noconfirm --onefile --windowed --name NatureLab launcher.pyw
if exist dist\NatureLab.exe (
  copy /y dist\NatureLab.exe NatureLab.exe >nul
  echo.
  echo NatureLab.exe created. Place it next to the backend/ and frontend/ folders.
) else (
  echo PyInstaller failed.
)
endlocal
