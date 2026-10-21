@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua cai dat. Chay: py -3.11 -m venv .venv
  echo Sau do: .venv\Scripts\python.exe -m pip install -e ".[test]"
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo Khong tim thay FFmpeg. Hay them thu muc bin cua FFmpeg vao PATH.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m logo_remover.app
if errorlevel 1 pause
