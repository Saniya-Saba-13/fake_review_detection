@echo off
REM Double-click this file (or run it in the VS Code terminal) to set up and start the app.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing requirements (first time takes a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installing the requirements failed. Copy the red text above and send it to me.
  pause
  exit /b 1
)
python check_setup.py
echo.
python app.py
pause
