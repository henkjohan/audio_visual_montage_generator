@echo off
echo Starting Autonomous Audio-Visual Montage Generator...

REM Change directory to the folder where this .bat file is located
cd /d "%~dp0"

REM Activate the virtual environment (use 'call' so the script doesn't exit here)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist ".env\Scripts\activate.bat" (
    call .env\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    pause
    exit /b 1
)

REM Run the python script with unbuffered output (-u)
python -u main.py

REM Keep the window open when finished so you can read the final output
echo.
pause
