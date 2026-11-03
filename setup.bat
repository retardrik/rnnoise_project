@echo off
setlocal
cd /d "%~dp0"
title RNNoise project setup
echo.
echo ============================================================
echo  RNNoise project setup  (safe to run again if it stops)
echo ============================================================

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" if exist "C:\Program Files\Python312\python.exe" set "PY=C:\Program Files\Python312\python.exe"

echo.
echo [1/5] Python 3.12
if exist "%PY%" (
    echo   Already installed: %PY%
) else (
    echo   Installing Python 3.12 with winget. Accept any prompt that appears.
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if not exist "%PY%" winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if not exist "%PY%" if exist "C:\Program Files\Python312\python.exe" set "PY=C:\Program Files\Python312\python.exe"
)
if not exist "%PY%" (
    echo.
    echo   Python could not be installed automatically.
    echo   Install Python 3.12 from python.org, then run this file again.
    goto :fail
)

echo.
echo [2/5] VS Code project settings + virtual environment (venv)
if not exist ".vscode" mkdir ".vscode"
copy /y "vscode_config\settings.json" ".vscode\settings.json" >nul
copy /y "vscode_config\extensions.json" ".vscode\extensions.json" >nul
if not exist "venv\Scripts\python.exe" (
    "%PY%" -m venv venv || goto :fail
) else (
    echo   venv already exists
)

echo.
echo [3/5] Python packages (a few minutes the first time, progress shown below)
"venv\Scripts\python.exe" -m pip install --upgrade pip --retries 10 --timeout 60 || goto :fail
"venv\Scripts\python.exe" -m pip install -r requirements.txt --retries 10 --timeout 60 || goto :fail
echo   Packages installed

echo.
echo [4/5] VS Code extensions
set "CODE=%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd"
if not exist "%CODE%" set "CODE=code"
call "%CODE%" --install-extension ms-python.python
call "%CODE%" --install-extension ms-toolsai.jupyter
call "%CODE%" --install-extension anthropic.claude-code --force

echo.
echo [5/5] Test: running official RNNoise on test_noisy.wav
"venv\Scripts\python.exe" run_rnnoise.py test_noisy.wav outputs\test_denoised.wav || goto :fail

echo.
echo ============================================================
echo  ALL DONE. Opening the project in VS Code...
echo ============================================================
call "%CODE%" "%~dp0."
pause
exit /b 0

:fail
echo.
echo ************************************************************
echo  Setup stopped with an error. Take a screenshot of this
echo  window and send it to Claude.
echo ************************************************************
pause
exit /b 1
