@echo off
REM run.bat - Changeling Imago Realtime. Sets up .\venv the first time (and again when requirements.txt changes),
REM fetches the swap model once, then starts app.py. Options pass through: run.bat --camera 1 ...
setlocal
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Creating venv ...
    py -3.11 -m venv venv || goto :fail
)
fc /b requirements.txt venv\requirements.installed >nul 2>&1
if errorlevel 1 (
    echo Installing requirements - the first time downloads about 3 GB ...
    "%PY%" -m pip install -r requirements.txt || goto :fail
    copy /y requirements.txt venv\requirements.installed >nul
)
if not exist "models\inswapper_128.onnx" (
    echo Downloading the swap model, about 550 MB, once ...
    curl -L --fail -o "models\inswapper_128.onnx.part" "https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx" || goto :fail
    move /y "models\inswapper_128.onnx.part" "models\inswapper_128.onnx" >nul
)
if not exist "models\xseg_1.onnx" (
    echo Downloading the occlusion model, about 70 MB, once ...
    curl -L --fail -o "models\xseg_1.onnx.part" "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx" || goto :fail
    move /y "models\xseg_1.onnx.part" "models\xseg_1.onnx" >nul
)
"%PY%" app.py %*
if errorlevel 1 pause
exit /b
:fail
echo Setup failed: Changeling Imago Realtime needs Python 3.11 and an NVIDIA GPU (CUDA 12 drivers).
pause
exit /b 1
