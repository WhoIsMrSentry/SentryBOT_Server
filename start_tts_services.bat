@echo off
setlocal
set START_SERVICES=1
set TTS_PRELOAD_ALL=1

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

REM Preload Piper + XTTS in memory
"%PY%" -X utf8 "%ROOT%tts_preload.py"

endlocal
