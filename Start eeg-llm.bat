@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-eeg-llm.ps1" %*
set "result=%errorlevel%"
echo.
if not "%result%"=="0" echo Setup stopped. The error and next steps are above.
pause
exit /b %result%
