@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem One-click Windows PC diagnostics for VIctory.
rem Usage: Diagnose-System.bat [GAME_FOLDER]

set "SCRIPT=%~dp0scripts\diagnose.ps1"
if not exist "%SCRIPT%" (
  echo Diagnostic script not found: %SCRIPT% 1>&2
  exit /b 2
)

if "%~1"=="" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -GameDir "%~1"
)
set "rc=%errorlevel%"

rem Keep the console window open when launched by double-clicking in Explorer.
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 pause
exit /b %rc%
