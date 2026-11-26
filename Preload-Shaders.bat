@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Preload and warm VIctory shaders into a local cache folder inside the game directory.
rem Usage: Preload-Shaders.bat [GAME_FOLDER]
rem You can also drag-and-drop your game folder onto this file, or copy it into the game folder.

set "SCAN_DIR=%~1"
if not defined SCAN_DIR if defined VICTORY_GAME_DIR set "SCAN_DIR=%VICTORY_GAME_DIR%"
if not defined SCAN_DIR if defined VICTORY_GAME_EXE for %%I in ("%VICTORY_GAME_EXE%") do set "SCAN_DIR=%%~dpI"
if not defined SCAN_DIR set "SCAN_DIR=%~dp0"
if "%SCAN_DIR:~-1%"=="\" set "SCAN_DIR=%SCAN_DIR:~0,-1%"

if not exist "%SCAN_DIR%" (
  echo Game folder not found: %SCAN_DIR% 1>&2
  exit /b 2
)

set "runner=%VICTORY_EXE%"
if not defined runner if exist "%~dp0target\release\victory.exe" set "runner=%~dp0target\release\victory.exe"
if not defined runner if exist "%~dp0target\diag\victory.exe" set "runner=%~dp0target\diag\victory.exe"
if not defined runner if exist "%~dp0victory.exe" set "runner=%~dp0victory.exe"
if not defined runner if exist "%SCAN_DIR%\victory.exe" set "runner=%SCAN_DIR%\victory.exe"
if not defined runner for %%I in (victory.exe) do if not "%%~$PATH:I"=="" set "runner=%%~$PATH:I"
if not exist "%runner%" (
  echo victory.exe not found. Set VICTORY_EXE, copy victory.exe next to this file, or build with: 1>&2
  echo   cargo build --release --locked -p victory 1>&2
  exit /b 2
)

if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%SCAN_DIR%\victory_shader_cache"
if not exist "%VICTORY_CACHE_DIR%" mkdir "%VICTORY_CACHE_DIR%"

echo [1/2] Pre-translating shaders in "%SCAN_DIR%"
echo       Cache folder: "%VICTORY_CACHE_DIR%"
"%runner%" shaders "%SCAN_DIR%"
set "rc=%errorlevel%"
if not "%rc%"=="0" (
  echo Shader pre-translation exited with code %rc%. 1>&2
  exit /b %rc%
)

if exist "%VICTORY_CACHE_DIR%\known-pipelines.bin" (
  echo [2/2] Warming recorded D3D12 pipelines into on-disk pipeline library...
  "%runner%" warm
  set "rc=%errorlevel%"
) else (
  echo [2/2] Skipping D3D12 PSO warm-up: known-pipelines.bin not recorded yet.
  echo       Launch once with Play-VIctory.cmd, then re-run Preload-Shaders.bat to warm GPU PSOs.
)

echo Done. Play-VIctory.cmd will automatically load from %VICTORY_CACHE_DIR%.
exit /b %rc%
