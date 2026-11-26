@echo off
setlocal EnableExtensions DisableDelayedExpansion
if not "%~1"=="" if /i not "%~1"=="pareto-residency" if /i not "%~1"=="native-rt" (
  echo Usage: Play-VIctory.cmd [pareto-residency^|native-rt] 1>&2
  exit /b 2
)
if not defined VICTORY_GAME_EXE (
  echo Set VICTORY_GAME_EXE to the full path of your own title executable. 1>&2
  exit /b 2
)
if not exist "%VICTORY_GAME_EXE%" (
  echo Executable not found: %VICTORY_GAME_EXE% 1>&2
  exit /b 2
)
set "runner=%~dp0target\release\victory.exe"
if not exist "%runner%" set "runner=%~dp0target\diag\victory.exe"
if not exist "%runner%" (
  echo Build first with: cargo build --release --locked -p victory 1>&2
  exit /b 2
)
if /i "%~1"=="native-rt" (
  if not defined VICTORY_NATIVE_RT set "VICTORY_NATIVE_RT=1"
  if not defined VICTORY_BVH_BUILDER set "VICTORY_BVH_BUILDER=1"
  if not defined VICTORY_NATIVE_RT_SUPPRESS_BUILDER set "VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1"
)
if not defined VICTORY_AMBIENT_AIR set "VICTORY_AMBIENT_AIR=1.5"
if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%LOCALAPPDATA%\VIctory\cache-pareto"
if not defined VICTORY_TEXTURE_NO_WAIT set "VICTORY_TEXTURE_NO_WAIT=1"
if not defined VICTORY_RELAY set "VICTORY_RELAY=2"
if not defined VICTORY_WIDE_BACK set "VICTORY_WIDE_BACK=1"
if not defined VICTORY_CARD_REHASH_RUNS set "VICTORY_CARD_REHASH_RUNS=1"
if not defined VICTORY_JOIN_COPIES set "VICTORY_JOIN_COPIES=1"
if not defined VICTORY_INDIRECT_WINDOW set "VICTORY_INDIRECT_WINDOW=1"
if not defined VICTORY_RELAY_COPY set "VICTORY_RELAY_COPY=1"
if not defined VICTORY_CARD_STEPS set "VICTORY_CARD_STEPS=2"
if not defined VICTORY_INDIRECT_BATCH set "VICTORY_INDIRECT_BATCH=2"
if not defined VICTORY_MAP_THREADS set "VICTORY_MAP_THREADS=1"
if not defined VICTORY_CARD_AWAKE set "VICTORY_CARD_AWAKE=1"
if not defined VICTORY_SHOW_AFTER set "VICTORY_SHOW_AFTER=1"
if not defined VICTORY_AMBIENT set "VICTORY_AMBIENT=0.8"
if not defined VICTORY_THIN set "VICTORY_THIN=1"
if not defined VICTORY_JOIN_PARTS set "VICTORY_JOIN_PARTS=8"
if not defined VICTORY_UPLOAD_AHEAD set "VICTORY_UPLOAD_AHEAD=1"
if not defined VICTORY_LEVELS_ON_DEMAND set "VICTORY_LEVELS_ON_DEMAND=1"
if not defined VICTORY_PORT_CBV set "VICTORY_PORT_CBV=1"
if not defined VICTORY_PORT_GUARD set "VICTORY_PORT_GUARD=1"
if not defined VICTORY_PORT_JOURNAL set "VICTORY_PORT_JOURNAL=1"
if not defined VICTORY_INPLACE set "VICTORY_INPLACE=2"
if not defined VICTORY_HEAP_WORKER set "VICTORY_HEAP_WORKER=1"
if not defined VICTORY_JOIN_BARRIERS set "VICTORY_JOIN_BARRIERS=1"
if not defined VICTORY_SCOPED_UAV_BARRIERS set "VICTORY_SCOPED_UAV_BARRIERS=1"
if not defined VICTORY_CARD_BUFFERS_MIN_KB set "VICTORY_CARD_BUFFERS_MIN_KB=64"
if not defined VICTORY_LOG if not defined RUST_LOG set "VICTORY_LOG=warn"
"%runner%" run "%VICTORY_GAME_EXE%" --console series-s --execute
exit /b %errorlevel%
