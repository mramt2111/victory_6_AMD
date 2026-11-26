@echo off
setlocal
if not "%~1"=="" goto terminal
py -3 -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0scripts\launcher.py"
    goto done
)
set "gui_python="
if defined PYENV (
    for /d %%V in ("%PYENV%\versions\*") do if not defined gui_python call :try_python "%%~fV\python.exe"
)
if defined gui_python (
    "%gui_python%" "%~dp0scripts\launcher.py"
    goto done
)
call python -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
    call python "%~dp0scripts\launcher.py"
    goto done
)
echo Install Python 3.10 or newer with Tcl/Tk from https://www.python.org/downloads/ to open the game library.
pause
exit /b 1

:terminal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\play-rdna.ps1" %*

:done
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%

:try_python
if not exist "%~1" exit /b
"%~1" -c "import sys, tkinter; sys.exit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 set "gui_python=%~1"
exit /b
