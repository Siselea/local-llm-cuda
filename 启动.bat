@echo off
rem ===========================================================================
rem  engine double-click launcher (pure ASCII on purpose).
rem
rem  What it does:
rem    1. locate engine.exe (this folder, .\build, or ..\build)
rem    2. detect an engine already listening on the port
rem    3. ask engine.exe to scan for .gguf models, show them, let the user pick
rem    4. ask for the context length (default 4096)
rem    5. start `engine.exe serve ...`, which opens the browser once it listens
rem
rem  Optional command line (for shortcuts / scripted starts):
rem    launcher.bat [model-number-or-path] [ctx]
rem
rem  Why chcp happens late: switching the console code page breaks `set /p` when
rem  stdin is redirected, and the model table is ASCII anyway.  The engine prints
rem  UTF-8 Chinese, so the code page is switched right before it starts.
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
title engine local chat launcher
cd /d "%~dp0"

set "PORT=7030"
set "HOST=127.0.0.1"

rem ---- 1) locate engine.exe ------------------------------------------------
set "ENGINE="
if exist "%~dp0engine.exe" set "ENGINE=%~dp0engine.exe"
if not defined ENGINE if exist "%~dp0build\engine.exe" set "ENGINE=%~dp0build\engine.exe"
if not defined ENGINE if exist "%~dp0..\build\engine.exe" set "ENGINE=%~dp0..\build\engine.exe"
if not defined ENGINE (
  echo.
  echo [ERROR] engine.exe was not found.
  echo         Put engine.exe next to this .bat, or keep the build\ folder next to it.
  echo         Build it first with:  bash build.sh
  echo.
  pause
  exit /b 1
)
echo Using engine: !ENGINE!
echo.

rem ---- 2) already running? -------------------------------------------------
netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if not errorlevel 1 (
  echo An engine is already listening on %HOST%:%PORT%.
  choice /c ORQ /n /m "Open the browser [O], Restart the engine [R], Quit [Q]? "
  if errorlevel 3 exit /b 0
  if errorlevel 2 goto restart
  start "" "http://%HOST%:%PORT%/"
  exit /b 0
)

:pick
echo ===========================================================================
echo  Models found by the engine scan
echo ===========================================================================
rem stderr is the localized "no model" hint; it is re-shown after chcp below
"%ENGINE%" models 2>nul
if errorlevel 1 goto nomodel

rem ---- 3) model selection (arguments first, prompts otherwise) --------------
set "SEL=%~1"
set "CTX=%~2"
if defined SEL goto havesel
echo.
set /p "SEL=Select model [Enter = 1, or paste a full .gguf path]: "
if not defined SEL set "SEL=1"
:havesel
set "NUMONLY=1"
for /f "delims=0123456789" %%a in ("!SEL!") do set "NUMONLY=0"

rem ---- 4) context length -----------------------------------------------------
if defined CTX goto havectx
set /p "CTX=Context length in tokens [Enter = 4096]: "
if not defined CTX set "CTX=4096"
:havectx
echo.

chcp 65001 >nul 2>nul
if "!NUMONLY!"=="1" (
  echo Starting: engine.exe serve --pick !SEL! --ctx !CTX! --port %PORT%
) else (
  echo Starting: engine.exe serve "!SEL!" --ctx !CTX! --port %PORT%
)
echo (The first load can take a while; the browser opens automatically when
echo  the server is ready. Close this window or press Ctrl+C to stop.)
echo.
if "!NUMONLY!"=="1" (
  "%ENGINE%" serve --pick !SEL! --ctx !CTX! --port %PORT% --host %HOST% --open-browser --yes
) else (
  "%ENGINE%" serve --model "!SEL!" --ctx !CTX! --port %PORT% --host %HOST% --open-browser --yes
)
set "RC=%ERRORLEVEL%"
echo.
echo Engine exited with code %RC%.
pause
exit /b %RC%

rem ---- restart branch ------------------------------------------------------
:restart
echo Stopping the running engine on %HOST%:%PORT% ...
set "HTTP_CODE="
for /f %%C in ('curl -s -m 3 -o nul -w "%%{http_code}" -X POST "http://%HOST%:%PORT%/shutdown" 2^>nul') do set "HTTP_CODE=%%C"
if not defined HTTP_CODE (
  echo [WARN] HTTP /shutdown failed ^(curl missing or server busy^).
) else if not "!HTTP_CODE!"=="200" (
  echo [WARN] HTTP /shutdown returned !HTTP_CODE! ^(not 200^).
)
for /l %%i in (1,1,20) do (
  ping -n 2 127.0.0.1 >nul 2>nul
  netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
  if errorlevel 1 goto restarted
)
echo Port %PORT% is still busy; resolving only its listening PID ...
set "FOREIGN="
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  set "PID=%%P"
  set "IMG="
  for /f "tokens=1 delims=," %%I in ('tasklist /fi "PID eq !PID!" /fo csv /nh 2^>nul') do set "IMG=%%~I"
  if /i "!IMG!"=="engine.exe" (
    echo [3/3] taskkill /pid !PID! ^(engine.exe on port %PORT%^) ...
    taskkill /f /pid !PID! >nul 2>nul
  ) else (
    if defined IMG (
      echo [WARN] port %PORT% is held by !IMG! ^(PID !PID!^), not engine.exe; refusing to kill it.
      set "FOREIGN=1"
    )
  )
)
for /l %%i in (1,1,20) do (
  ping -n 2 127.0.0.1 >nul 2>nul
  netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
  if errorlevel 1 goto restarted
)
if defined FOREIGN (
  echo [ERROR] Port %PORT% is held by a non-engine process; refusing to restart.
  echo         Close it manually, or choose a different port.
  pause
  exit /b 1
)
echo [ERROR] Could not stop the engine on port %PORT%.
echo         No unrelated engine.exe process was touched.
pause
exit /b 1
:restarted
echo.
goto pick

:nomodel
chcp 65001 >nul 2>nul
rem stdout suppressed: only the UTF-8 "where to put a .gguf" hint is re-printed
"%ENGINE%" models 1>nul
echo.
echo [ERROR] No .gguf model was found. See the search folders above.
echo         Copy a model into one of them, or paste a full path at the prompt.
echo.
pause
exit /b 1
