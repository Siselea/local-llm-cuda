@echo off
rem ===========================================================================
rem  Stop only THIS engine instance (port-scoped), not every engine.exe.
rem
rem  Order of attack:
rem    1. POST /shutdown on the configured port, and CHECK THE HTTP STATUS
rem    2. wait up to ~15 s for the port to be released
rem    3. if still busy: find the PID LISTENING on the port; kill only that PID
rem       (and only if its image is engine.exe)
rem    4. taskkill /im engine.exe ONLY with the explicit --all flag
rem
rem  Usage:
rem    stop.bat [--port N] [--all]
rem      --port N  port to stop (default 7030)
rem      --all     explicit escape hatch: kill all engine.exe processes
rem
rem  Pure ASCII output (the engine may not be running).
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>nul
title engine stop
cd /d "%~dp0"

set "PORT=7030"
set "HOST=127.0.0.1"
set "STOP_ALL=0"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--all" goto opt_all
if /i "%~1"=="/all" goto opt_all
if /i "%~1"=="--port" goto opt_port
echo [ERROR] unknown argument: %~1
echo Usage: stop.bat [--port N] [--all]
pause
exit /b 2

:opt_all
set "STOP_ALL=1"
shift
goto parse_args

:opt_port
if "%~2"=="" (
  echo [ERROR] --port requires a value
  pause
  exit /b 2
)
set "PORT=%~2"
shift
shift
goto parse_args

:args_done
echo Stopping the engine on %HOST%:%PORT% ...

rem ---- is anything listening? --------------------------------------------
set "LISTEN_PIDS="
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  set "LISTEN_PIDS=!LISTEN_PIDS! %%P"
)
if not defined LISTEN_PIDS (
  echo Nothing is listening on port %PORT%.
  if "%STOP_ALL%"=="1" (
    echo [--all] terminating every engine.exe as explicitly requested ...
    taskkill /f /im engine.exe >nul 2>nul
    if errorlevel 1 (
      echo No engine.exe process was running.
    ) else (
      echo engine.exe processes terminated ^(--all^).
    )
  ) else (
    echo No unrelated engine.exe process was touched.
    echo If an engine is running on another port, use: stop.bat --port N
  )
  echo.
  pause
  exit /b 0
)

rem ---- 1) ask the HTTP endpoint and verify its status --------------------
set "HTTP_CODE="
for /f %%C in ('curl -s -m 3 -o nul -w "%%{http_code}" -X POST "http://%HOST%:%PORT%/shutdown" 2^>nul') do set "HTTP_CODE=%%C"
if not defined HTTP_CODE (
  echo [1/3] HTTP /shutdown request failed ^(curl missing, timeout, or no HTTP server^).
) else if "%HTTP_CODE%"=="200" (
  echo [1/3] HTTP /shutdown returned 200.
) else (
  echo [1/3] HTTP /shutdown returned %HTTP_CODE% ^(not 200^); the server did not acknowledge shutdown.
)

rem ---- 2) wait for the port to be released -------------------------------
for /l %%i in (1,1,15) do (
  ping -n 2 127.0.0.1 >nul 2>nul
  netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
  if errorlevel 1 goto stopped
)

rem ---- 3) precise fallback: PID listening on this port, check image ------
echo [2/3] Port %PORT% is still busy; resolving the listening PID ...
set "FOREIGN=0"
set "KILLED=0"
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  set "PID=%%P"
  set "IMG="
  for /f "tokens=1 delims=," %%I in ('tasklist /fi "PID eq !PID!" /fo csv /nh 2^>nul') do set "IMG=%%~I"
  if /i "!IMG!"=="engine.exe" (
    echo [3/3] taskkill /pid !PID! ^(engine.exe on port %PORT%^) ...
    taskkill /f /pid !PID! >nul 2>nul
    set "KILLED=1"
  ) else (
    if defined IMG (
      echo [WARN] port %PORT% is held by !IMG! ^(PID !PID!^), not engine.exe; refusing to kill it.
      set "FOREIGN=1"
    )
  )
)

rem wait briefly after the PID kill
for /l %%i in (1,1,15) do (
  ping -n 2 127.0.0.1 >nul 2>nul
  netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
  if errorlevel 1 goto stopped
)

rem ---- 4) only the explicit --all path may touch every engine.exe --------
if "%STOP_ALL%"=="1" (
  echo [--all] port %PORT% is still busy; terminating every engine.exe as explicitly requested ...
  taskkill /f /im engine.exe >nul 2>nul
  for /l %%i in (1,1,15) do (
    ping -n 2 127.0.0.1 >nul 2>nul
    netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
    if errorlevel 1 goto stopped
  )
)

if "%FOREIGN%"=="1" (
  echo.
  echo [ERROR] Port %PORT% is held by a non-engine process; refusing to kill it.
  echo         Close that process manually, or use a different port.
  pause
  exit /b 1
)

echo.
echo [ERROR] Could not stop the engine on port %PORT%.
echo         No unrelated engine.exe process was touched.
echo         If you really want to kill every engine.exe, run: stop.bat --all
pause
exit /b 1

:stopped
echo.
echo Engine stopped; port %PORT% is free.
pause
exit /b 0
