@echo off
rem ===========================================================================
rem  Stop the engine started by the launcher bat (or by `engine.exe serve`).
rem
rem  Order of attack:
rem    1. POST /shutdown on the port (clean: frees VRAM and exits the process)
rem    2. wait up to ~15 s for the port to be released
rem    3. fall back to taskkill engine.exe
rem  Pure ASCII output (the engine may not be running, so nothing is localized).
rem ===========================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>nul
title engine stop
cd /d "%~dp0"

set "PORT=7030"
set "HOST=127.0.0.1"

echo Stopping the engine on %HOST%:%PORT% ...

netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if errorlevel 1 (
  echo Nothing is listening on port %PORT%.
  echo If an engine is still running on another port, close its window or run:
  echo   taskkill /f /im engine.exe
  echo.
  pause
  exit /b 0
)

curl -s -m 3 -X POST "http://%HOST%:%PORT%/shutdown" >nul 2>nul
if errorlevel 1 (
  echo [1/2] HTTP /shutdown request failed ^(curl missing or service busy^).
) else (
  echo [1/2] Shutdown request sent.
)

for /l %%i in (1,1,15) do (
  ping -n 2 127.0.0.1 >nul 2>nul
  netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
  if errorlevel 1 goto stopped
)

echo [2/2] Port %PORT% is still busy; terminating engine.exe ...
taskkill /f /im engine.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Could not stop the engine. Close its console window manually.
  pause
  exit /b 1
)
echo engine.exe terminated. Port %PORT% should be free now.
pause
exit /b 0

:stopped
echo Engine stopped; port %PORT% is free.
pause
exit /b 0
