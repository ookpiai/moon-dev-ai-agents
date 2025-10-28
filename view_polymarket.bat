@echo off
REM View Polymarket System in 3 Windows

echo Starting Polymarket monitoring windows...

REM Window 1 - Data Collector
start "Polymarket Data Collector" cmd /k "cd /d %~dp0 && powershell Get-Content src\data\polymarket\logs\data_collector.log -Wait -Tail 50"

REM Window 2 - Scanner (check every 30 seconds since it updates slowly)
start "Polymarket Scanner" cmd /k "cd /d %~dp0 && echo Checking scanner status every 30 seconds... && :loop && type polymarket_scanner.log 2>nul && timeout /t 30 /nobreak >nul && goto loop"

REM Window 3 - Dashboard (update every 60 seconds)
start "Polymarket Dashboard" cmd /k "cd /d %~dp0 && :loop && cls && python polymarket_dashboard.py && timeout /t 60 /nobreak >nul && goto loop"

echo.
echo Three monitoring windows opened:
echo   1. Data Collector (live log)
echo   2. Scanner Status (checks every 30s)
echo   3. Dashboard (updates every 60s)
echo.
echo Press any key to exit this window (monitoring will continue)...
pause >nul
