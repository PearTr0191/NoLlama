@echo off
rem ===========================================================================
rem  install-windows.bat -- NoLlama setup for people who just downloaded a ZIP
rem
rem  Double-click this file. It checks for the two things the real installer
rem  (install.ps1) needs -- PowerShell 7 and Python 3.10+ -- offers to install
rem  whatever is missing via winget, then hands off to install.ps1.
rem
rem  Why this file exists: Windows 10/11 ship PowerShell 5.1, which cannot run
rem  install.ps1 (#requires -Version 7.0), and scripts extracted from a
rem  downloaded ZIP are blocked by execution policy. A .bat has neither
rem  problem, so this is the reliable first double-click.
rem ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title NoLlama install
echo.
echo  === NoLlama install (Windows) ===
echo.

rem --- 1. Find PowerShell 7 (pwsh.exe) ------------------------------------
set "PWSH="
where pwsh >nul 2>nul && set "PWSH=pwsh"
if not defined PWSH if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "PWSH=%ProgramFiles%\PowerShell\7\pwsh.exe"

if not defined PWSH (
    echo  PowerShell 7 is not installed. Windows ships PowerShell 5.1, which is
    echo  too old for NoLlama's installer.
    echo.
    echo  The command to install it ^(this is all this script will run^):
    echo.
    echo      winget install --id Microsoft.PowerShell --source winget
    echo.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo  ...but 'winget' was not found either. Two options:
        echo    a^) Install "App Installer" from the Microsoft Store: https://aka.ms/getwinget
        echo    b^) Download PowerShell 7 directly: https://aka.ms/powershell-release?tag=stable
        echo  Then double-click this file again.
        goto :fail
    )
    choice /C YN /M "  Run it now"
    if errorlevel 2 goto :fail
    winget install --id Microsoft.PowerShell --source winget
    if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" (
        set "PWSH=%ProgramFiles%\PowerShell\7\pwsh.exe"
    ) else (
        echo.
        echo  Installed, but this window's PATH is stale. Close this window and
        echo  double-click install-windows.bat again.
        goto :fail
    )
)
echo  [+] PowerShell 7 found.

rem --- 2. Find Python 3.10+ ------------------------------------------------
rem 'where python' is not enough: Windows ships a Store stub called python.exe
rem that opens the Microsoft Store instead of running. Actually execute it.
set "PYOK="
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "PYOK=1"
if not defined PYOK py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "PYOK=1"

if not defined PYOK (
    echo  Python 3.10+ is not installed ^(or only the Microsoft Store stub is^).
    echo.
    echo  The command to install it:
    echo.
    echo      winget install -e --id Python.Python.3.12
    echo.
    choice /C YN /M "  Run it now"
    if errorlevel 2 goto :fail
    winget install -e --id Python.Python.3.12
    echo.
    echo  Python installed. The PATH of this window is stale -- close this
    echo  window and double-click install-windows.bat again to continue.
    pause
    exit /b 0
)
echo  [+] Python 3.10+ found.

rem --- 3. Hand off to the real installer ------------------------------------
echo.
echo  Handing off to install.ps1 ^(device detection + model menu^)...
echo.
"%PWSH%" -NoLogo -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 goto :fail

rem --- 4. Offer to start the server -----------------------------------------
if exist "%~dp0start.ps1" (
    echo.
    choice /C YN /M "  Install complete. Start NoLlama now"
    if errorlevel 2 goto :done
    "%PWSH%" -NoLogo -ExecutionPolicy Bypass -File "%~dp0start.ps1"
)

:done
echo.
echo  Later, start NoLlama with:  start.ps1  ^(right-click, Run with PowerShell 7^)
pause
exit /b 0

:fail
echo.
pause
exit /b 1
