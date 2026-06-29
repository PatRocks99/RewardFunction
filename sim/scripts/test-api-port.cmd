@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-api-port.ps1" %*
