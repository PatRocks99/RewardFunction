@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0get-wsl-ip.ps1" %*
