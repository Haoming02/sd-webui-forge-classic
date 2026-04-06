@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process python,pythonw -ErrorAction SilentlyContinue | Stop-Process -Force"


echo Stopped python/pythonw processes (if any were running).