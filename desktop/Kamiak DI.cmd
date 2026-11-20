@echo off
cd /d "%~dp0"
py -3 kamiak_di.py
if errorlevel 1 pause
