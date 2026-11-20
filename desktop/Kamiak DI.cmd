@echo off
cd /d "%~dp0"
if exist "shell\node_modules\electron\dist\electron.exe" (
  "shell\node_modules\electron\dist\electron.exe" "shell\main.cjs"
) else (
  echo Install the desktop shell dependencies or use a packaged Kamiak DI app.
  pause
)
