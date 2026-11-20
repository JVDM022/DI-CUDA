#!/bin/bash
# Finder supplies a minimal PATH; prefer a current Python/Tk installation.
set -u
cd "$(dirname "$0")" || exit 1
unset PYTHONHOME PYTHONPATH
export PATH="/usr/local/bin:/opt/homebrew/bin:/opt/local/bin:$PATH"
log="${TMPDIR:-/tmp}/Kamiak-DI-launch.log"
exec >>"$log" 2>&1
printf '\nKamiak DI launch: %s\n' "$(date)"
native="../dist/native/Kamiak DI-darwin-arm64/Kamiak DI.app/Contents/MacOS/Kamiak DI"
if [[ -x "$native" ]]; then
  exec "$native"
fi
electron="shell/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
if [[ -x "$electron" ]]; then
  exec "$electron" "$PWD/shell/main.cjs"
fi
printf 'Native GUI runtime not found. Use the packaged Kamiak DI app, or install desktop/shell dependencies.\n'
exit 1
