#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  py="$PWD/.venv/bin/python"
elif [[ -x ../.venv-kikuchi/bin/python ]]; then
  py="$PWD/../.venv-kikuchi/bin/python"
else
  echo 'Install the app environment first. See app/README.md.'
  read -r -p 'Press Enter to close.'
  exit 1
fi
export PATH="$(dirname "$py"):$PATH"
exec "$py" -m streamlit run app/indexing_app.py --server.address 127.0.0.1
