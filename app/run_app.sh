#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m streamlit run app/indexing_app.py --server.address 127.0.0.1 "$@"
