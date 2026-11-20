#!/bin/bash
cd "$(dirname "$0")" || exit 1
for python in python3 /Library/Frameworks/Python.framework/Versions/Current/bin/python3 "$HOME/miniconda3/bin/python3"; do
  if "$python" -c 'import tkinter' >/dev/null 2>&1; then
    exec "$python" kamiak_di.py
  fi
done
echo 'Install Python 3 with Tk support from python.org, then reopen this launcher.'
