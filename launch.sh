#!/bin/sh
set -eu

if ! python3 -c 'import sys, tkinter; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    echo 'Install Python 3.10 or newer and its Tkinter package to open the game library.' >&2
    exit 1
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/scripts/launcher.py"
