#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" -m unittest discover -s tests -t . -v
