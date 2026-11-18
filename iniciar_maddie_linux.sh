#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

PYTHON=".venv/bin/python"
"$PYTHON" -m pip install --upgrade -r requirements.txt

if ! "$PYTHON" -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:11434/api/version", timeout=2)' >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
        echo "[Maddie] Iniciando o servidor do Ollama em segundo plano..."
        ollama serve >/dev/null 2>&1 </dev/null &
    else
        echo "[Maddie] Aviso: comando 'ollama' não encontrado. Consulte o README para instalar."
    fi
fi

"$PYTHON" main.py
