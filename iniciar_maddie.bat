@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [Maddie] Python nao encontrado no PATH. Instale o Python 3.10+ em https://www.python.org/downloads/
    pause
    exit /b 1
)

where ollama >nul 2>nul
if errorlevel 1 (
    echo [Maddie] Aviso: comando 'ollama' nao encontrado. Veja o README para instalar.
)

rem Se a API local nao responde, tenta subir o servidor do Ollama em segundo plano.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:11434/api/version | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    where ollama >nul 2>nul && (
        echo [Maddie] Iniciando o servidor do Ollama...
        start "" /min ollama serve
    )
)

rem pythonw: abre a janela sem console. Para ver erros no terminal, use: python main.py
start "" pythonw "%~dp0main.py"
endlocal
