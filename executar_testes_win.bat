@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo === 1. Testes deterministicos (nao precisam do Ollama) ===
python -m unittest discover -s tests -t . -v
echo.
echo === 2. Teste real de fumaca com o Ollama ===
set MADDIE_TEST_REAL=1
python -m unittest tests.test_real_ollama -v
set MADDIE_TEST_REAL=
echo.
echo === 3. Avaliacao conversacional (probabilistica, cerca de 1 hora no modelo de 4B) ===
choice /m "Rodar a bateria A-J agora"
if errorlevel 2 goto fim
python avaliacao\avaliar.py
:fim
pause
