@echo off
REM EliBotHFT - Windows 11 Setup Script
REM Execute como Administrador se necessário

echo ============================================
echo   EliBotHFT - Windows 11 Setup
echo ============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Instale Python 3.10+ de https://python.org
    pause
    exit /b 1
)

echo [1/6] Python encontrado
python --version

REM Navegar para diretório do projeto
cd /d "%~dp0\.."

REM Criar ambiente virtual
echo [2/6] Criando ambiente virtual...
if not exist "venv" (
    python -m venv venv
)

REM Ativar ambiente virtual
echo [3/6] Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [4/6] Atualizando pip...
python -m pip install --upgrade pip

REM Instalar dependências
echo [5/6] Instalando dependencias...
pip install -r requirements.txt

REM Criar diretórios
echo [6/6] Criando diretorios...
if not exist "data\logs" mkdir data\logs
if not exist "data\market_data" mkdir data\market_data
if not exist "bin" mkdir bin

echo.
echo ============================================
echo   Setup concluido!
echo ============================================
echo.
echo Para iniciar o bot:
echo   python main.py
echo.
echo Para iniciar o dashboard:
echo   python python_control\dashboard_ui.py
echo.
echo Para testar latencia:
echo   python scripts\latency_test.py
echo.

pause
