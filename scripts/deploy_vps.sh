#!/bin/bash
# EliBotHFT - Script de Deploy para VPS
# Otimizado para Windows Server / Linux

set -e

echo "============================================"
echo "  EliBotHFT - Deploy Script"
echo "============================================"

# Diretório do projeto
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_DIR

# Verificar Python
echo "[1/6] Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERRO: Python não encontrado!"
    exit 1
fi

echo "Python encontrado: $($PYTHON --version)"

# Criar ambiente virtual
echo "[2/6] Criando ambiente virtual..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
fi

# Ativar ambiente virtual
echo "[3/6] Ativando ambiente virtual..."
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Instalar dependências
echo "[4/6] Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Criar diretórios necessários
echo "[5/6] Criando diretórios..."
mkdir -p data/logs
mkdir -p data/market_data
mkdir -p bin

# Verificar configuração
echo "[6/6] Verificando configuração..."
if [ ! -f "config/risk_limits.json" ]; then
    echo "AVISO: risk_limits.json não encontrado!"
fi

if [ ! -f "config/strategy_params.yaml" ]; then
    echo "AVISO: strategy_params.yaml não encontrado!"
fi

echo ""
echo "============================================"
echo "  Deploy concluído!"
echo "============================================"
echo ""
echo "Para iniciar o bot:"
echo "  $PYTHON main.py"
echo ""
echo "Para iniciar o dashboard:"
echo "  $PYTHON python_control/dashboard_ui.py"
echo ""
