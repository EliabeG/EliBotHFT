# EliBotHFT - High Frequency Trading Robot for Forex

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Robô de trading de alta frequência (HFT) desenvolvido para operar no mercado Forex através da plataforma FXOpen TickTrader.

## Características

- **Motor de Baixa Latência**: Otimizado para execução rápida de ordens
- **Múltiplas Estratégias**: Scalping, Momentum, Mean Reversion
- **Gestão de Risco**: Controle de drawdown, position sizing, circuit breaker
- **Order Book L2**: Reconstrução em tempo real do livro de ofertas
- **Dashboard em Tempo Real**: Monitoramento de performance
- **Otimização de Parâmetros**: Grid search e random search
- **Replay de Dados**: Backtesting com dados históricos

## Estrutura do Projeto

```
EliBotHFT/
├── config/                     # Arquivos de configuração
│   ├── fxopen_fix.cfg         # Config FIX
│   ├── risk_limits.json       # Limites de risco
│   ├── instruments.csv        # Mapeamento de instrumentos
│   └── strategy_params.yaml   # Parâmetros de estratégia
├── data/
│   ├── logs/                  # Logs de execução
│   └── market_data/           # Dados de mercado
├── src/
│   ├── core/                  # Motor de baixa latência
│   │   ├── network/           # TCP/WebSocket
│   │   ├── fix_engine/        # Parser FIX
│   │   ├── memory/            # Memory pools
│   │   └── logger/            # Logging assíncrono
│   ├── trading/               # Lógica de mercado
│   │   ├── book/              # Order Book
│   │   ├── strategy/          # Estratégias
│   │   ├── oms/               # Order Management
│   │   └── risk/              # Gestão de risco
│   └── bindings/              # API Python e cliente FXOpen
├── python_control/            # Scripts de controle
│   ├── dashboard_ui.py        # Dashboard
│   ├── param_optimizer.py     # Otimizador
│   └── replay_engine.py       # Replay
├── scripts/                   # Scripts de automação
├── main.py                    # Ponto de entrada
└── requirements.txt           # Dependências
```

## Requisitos

- Python 3.10 ou superior
- Windows 11 (otimizado) ou Linux
- Conta FXOpen TickTrader

## Instalação

### Windows

```batch
# Clone o repositório
git clone https://github.com/seu-usuario/EliBotHFT.git
cd EliBotHFT

# Execute o setup
scripts\setup_windows.bat
```

### Linux/Mac

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/EliBotHFT.git
cd EliBotHFT

# Execute o setup
chmod +x scripts/deploy_vps.sh
./scripts/deploy_vps.sh
```

### Manual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Configuração

### 1. Credenciais FXOpen

Edite as credenciais em `main.py` ou crie um arquivo `.env`:

```env
FXOPEN_LOGIN=seu_login
FXOPEN_PASSWORD=sua_senha
FXOPEN_TOKEN_ID=seu_token_id
FXOPEN_TOKEN_KEY=sua_token_key
FXOPEN_TOKEN_SECRET=seu_token_secret
```

### 2. Parâmetros de Estratégia

Edite `config/strategy_params.yaml` para ajustar:
- Símbolos operados
- Parâmetros de entrada/saída
- Timeframes
- Filtros de mercado

### 3. Limites de Risco

Edite `config/risk_limits.json` para definir:
- Perda máxima diária
- Tamanho máximo de posição
- Drawdown máximo
- Circuit breaker

## Uso

### Iniciar o Bot

```bash
# Modo paper (demo)
python main.py --mode paper

# Modo live
python main.py --mode live

# Com dashboard
python main.py --dashboard

# Símbolos específicos
python main.py --symbols XAUUSD EURUSD GBPUSD
```

### Dashboard Standalone

```bash
python python_control/dashboard_ui.py
```

### Teste de Latência

```bash
python scripts/latency_test.py
```

### Otimização de Parâmetros

```bash
python python_control/param_optimizer.py
```

### Replay de Dados

```bash
python python_control/replay_engine.py
```

## Estratégias Implementadas

### 1. Scalping Strategy
- Opera em micro movimentos de preço
- Usa imbalance do order book
- Tempo de holding muito curto

### 2. Momentum Strategy
- Detecta movimentos direcionais fortes
- Múltiplos timeframes de confirmação
- Trailing stop dinâmico

## Gestão de Risco

O sistema inclui múltiplas camadas de proteção:

1. **Pre-Trade Risk Check**: Verifica cada ordem antes de enviar
2. **Position Limits**: Controla tamanho e número de posições
3. **Daily Loss Limit**: Para trading ao atingir perda máxima
4. **Circuit Breaker**: Pausa após sequência de perdas
5. **Rate Limiting**: Controla frequência de ordens

## Métricas Monitoradas

- Sharpe Ratio
- Profit Factor
- Win Rate
- Maximum Drawdown
- Average Trade Duration
- Latência de execução

## Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Add NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## Aviso Legal

**IMPORTANTE**: Este software é fornecido apenas para fins educacionais. Trading de alta frequência envolve riscos significativos. Não invista dinheiro que você não pode perder. O autor não se responsabiliza por perdas financeiras resultantes do uso deste software.

## Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

Desenvolvido com Python para FXOpen TickTrader Platform
