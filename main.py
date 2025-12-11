#!/usr/bin/env python3
"""
EliBotHFT - High Frequency Trading Robot for Forex
Ponto de entrada principal do robô de trading

Desenvolvido para FXOpen TickTrader Platform
Compatível com Windows 11

Uso:
    python main.py [--mode paper|live] [--config path] [--dashboard]

Argumentos:
    --mode      Modo de operação: 'paper' (demo) ou 'live'
    --config    Caminho para arquivo de configuração
    --dashboard Iniciar com dashboard
    --help      Mostrar ajuda
"""

import asyncio
import argparse
import sys
import os
import signal
import logging
from pathlib import Path
from datetime import datetime

# Configurar path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Criar diretórios necessários ANTES de configurar logging
(PROJECT_ROOT / 'data' / 'logs').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'data' / 'market_data').mkdir(parents=True, exist_ok=True)

# Importar componentes
from src.bindings import EliBotAPI, FXOpenConfig
from src.bindings.elibot_api import BotConfig, BotState

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            PROJECT_ROOT / 'data' / 'logs' / f'elibot_{datetime.now().strftime("%Y%m%d")}.log'
        )
    ]
)

logger = logging.getLogger('EliBotHFT')


def print_banner():
    """Imprime banner do bot"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗██╗     ██╗██████╗  ██████╗ ████████╗           ║
    ║   ██╔════╝██║     ██║██╔══██╗██╔═══██╗╚══██╔══╝           ║
    ║   █████╗  ██║     ██║██████╔╝██║   ██║   ██║              ║
    ║   ██╔══╝  ██║     ██║██╔══██╗██║   ██║   ██║              ║
    ║   ███████╗███████╗██║██████╔╝╚██████╔╝   ██║              ║
    ║   ╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝    ╚═╝              ║
    ║                                                           ║
    ║            High Frequency Trading Robot                   ║
    ║                  for Forex (FXOpen)                       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='EliBotHFT - High Frequency Trading Robot'
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live'],
        default='paper',
        help='Modo de operação (default: paper)'
    )

    parser.add_argument(
        '--config', '-c',
        default='config/strategy_params.yaml',
        help='Arquivo de configuração'
    )

    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Iniciar com dashboard'
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['XAUUSD', 'EURUSD'],
        help='Símbolos para operar'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Executar sem conectar (teste)'
    )

    return parser.parse_args()


async def run_with_dashboard(bot: EliBotAPI):
    """Executa bot com dashboard"""
    from python_control.dashboard_ui import Dashboard

    dashboard = Dashboard()
    dashboard.set_bot_api(bot)

    # Iniciar bot
    if not await bot.start():
        logger.error("Falha ao iniciar bot")
        return

    # Executar dashboard
    try:
        await dashboard.run()
    finally:
        await bot.stop()


async def run_headless(bot: EliBotAPI):
    """Executa bot sem interface"""
    # Handler para shutdown gracioso
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Sinal de shutdown recebido")
        shutdown_event.set()

    # Registrar handlers de sinal
    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows não suporta add_signal_handler
        pass

    # Iniciar bot
    if not await bot.start():
        logger.error("Falha ao iniciar bot")
        return

    logger.info("Bot rodando. Pressione Ctrl+C para parar.")

    # Aguardar shutdown
    try:
        while not shutdown_event.is_set() and bot.is_running:
            await asyncio.sleep(1)

            # Log periódico de status
            if int(bot.uptime) % 60 == 0 and int(bot.uptime) > 0:
                stats = bot.stats
                logger.info(f"Uptime: {bot.uptime:.0f}s | "
                          f"Trades: {stats.get('oms', {}).get('orders_filled', 0)}")

    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


def main():
    """Função principal"""
    print_banner()

    args = parse_args()

    logger.info(f"Iniciando EliBotHFT")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Símbolos: {args.symbols}")
    logger.info(f"Config: {args.config}")

    # Configurar FXOpen
    fxopen_config = FXOpenConfig(
        login='28503781',
        password='rngGNGMW',
        server='ttdemomarginal.fxopen.net',
        token_id='0473113a-f96d-4576-bd1b-507e71ec3d4f',
        token_key='EGqeZPpJQSW2BjCb',
        token_secret='YdafQEND2Fnrc5JGryX6ZPCJ5pf9rmyHnAk6wTDjWGddcRjWtxw369YhKzkBzPkM',
        leverage=500
    )

    # Configurar bot
    bot_config = BotConfig(
        config_dir=str(PROJECT_ROOT / 'config'),
        data_dir=str(PROJECT_ROOT / 'data'),
        log_dir=str(PROJECT_ROOT / 'data' / 'logs'),
        symbols=args.symbols,
        mode=args.mode,
        enabled=True,
        fxopen=fxopen_config
    )

    # Dry run - apenas testar configuração
    if args.dry_run:
        logger.info("Dry run - configuração válida")
        logger.info(f"Bot config: {bot_config}")
        return

    # Criar bot
    bot = EliBotAPI(config=bot_config)

    # Callbacks de exemplo
    def on_state_change(state):
        logger.info(f"Estado do bot: {state}")

    def on_signal(signal):
        logger.info(f"Sinal: {signal.signal_type.value} {signal.symbol} @ {signal.price}")

    bot.register_state_callback(on_state_change)
    bot.register_signal_callback(on_signal)

    # Executar
    try:
        if args.dashboard:
            asyncio.run(run_with_dashboard(bot))
        else:
            asyncio.run(run_headless(bot))

    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)

    logger.info("EliBotHFT finalizado")


if __name__ == '__main__':
    main()
