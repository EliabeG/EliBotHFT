#!/usr/bin/env python3
"""
Test de Conexão - Verifica se o robô consegue conectar na FXOpen via WebSocket
"""

import asyncio
import sys
import os
import logging

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.bindings.fxopen_client import FXOpenClient, FXOpenConfig

async def test_connection():
    """Testa conexão com FXOpen via WebSocket"""
    print("=" * 60)
    print("  EliBotHFT - Teste de Conexão WebSocket")
    print("=" * 60)

    # Configuração
    config = FXOpenConfig(
        login='28503781',
        password='rngGNGMW',
        server='ttdemomarginal.fxopen.net',
        token_id='0473113a-f96d-4576-bd1b-507e71ec3d4f',
        token_key='EGqeZPpJQSW2BjCb',
        token_secret='YdafQEND2Fnrc5JGryX6ZPCJ5pf9rmyHnAk6wTDjWGddcRjWtxw369YhKzkBzPkM',
        leverage=500
    )

    print(f"\n[INFO] Servidor: {config.server}")
    print(f"[INFO] Feed WebSocket: {config.feed_url}")
    print(f"[INFO] Trade WebSocket: {config.trade_url}")

    client = FXOpenClient(config)

    try:
        # Tentar conectar
        print("\n[1/4] Conectando aos WebSockets...")
        connected = await client.connect()

        if connected:
            print("[OK] Conexão estabelecida!")

            # Account info
            print("\n[2/4] Obtendo informações da conta...")
            account = client.account
            if account:
                print(f"[OK] Conta: {account.account_id}")
                print(f"     Balance: ${account.balance:,.2f}")
                print(f"     Equity: ${account.equity:,.2f}")
                print(f"     Margem Livre: ${account.free_margin:,.2f}")
                print(f"     Alavancagem: 1:{account.leverage}")
            else:
                print("[WARN] Não foi possível obter info da conta")

            # Subscribe to ticks
            print("\n[3/4] Inscrevendo para receber cotações...")
            symbols = ['EURUSD', 'GBPUSD']
            await client.subscribe_ticks(symbols)

            # Aguardar alguns ticks
            print("[INFO] Aguardando ticks (5 segundos)...")
            await asyncio.sleep(5)

            # Mostrar quotes recebidas
            for symbol in symbols:
                tick = client.quotes.get(symbol)
                if tick:
                    print(f"[OK] {symbol}: Bid={tick.bid:.5f} Ask={tick.ask:.5f} Spread={tick.spread:.5f}")
                else:
                    print(f"[WARN] Não recebeu tick de {symbol}")

            # Posições
            print("\n[4/4] Verificando posições abertas...")
            positions = await client.get_positions()
            if positions:
                print(f"[OK] {len(positions)} posições abertas:")
                for pos in positions:
                    print(f"     {pos.symbol} {pos.side} {pos.volume} lots @ {pos.open_price} P&L: {pos.profit}")
            else:
                print("[OK] Nenhuma posição aberta")

            # Desconectar
            await client.disconnect()
            print("\n[OK] Desconectado com sucesso")

            return True
        else:
            print("[ERRO] Falha ao conectar")
            return False

    except Exception as e:
        print(f"\n[ERRO] Exceção: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_modules():
    """Testa importação dos módulos"""
    print("\n" + "=" * 60)
    print("  Teste de Módulos")
    print("=" * 60)

    modules_ok = True

    try:
        print("\n[1/6] Testando core.network...")
        from src.core.network import NetworkManager, WebSocketClient, TCPClient
        print("[OK] core.network importado")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    try:
        print("[2/6] Testando core.fix_engine...")
        from src.core.fix_engine import FIXEngine, FIXMessage, FIXParser
        print("[OK] core.fix_engine importado")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    try:
        print("[3/6] Testando core.memory...")
        from src.core.memory import MemoryPool, RingBuffer, LockFreeQueue
        print("[OK] core.memory importado")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    try:
        print("[4/6] Testando trading.book...")
        from src.trading.book import OrderBook
        book = OrderBook('EURUSD')
        book.update(0, 1.10050, 100)  # Bid
        book.update(1, 1.10060, 100)  # Ask
        quote = book.get_quote()
        print(f"[OK] OrderBook: bid={quote.bid_price} ask={quote.ask_price} spread={quote.spread:.5f}")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    try:
        print("[5/6] Testando trading.strategy...")
        from src.trading.strategy import StrategyEngine, ScalpingStrategy, MomentumStrategy
        print("[OK] Estratégias importadas")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    try:
        print("[6/6] Testando trading.risk...")
        from src.trading.risk import RiskManager, RiskLimits
        rm = RiskManager()
        rm.update_account(10000, 10000)
        print(f"[OK] RiskManager: trading_allowed={rm.is_trading_allowed}")
    except Exception as e:
        print(f"[ERRO] {e}")
        modules_ok = False

    return modules_ok

async def main():
    """Função principal"""
    # Testar módulos primeiro
    modules_ok = await test_modules()

    if not modules_ok:
        print("\n[ERRO] Alguns módulos falharam. Verifique os erros acima.")
        return 1

    # Testar conexão
    print("\n")
    connection_ok = await test_connection()

    # Resumo
    print("\n" + "=" * 60)
    print("  RESUMO DO TESTE")
    print("=" * 60)
    print(f"  Módulos: {'OK' if modules_ok else 'FALHOU'}")
    print(f"  Conexão: {'OK' if connection_ok else 'FALHOU'}")
    print("=" * 60)

    if modules_ok and connection_ok:
        print("\n[OK] Todos os testes passaram! O robô está pronto para uso.")
        print("\nPara iniciar o bot:")
        print("  python main.py --mode paper")
        return 0
    else:
        print("\n[X] Alguns testes falharam. Verifique os erros acima.")
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
