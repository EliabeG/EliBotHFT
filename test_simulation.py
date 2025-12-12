#!/usr/bin/env python3
"""
Test de Simulação - Testa o robô em modo simulação sem conexão real
"""

import asyncio
import sys
import os
import time
import random

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.trading.book import OrderBook, Quote, BookSide
from src.trading.strategy import StrategyEngine, ScalpingStrategy, MomentumStrategy, Signal, SignalType
from src.trading.oms import OrderManagementSystem, Order, Side, OrderType
from src.trading.risk import RiskManager, RiskLimits
from src.core.logger import LatencyLogger

def generate_tick(symbol: str, base_price: float, volatility: float = 0.0001):
    """Gera tick simulado"""
    change = random.gauss(0, volatility) * base_price
    mid = base_price + change
    spread = base_price * 0.00015  # 1.5 pips

    return {
        'symbol': symbol,
        'bid': mid - spread/2,
        'ask': mid + spread/2,
        'timestamp': time.time()
    }

async def simulate_trading():
    """Simula sessão de trading"""
    print("=" * 60)
    print("  EliBotHFT - Simulação de Trading")
    print("=" * 60)

    # Inicializar componentes
    print("\n[INIT] Inicializando componentes...")

    # Order Books
    books = {
        'XAUUSD': OrderBook('XAUUSD', tick_size=0.01),
        'EURUSD': OrderBook('EURUSD', tick_size=0.00001)
    }
    print("[OK] Order Books criados")

    # Strategy Engine
    strategy_engine = StrategyEngine()
    print("[OK] Strategy Engine iniciado")

    # OMS
    oms = OrderManagementSystem()
    print("[OK] OMS iniciado")

    # Risk Manager
    risk_manager = RiskManager()
    risk_manager.update_account(10000, 10000)  # $10,000 balance
    print("[OK] Risk Manager iniciado")

    # Latency Logger
    latency_logger = LatencyLogger()

    # Estatísticas
    signals_generated = 0
    orders_created = 0
    ticks_processed = 0

    # Preços base
    prices = {
        'XAUUSD': 2000.0,
        'EURUSD': 1.1000
    }

    # Callback para sinais
    def on_signal(signal: Signal):
        nonlocal signals_generated, orders_created
        signals_generated += 1

        print(f"\n[SIGNAL] {signal.signal_type.value.upper()} {signal.symbol} @ {signal.price:.5f}")
        print(f"         Confidence: {signal.confidence:.1%}, Strategy: {signal.strategy_name}")

        # Verificar risco
        order = Order(
            symbol=signal.symbol,
            side=Side.BUY if signal.signal_type == SignalType.BUY else Side.SELL,
            quantity=0.01,
            order_type=OrderType.MARKET
        )

        risk_check = risk_manager.check_pre_trade(order)
        if risk_check:
            print(f"         Risk Check: PASSED")
            orders_created += 1
        else:
            print(f"         Risk Check: BLOCKED - {risk_check.message}")

    strategy_engine.register_signal_callback(on_signal)

    # Simular ticks
    print("\n[SIM] Iniciando simulação (500 ticks)...")
    print("-" * 60)

    start_time = time.perf_counter()

    for i in range(500):
        # Escolher símbolo aleatório
        symbol = random.choice(['XAUUSD', 'EURUSD'])

        # Gerar tick
        tick_start = latency_logger.start_timer()

        # Atualizar preço com random walk
        prices[symbol] *= (1 + random.gauss(0, 0.0001))
        tick_data = generate_tick(symbol, prices[symbol])

        # Atualizar order book
        book = books[symbol]
        book.update(BookSide.BID, tick_data['bid'], random.uniform(10, 100))
        book.update(BookSide.ASK, tick_data['ask'], random.uniform(10, 100))

        # Criar quote
        quote = book.get_quote()

        # Processar no strategy engine
        strategy_engine.on_tick(symbol, quote, book)

        latency_logger.stop_timer(tick_start, 'tick_processing')
        ticks_processed += 1

        # Progresso a cada 100 ticks
        if (i + 1) % 100 == 0:
            print(f"[PROGRESS] {i+1}/500 ticks processados, {signals_generated} sinais gerados")

    elapsed = time.perf_counter() - start_time

    # Resultados
    print("\n" + "=" * 60)
    print("  RESULTADOS DA SIMULAÇÃO")
    print("=" * 60)

    print(f"\n[PERFORMANCE]")
    print(f"  Ticks processados: {ticks_processed}")
    print(f"  Tempo total: {elapsed:.3f}s")
    print(f"  Throughput: {ticks_processed/elapsed:.0f} ticks/segundo")

    # Latência
    lat_stats = latency_logger.get_stats('tick_processing')
    if lat_stats:
        print(f"\n[LATÊNCIA]")
        print(f"  Média: {lat_stats.avg_ns/1000:.2f} µs")
        print(f"  Min: {lat_stats.min_ns/1000:.2f} µs")
        print(f"  Max: {lat_stats.max_ns/1000:.2f} µs")
        print(f"  P95: {lat_stats.p95_ns/1000:.2f} µs")
        print(f"  P99: {lat_stats.p99_ns/1000:.2f} µs")

    print(f"\n[TRADING]")
    print(f"  Sinais gerados: {signals_generated}")
    print(f"  Ordens aprovadas: {orders_created}")

    # Strategy stats
    print(f"\n[ESTRATÉGIAS]")
    stats = strategy_engine.get_aggregate_stats()
    for name, s in stats.get('strategies', {}).items():
        print(f"  {name}:")
        print(f"    Sinais: {s.get('signals_generated', 0)}")
        print(f"    Trades: {s.get('trades_executed', 0)}")

    # Risk stats
    print(f"\n[RISCO]")
    risk_stats = risk_manager.stats
    print(f"  Trading permitido: {risk_stats['trading_allowed']}")
    print(f"  Circuit Breaker: {'ATIVO' if risk_stats['circuit_breaker_active'] else 'Inativo'}")

    # Order Book final
    print(f"\n[ORDER BOOKS FINAIS]")
    for symbol, book in books.items():
        quote = book.get_quote()
        print(f"  {symbol}: Bid={quote.bid_price:.5f} Ask={quote.ask_price:.5f} "
              f"Spread={quote.spread_bps:.1f}bps")

    print("\n" + "=" * 60)
    print("  ✓ Simulação concluída com sucesso!")
    print("=" * 60)

    return True

async def main():
    """Função principal"""
    try:
        success = await simulate_trading()
        return 0 if success else 1
    except Exception as e:
        print(f"\n[ERRO] Exceção durante simulação: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
