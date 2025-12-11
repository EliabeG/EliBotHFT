"""
Replay Engine - Simulador usando dados de mercado capturados
Permite testar estratégias com dados históricos
"""

import os
import sys
import csv
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Dados de tick para replay"""
    timestamp: float
    symbol: str
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class ReplayStats:
    """Estatísticas do replay"""
    ticks_processed: int = 0
    trades_executed: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    replay_speed: float = 1.0


class ReplayEngine:
    """
    Motor de Replay para Backtesting

    Funcionalidades:
    - Replay de dados tick-by-tick
    - Velocidade ajustável
    - Integração com estratégias
    - Métricas de performance
    """

    def __init__(self, data_dir: str = None):
        """
        Inicializa replay engine

        Args:
            data_dir: Diretório com dados de mercado
        """
        self.data_dir = data_dir or 'data/market_data'

        # Estado
        self._running = False
        self._paused = False
        self._speed = 1.0  # 1.0 = tempo real
        self._current_time = 0.0

        # Dados
        self._data: Dict[str, List[TickData]] = {}
        self._data_index: Dict[str, int] = {}

        # Callbacks
        self._on_tick: List[Callable[[TickData], None]] = []
        self._on_trade: List[Callable[[dict], None]] = []

        # Estatísticas
        self._stats = ReplayStats()

        # Simulação de conta
        self._balance = 10000.0
        self._equity = 10000.0
        self._positions: Dict[str, dict] = {}

        logger.info("ReplayEngine inicializado")

    def load_data(self, symbol: str, filepath: str = None) -> int:
        """
        Carrega dados de arquivo

        Args:
            symbol: Símbolo
            filepath: Caminho do arquivo (opcional)

        Returns:
            Número de ticks carregados
        """
        if filepath is None:
            filepath = os.path.join(self.data_dir, f'{symbol}_ticks.csv')

        if not os.path.exists(filepath):
            logger.warning(f"Arquivo não encontrado: {filepath}")
            # Gerar dados sintéticos
            self._data[symbol] = self._generate_synthetic_ticks(symbol, 10000)
            logger.info(f"Dados sintéticos gerados para {symbol}: {len(self._data[symbol])} ticks")
            return len(self._data[symbol])

        ticks = []

        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tick = TickData(
                        timestamp=float(row.get('timestamp', 0)),
                        symbol=symbol,
                        bid=float(row.get('bid', 0)),
                        ask=float(row.get('ask', 0)),
                        bid_volume=float(row.get('bid_volume', 0)),
                        ask_volume=float(row.get('ask_volume', 0))
                    )
                    ticks.append(tick)

            self._data[symbol] = ticks
            self._data_index[symbol] = 0

            logger.info(f"Carregados {len(ticks)} ticks para {symbol}")
            return len(ticks)

        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return 0

    def _generate_synthetic_ticks(self, symbol: str, num_ticks: int) -> List[TickData]:
        """Gera ticks sintéticos para teste"""
        import numpy as np
        np.random.seed(42)

        # Parâmetros por símbolo
        params = {
            'XAUUSD': {'base': 2000.0, 'vol': 0.0003, 'spread': 0.3},
            'EURUSD': {'base': 1.10, 'vol': 0.0001, 'spread': 0.00015},
            'GBPUSD': {'base': 1.27, 'vol': 0.0001, 'spread': 0.00018},
        }

        p = params.get(symbol, {'base': 100.0, 'vol': 0.0002, 'spread': 0.01})

        ticks = []
        price = p['base']
        start_time = time.time() - (num_ticks * 0.1)  # 100ms entre ticks

        for i in range(num_ticks):
            # Random walk
            price *= (1 + np.random.normal(0, p['vol']))

            tick = TickData(
                timestamp=start_time + (i * 0.1),
                symbol=symbol,
                bid=price - p['spread'] / 2,
                ask=price + p['spread'] / 2,
                bid_volume=np.random.exponential(100),
                ask_volume=np.random.exponential(100)
            )
            ticks.append(tick)

        return ticks

    def save_data(self, symbol: str, filepath: str = None) -> bool:
        """
        Salva dados em arquivo

        Args:
            symbol: Símbolo
            filepath: Caminho do arquivo

        Returns:
            True se salvo com sucesso
        """
        if symbol not in self._data:
            logger.warning(f"Nenhum dado para {symbol}")
            return False

        if filepath is None:
            os.makedirs(self.data_dir, exist_ok=True)
            filepath = os.path.join(self.data_dir, f'{symbol}_ticks.csv')

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'bid', 'ask', 'bid_volume', 'ask_volume'])

                for tick in self._data[symbol]:
                    writer.writerow([
                        tick.timestamp, tick.symbol, tick.bid, tick.ask,
                        tick.bid_volume, tick.ask_volume
                    ])

            logger.info(f"Dados salvos em {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
            return False

    def register_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """Registra callback para ticks"""
        self._on_tick.append(callback)

    def register_trade_callback(self, callback: Callable[[dict], None]) -> None:
        """Registra callback para trades"""
        self._on_trade.append(callback)

    def _emit_tick(self, tick: TickData) -> None:
        """Emite tick para callbacks"""
        for cb in self._on_tick:
            try:
                cb(tick)
            except Exception as e:
                logger.error(f"Erro no callback de tick: {e}")

    def _merge_ticks(self) -> Generator[TickData, None, None]:
        """Merge ticks de múltiplos símbolos por timestamp"""
        indices = {s: 0 for s in self._data.keys()}

        while True:
            # Encontrar próximo tick
            next_tick = None
            next_symbol = None

            for symbol, data in self._data.items():
                idx = indices[symbol]
                if idx < len(data):
                    tick = data[idx]
                    if next_tick is None or tick.timestamp < next_tick.timestamp:
                        next_tick = tick
                        next_symbol = symbol

            if next_tick is None:
                break

            indices[next_symbol] += 1
            yield next_tick

    async def run(self, symbols: List[str] = None, speed: float = 1.0) -> ReplayStats:
        """
        Executa replay

        Args:
            symbols: Símbolos a processar (None = todos carregados)
            speed: Velocidade (1.0 = tempo real, 0 = máxima)

        Returns:
            Estatísticas do replay
        """
        if symbols:
            for symbol in symbols:
                if symbol not in self._data:
                    self.load_data(symbol)

        if not self._data:
            logger.error("Nenhum dado carregado")
            return self._stats

        self._running = True
        self._speed = speed
        self._stats = ReplayStats(speed=speed)
        self._stats.start_time = time.time()

        logger.info(f"Replay iniciado: {list(self._data.keys())}, speed={speed}x")

        last_tick_time = None

        for tick in self._merge_ticks():
            if not self._running:
                break

            while self._paused:
                await asyncio.sleep(0.1)

            # Simular delay temporal
            if speed > 0 and last_tick_time is not None:
                delay = (tick.timestamp - last_tick_time) / speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))  # Cap em 1 segundo

            last_tick_time = tick.timestamp
            self._current_time = tick.timestamp

            # Emitir tick
            self._emit_tick(tick)
            self._stats.ticks_processed += 1

            # Atualizar posições
            self._update_positions(tick)

        self._stats.end_time = time.time()
        self._running = False

        logger.info(f"Replay finalizado: {self._stats.ticks_processed} ticks processados")

        return self._stats

    def run_sync(self, symbols: List[str] = None, speed: float = 0) -> ReplayStats:
        """
        Executa replay síncronamente (sem delays)

        Args:
            symbols: Símbolos
            speed: Ignorado (sempre máxima velocidade)

        Returns:
            Estatísticas
        """
        if symbols:
            for symbol in symbols:
                if symbol not in self._data:
                    self.load_data(symbol)

        if not self._data:
            logger.error("Nenhum dado carregado")
            return self._stats

        self._running = True
        self._stats = ReplayStats(speed=0)
        self._stats.start_time = time.time()

        for tick in self._merge_ticks():
            if not self._running:
                break

            self._current_time = tick.timestamp
            self._emit_tick(tick)
            self._stats.ticks_processed += 1
            self._update_positions(tick)

        self._stats.end_time = time.time()
        self._running = False

        return self._stats

    def _update_positions(self, tick: TickData) -> None:
        """Atualiza P&L das posições"""
        for pos_id, pos in list(self._positions.items()):
            if pos['symbol'] != tick.symbol:
                continue

            # Calcular P&L
            if pos['side'] == 'buy':
                pnl = (tick.bid - pos['entry_price']) * pos['volume'] * 100
            else:
                pnl = (pos['entry_price'] - tick.ask) * pos['volume'] * 100

            pos['unrealized_pnl'] = pnl

            # Check stop loss / take profit
            if pos.get('stop_loss'):
                if pos['side'] == 'buy' and tick.bid <= pos['stop_loss']:
                    self._close_position(pos_id, tick.bid, 'stop_loss')
                elif pos['side'] == 'sell' and tick.ask >= pos['stop_loss']:
                    self._close_position(pos_id, tick.ask, 'stop_loss')

            if pos.get('take_profit'):
                if pos['side'] == 'buy' and tick.bid >= pos['take_profit']:
                    self._close_position(pos_id, tick.bid, 'take_profit')
                elif pos['side'] == 'sell' and tick.ask <= pos['take_profit']:
                    self._close_position(pos_id, tick.ask, 'take_profit')

    def open_position(self, symbol: str, side: str, volume: float,
                     price: float, stop_loss: float = None,
                     take_profit: float = None) -> str:
        """
        Abre posição simulada

        Args:
            symbol: Símbolo
            side: 'buy' ou 'sell'
            volume: Volume em lots
            price: Preço de entrada
            stop_loss: Stop loss
            take_profit: Take profit

        Returns:
            ID da posição
        """
        pos_id = f"SIM_{int(time.time()*1000)}"

        self._positions[pos_id] = {
            'id': pos_id,
            'symbol': symbol,
            'side': side,
            'volume': volume,
            'entry_price': price,
            'entry_time': self._current_time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0.0
        }

        logger.debug(f"Posição aberta: {pos_id} {side} {volume} {symbol} @ {price}")
        return pos_id

    def _close_position(self, pos_id: str, price: float, reason: str = 'manual') -> float:
        """Fecha posição e retorna P&L"""
        if pos_id not in self._positions:
            return 0.0

        pos = self._positions.pop(pos_id)

        if pos['side'] == 'buy':
            pnl = (price - pos['entry_price']) * pos['volume'] * 100
        else:
            pnl = (pos['entry_price'] - price) * pos['volume'] * 100

        self._balance += pnl
        self._equity = self._balance
        self._stats.trades_executed += 1
        self._stats.total_pnl += pnl

        # Emit trade
        trade = {
            'position_id': pos_id,
            'symbol': pos['symbol'],
            'side': pos['side'],
            'volume': pos['volume'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }

        for cb in self._on_trade:
            try:
                cb(trade)
            except Exception as e:
                logger.error(f"Erro no callback de trade: {e}")

        logger.debug(f"Posição fechada: {pos_id}, PnL: {pnl:.2f}, Reason: {reason}")
        return pnl

    def close_position(self, pos_id: str, price: float = None) -> float:
        """Fecha posição manualmente"""
        if pos_id not in self._positions:
            return 0.0

        pos = self._positions[pos_id]
        if price is None:
            # Usar último preço conhecido
            if pos['symbol'] in self._data and self._data_index.get(pos['symbol'], 0) > 0:
                idx = self._data_index[pos['symbol']] - 1
                tick = self._data[pos['symbol']][idx]
                price = tick.bid if pos['side'] == 'buy' else tick.ask
            else:
                price = pos['entry_price']

        return self._close_position(pos_id, price, 'manual')

    def stop(self) -> None:
        """Para o replay"""
        self._running = False

    def pause(self) -> None:
        """Pausa o replay"""
        self._paused = True

    def resume(self) -> None:
        """Retoma o replay"""
        self._paused = False

    def set_speed(self, speed: float) -> None:
        """Define velocidade do replay"""
        self._speed = speed

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def stats(self) -> ReplayStats:
        return self._stats

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def positions(self) -> Dict[str, dict]:
        return self._positions


def main():
    """Executa replay de demonstração"""
    print("EliBotHFT - Replay Engine")
    print("=" * 40)

    engine = ReplayEngine()

    # Carregar dados (ou gerar sintéticos)
    engine.load_data('XAUUSD')
    engine.load_data('EURUSD')

    # Contador de ticks
    tick_count = [0]
    trades = []

    def on_tick(tick):
        tick_count[0] += 1
        if tick_count[0] % 1000 == 0:
            print(f"Processados {tick_count[0]} ticks, último: {tick.symbol} @ {tick.mid:.5f}")

    def on_trade(trade):
        trades.append(trade)
        print(f"Trade: {trade['side']} {trade['symbol']} PnL: {trade['pnl']:.2f}")

    engine.register_tick_callback(on_tick)
    engine.register_trade_callback(on_trade)

    # Executar replay síncrono
    print("\nIniciando replay...")
    stats = engine.run_sync(['XAUUSD', 'EURUSD'])

    # Resultados
    duration = stats.end_time - stats.start_time
    print(f"\n{'=' * 40}")
    print("RESULTADOS DO REPLAY")
    print(f"{'=' * 40}")
    print(f"Ticks processados: {stats.ticks_processed}")
    print(f"Trades executados: {stats.trades_executed}")
    print(f"P&L Total: ${stats.total_pnl:.2f}")
    print(f"Duração: {duration:.2f}s")
    print(f"Velocidade: {stats.ticks_processed / duration:.0f} ticks/s")


if __name__ == '__main__':
    main()
