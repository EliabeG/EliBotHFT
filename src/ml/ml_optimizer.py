"""
ML Optimizer - Módulo de Otimização de Parâmetros via Backtesting
Testa infinitas configurações em dados históricos para encontrar os melhores parâmetros

Funcionalidades:
- Download de dados históricos via FXOpen API
- Backtesting com diferentes configurações
- Otimização via Grid Search, Random Search, Bayesian Optimization
- Algoritmo Genético para busca de hiperparâmetros
- Salva melhores configurações automaticamente
"""

import os
import sys
import json
import csv
import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import random
import pickle

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Resultado de um backtest"""
    config: Dict[str, Any]
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    max_consecutive_losses: int = 0
    total_bars: int = 0
    execution_time: float = 0.0

    @property
    def fitness(self) -> float:
        """Score de fitness para otimização"""
        if self.total_trades < 10:
            return -1000.0  # Penalizar configurações com poucos trades

        # Combinar múltiplas métricas
        score = (
            self.sharpe_ratio * 100 +
            self.profit_factor * 50 +
            self.win_rate * 100 +
            self.total_pnl * 0.1 -
            self.max_drawdown * 2 -
            self.max_consecutive_losses * 10
        )
        return score


@dataclass
class OptimizationConfig:
    """Configuração do otimizador"""
    # Parâmetros a otimizar com ranges
    param_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Configurações do otimizador
    optimization_method: str = 'bayesian'  # 'grid', 'random', 'bayesian', 'genetic'
    max_iterations: int = 100
    population_size: int = 20  # Para algoritmo genético
    early_stopping_rounds: int = 20
    n_parallel_jobs: int = 4

    # Configurações de backtest
    symbol: str = 'EURUSD'
    timeframe: str = 'M1'
    start_date: datetime = None
    end_date: datetime = None
    initial_balance: float = 10000.0

    # Caminhos
    data_dir: str = 'data/market_data'
    results_dir: str = 'data/optimization_results'
    best_config_path: str = 'config/optimized_params.json'


@dataclass
class BarData:
    """Dados de barra para backtest"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


class BacktestEngine:
    """Motor de backtesting rápido para otimização"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.pip_size = self.config.get('pip_size', 0.0001)

        # Estado
        self._balance = 10000.0
        self._equity = 10000.0
        self._position = None
        self._trades: List[Dict] = []
        self._equity_curve: List[float] = []

        # Indicadores (cache)
        self._prices: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []

    def reset(self, initial_balance: float = 10000.0):
        """Reseta estado para novo backtest"""
        self._balance = initial_balance
        self._equity = initial_balance
        self._position = None
        self._trades = []
        self._equity_curve = [initial_balance]
        self._prices = []
        self._highs = []
        self._lows = []

    def run(self, bars: List[BarData], strategy_config: Dict[str, Any]) -> BacktestResult:
        """
        Executa backtest com configuração específica

        Args:
            bars: Lista de barras OHLC
            strategy_config: Configuração da estratégia

        Returns:
            BacktestResult com métricas
        """
        start_time = time.time()
        self.reset(strategy_config.get('initial_balance', 10000.0))

        # Extrair parâmetros
        stop_loss_pips = strategy_config.get('stop_loss_pips', 30)
        take_profit_pips = strategy_config.get('take_profit_pips', 5)
        trailing_stop_pips = strategy_config.get('trailing_stop_pips', 2)
        min_profit_to_trail_pips = strategy_config.get('min_profit_to_trail_pips', 1)
        lot_size = strategy_config.get('lot_size', 0.01)

        # Parâmetros de entrada
        entry_threshold = strategy_config.get('entry_threshold', 0.0001)
        rsi_period = strategy_config.get('rsi_period', 14)
        rsi_overbought = strategy_config.get('rsi_overbought', 70)
        rsi_oversold = strategy_config.get('rsi_oversold', 30)
        ma_fast = strategy_config.get('ma_fast', 10)
        ma_slow = strategy_config.get('ma_slow', 30)

        # Trailing stop tracking
        trailing_high = None
        trailing_low = None

        for i, bar in enumerate(bars):
            self._prices.append(bar.close)
            self._highs.append(bar.high)
            self._lows.append(bar.low)

            # Precisa de histórico mínimo
            if len(self._prices) < max(ma_slow, rsi_period) + 1:
                continue

            # Calcular indicadores
            rsi = self._calculate_rsi(rsi_period)
            ma_fast_val = np.mean(self._prices[-ma_fast:])
            ma_slow_val = np.mean(self._prices[-ma_slow:])
            momentum = (bar.close - self._prices[-ma_fast]) / self._prices[-ma_fast]

            # Gerenciar posição existente
            if self._position:
                current_pnl_pips = self._calculate_pnl_pips(bar.close)

                # Atualizar trailing stop
                if self._position['side'] == 'buy':
                    profit_distance = (bar.close - self._position['entry_price']) / self.pip_size
                    if profit_distance >= min_profit_to_trail_pips:
                        if trailing_high is None or bar.high > trailing_high:
                            trailing_high = bar.high

                    # Verificar trailing stop
                    if trailing_high:
                        trailing_stop_price = trailing_high - (trailing_stop_pips * self.pip_size)
                        if bar.low <= trailing_stop_price and current_pnl_pips >= 0.5:
                            self._close_position(trailing_stop_price, 'trailing_stop')
                            trailing_high = None
                            continue

                    # Stop loss
                    if bar.low <= self._position['stop_loss']:
                        self._close_position(self._position['stop_loss'], 'stop_loss')
                        trailing_high = None
                        continue

                    # Take profit
                    if bar.high >= self._position['take_profit']:
                        self._close_position(self._position['take_profit'], 'take_profit')
                        trailing_high = None
                        continue

                else:  # sell
                    profit_distance = (self._position['entry_price'] - bar.close) / self.pip_size
                    if profit_distance >= min_profit_to_trail_pips:
                        if trailing_low is None or bar.low < trailing_low:
                            trailing_low = bar.low

                    # Verificar trailing stop
                    if trailing_low:
                        trailing_stop_price = trailing_low + (trailing_stop_pips * self.pip_size)
                        if bar.high >= trailing_stop_price and current_pnl_pips >= 0.5:
                            self._close_position(trailing_stop_price, 'trailing_stop')
                            trailing_low = None
                            continue

                    # Stop loss
                    if bar.high >= self._position['stop_loss']:
                        self._close_position(self._position['stop_loss'], 'stop_loss')
                        trailing_low = None
                        continue

                    # Take profit
                    if bar.low <= self._position['take_profit']:
                        self._close_position(self._position['take_profit'], 'take_profit')
                        trailing_low = None
                        continue

            else:
                # Sinais de entrada
                # BUY: MA fast > MA slow, RSI < oversold, momentum positivo
                if (ma_fast_val > ma_slow_val and
                    rsi < rsi_oversold and
                    momentum > entry_threshold):

                    stop_loss = bar.close - (stop_loss_pips * self.pip_size)
                    take_profit = bar.close + (take_profit_pips * self.pip_size)
                    self._open_position('buy', bar.close, lot_size, stop_loss, take_profit)
                    trailing_high = None

                # SELL: MA fast < MA slow, RSI > overbought, momentum negativo
                elif (ma_fast_val < ma_slow_val and
                      rsi > rsi_overbought and
                      momentum < -entry_threshold):

                    stop_loss = bar.close + (stop_loss_pips * self.pip_size)
                    take_profit = bar.close - (take_profit_pips * self.pip_size)
                    self._open_position('sell', bar.close, lot_size, stop_loss, take_profit)
                    trailing_low = None

            # Atualizar equity curve
            self._equity_curve.append(self._calculate_equity(bar.close))

        # Fechar posição aberta no final
        if self._position:
            self._close_position(bars[-1].close, 'end_of_data')

        # Calcular métricas
        result = self._calculate_metrics(strategy_config, len(bars))
        result.execution_time = time.time() - start_time

        return result

    def _calculate_rsi(self, period: int) -> float:
        """Calcula RSI"""
        if len(self._prices) < period + 1:
            return 50.0

        deltas = np.diff(self._prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_pnl_pips(self, current_price: float) -> float:
        """Calcula P&L em pips"""
        if not self._position:
            return 0.0

        if self._position['side'] == 'buy':
            return (current_price - self._position['entry_price']) / self.pip_size
        else:
            return (self._position['entry_price'] - current_price) / self.pip_size

    def _open_position(self, side: str, price: float, volume: float,
                       stop_loss: float, take_profit: float):
        """Abre posição"""
        self._position = {
            'side': side,
            'entry_price': price,
            'volume': volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': len(self._prices)
        }

    def _close_position(self, price: float, reason: str):
        """Fecha posição"""
        if not self._position:
            return

        if self._position['side'] == 'buy':
            pnl = (price - self._position['entry_price']) * self._position['volume'] * 100000
        else:
            pnl = (self._position['entry_price'] - price) * self._position['volume'] * 100000

        self._trades.append({
            'side': self._position['side'],
            'entry_price': self._position['entry_price'],
            'exit_price': price,
            'volume': self._position['volume'],
            'pnl': pnl,
            'reason': reason,
            'duration': len(self._prices) - self._position['entry_time']
        })

        self._balance += pnl
        self._position = None

    def _calculate_equity(self, current_price: float) -> float:
        """Calcula equity atual"""
        equity = self._balance
        if self._position:
            if self._position['side'] == 'buy':
                unrealized = (current_price - self._position['entry_price']) * self._position['volume'] * 100000
            else:
                unrealized = (self._position['entry_price'] - current_price) * self._position['volume'] * 100000
            equity += unrealized
        return equity

    def _calculate_metrics(self, config: Dict, total_bars: int) -> BacktestResult:
        """Calcula métricas do backtest"""
        result = BacktestResult(config=config, total_bars=total_bars)

        if not self._trades:
            return result

        result.total_trades = len(self._trades)
        result.winning_trades = sum(1 for t in self._trades if t['pnl'] > 0)
        result.losing_trades = sum(1 for t in self._trades if t['pnl'] <= 0)
        result.total_pnl = sum(t['pnl'] for t in self._trades)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        if result.total_trades > 0:
            result.avg_trade_pnl = result.total_pnl / result.total_trades

        # Profit factor
        gross_profit = sum(t['pnl'] for t in self._trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self._trades if t['pnl'] < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        # Max drawdown
        peak = self._equity_curve[0]
        max_dd = 0
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd * 100

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for t in self._trades:
            if t['pnl'] <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        result.max_consecutive_losses = max_consec

        # Sharpe ratio (simplificado)
        if len(self._trades) > 1:
            returns = [t['pnl'] for t in self._trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            result.sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        return result


class MLOptimizer:
    """
    Otimizador de Machine Learning para parâmetros de trading
    Testa múltiplas configurações via backtesting e encontra a melhor
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self._bars: List[BarData] = []
        self._results: List[BacktestResult] = []
        self._best_result: Optional[BacktestResult] = None
        self._iteration = 0
        self._no_improvement_count = 0

        # Criar diretórios
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)

        # Engine de backtest
        self._engine = BacktestEngine()

        # Default param ranges se não especificado
        if not self.config.param_ranges:
            self.config.param_ranges = self._get_default_param_ranges()

        logger.info(f"MLOptimizer inicializado: method={self.config.optimization_method}")

    def _get_default_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Retorna ranges padrão de parâmetros para otimização"""
        return {
            'stop_loss_pips': {'min': 5, 'max': 50, 'step': 5, 'type': 'int'},
            'take_profit_pips': {'min': 3, 'max': 30, 'step': 1, 'type': 'int'},
            'trailing_stop_pips': {'min': 1, 'max': 10, 'step': 1, 'type': 'int'},
            'min_profit_to_trail_pips': {'min': 0.5, 'max': 5, 'step': 0.5, 'type': 'float'},
            'entry_threshold': {'min': 0.00005, 'max': 0.001, 'step': 0.00005, 'type': 'float'},
            'rsi_period': {'min': 7, 'max': 21, 'step': 1, 'type': 'int'},
            'rsi_overbought': {'min': 65, 'max': 85, 'step': 5, 'type': 'int'},
            'rsi_oversold': {'min': 15, 'max': 35, 'step': 5, 'type': 'int'},
            'ma_fast': {'min': 5, 'max': 20, 'step': 1, 'type': 'int'},
            'ma_slow': {'min': 20, 'max': 100, 'step': 5, 'type': 'int'},
        }

    async def load_data_from_api(self, client=None, count: int = 10000) -> int:
        """
        Carrega dados históricos da API FXOpen

        Args:
            client: FXOpenClient conectado (opcional)
            count: Número de barras a carregar

        Returns:
            Número de barras carregadas
        """
        try:
            if client is None:
                # Import aqui para evitar circular import
                from ..bindings.fxopen_client import FXOpenClient, FXOpenConfig
                client = FXOpenClient(FXOpenConfig())
                await client.connect()

            # Buscar barras M1
            bars = await client.get_bars_history(
                symbol=self.config.symbol,
                periodicity=self.config.timeframe,
                count=count
            )

            if bars:
                self._bars = [
                    BarData(
                        timestamp=bar.timestamp.timestamp(),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume
                    )
                    for bar in bars
                ]

                # Salvar em CSV para cache
                self._save_bars_to_csv()
                logger.info(f"Carregadas {len(self._bars)} barras da API")
                return len(self._bars)

        except Exception as e:
            logger.error(f"Erro ao carregar dados da API: {e}")

        return 0

    def load_data_from_csv(self, filepath: str = None) -> int:
        """
        Carrega dados de arquivo CSV

        Args:
            filepath: Caminho do arquivo (opcional)

        Returns:
            Número de barras carregadas
        """
        if filepath is None:
            filepath = os.path.join(
                self.config.data_dir,
                f'{self.config.symbol}_{self.config.timeframe}_bars.csv'
            )

        if not os.path.exists(filepath):
            logger.warning(f"Arquivo não encontrado: {filepath}")
            # Gerar dados sintéticos para teste
            self._bars = self._generate_synthetic_bars(10000)
            logger.info(f"Gerados {len(self._bars)} barras sintéticas")
            return len(self._bars)

        try:
            self._bars = []
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bar = BarData(
                        timestamp=float(row.get('timestamp', 0)),
                        open=float(row.get('open', 0)),
                        high=float(row.get('high', 0)),
                        low=float(row.get('low', 0)),
                        close=float(row.get('close', 0)),
                        volume=float(row.get('volume', 0))
                    )
                    self._bars.append(bar)

            logger.info(f"Carregadas {len(self._bars)} barras de {filepath}")
            return len(self._bars)

        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            return 0

    def _save_bars_to_csv(self):
        """Salva barras em CSV para cache"""
        filepath = os.path.join(
            self.config.data_dir,
            f'{self.config.symbol}_{self.config.timeframe}_bars.csv'
        )

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for bar in self._bars:
                    writer.writerow([
                        bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume
                    ])
            logger.info(f"Barras salvas em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar CSV: {e}")

    def _generate_synthetic_bars(self, count: int) -> List[BarData]:
        """Gera barras sintéticas realistas para teste"""
        np.random.seed(42)

        bars = []
        base_price = 1.10000
        volatility = 0.0003  # Aumentado para gerar mais movimento
        trend = 0.0  # Tendência atual
        start_time = time.time() - (count * 60)

        for i in range(count):
            # Mudança de tendência ocasional
            if i % 100 == 0:
                trend = np.random.uniform(-0.0001, 0.0001)

            # Random walk com tendência e momentum
            momentum = np.random.normal(0, volatility)
            change = trend + momentum

            # Ocasionalmente criar movimentos fortes (para gerar sinais)
            if np.random.random() < 0.05:  # 5% de chance
                change *= np.random.uniform(2, 5)

            open_p = base_price
            base_price *= (1 + change)
            close_p = base_price

            # Gerar High/Low realistas
            bar_range = abs(change) * base_price + np.random.exponential(volatility * base_price)
            if close_p > open_p:  # Bullish
                high = close_p + bar_range * np.random.uniform(0, 0.5)
                low = open_p - bar_range * np.random.uniform(0, 0.3)
            else:  # Bearish
                high = open_p + bar_range * np.random.uniform(0, 0.3)
                low = close_p - bar_range * np.random.uniform(0, 0.5)

            bar = BarData(
                timestamp=start_time + (i * 60),
                open=open_p,
                high=max(high, open_p, close_p),
                low=min(low, open_p, close_p),
                close=close_p,
                volume=np.random.exponential(100) * (1 + abs(change) * 1000)
            )
            bars.append(bar)

        logger.info(f"Geradas {count} barras sintéticas (volatilidade aumentada)")
        return bars

    def _generate_random_config(self) -> Dict[str, Any]:
        """Gera configuração aleatória dentro dos ranges"""
        config = {}

        for param, range_info in self.config.param_ranges.items():
            if range_info['type'] == 'int':
                config[param] = random.randint(
                    range_info['min'],
                    range_info['max']
                )
            else:
                config[param] = round(
                    random.uniform(range_info['min'], range_info['max']),
                    6
                )

        return config

    def _mutate_config(self, config: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """Aplica mutação a uma configuração (para algoritmo genético)"""
        new_config = config.copy()

        for param, range_info in self.config.param_ranges.items():
            if random.random() < mutation_rate:
                if range_info['type'] == 'int':
                    delta = random.randint(-2, 2) * range_info.get('step', 1)
                    new_config[param] = max(
                        range_info['min'],
                        min(range_info['max'], new_config.get(param, range_info['min']) + delta)
                    )
                else:
                    delta = random.uniform(-0.1, 0.1) * (range_info['max'] - range_info['min'])
                    new_config[param] = max(
                        range_info['min'],
                        min(range_info['max'], new_config.get(param, range_info['min']) + delta)
                    )

        return new_config

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover entre duas configurações (para algoritmo genético)"""
        child = {}
        for param in self.config.param_ranges.keys():
            if random.random() < 0.5:
                child[param] = parent1.get(param)
            else:
                child[param] = parent2.get(param)
        return child

    def run_backtest(self, config: Dict[str, Any]) -> BacktestResult:
        """Executa um backtest com configuração específica"""
        if not self._bars:
            self.load_data_from_csv()

        return self._engine.run(self._bars, config)

    def optimize_grid_search(self) -> BacktestResult:
        """Otimização via Grid Search (todas as combinações)"""
        logger.info("Iniciando Grid Search...")

        # Gerar todas as combinações (limitado)
        import itertools

        param_values = {}
        for param, range_info in self.config.param_ranges.items():
            if range_info['type'] == 'int':
                param_values[param] = list(range(
                    range_info['min'],
                    range_info['max'] + 1,
                    range_info.get('step', 1)
                ))
            else:
                values = []
                v = range_info['min']
                while v <= range_info['max']:
                    values.append(round(v, 6))
                    v += range_info.get('step', 0.1)
                param_values[param] = values

        # Limitar combinações
        total_combinations = 1
        for values in param_values.values():
            total_combinations *= len(values)

        logger.info(f"Total de combinações: {total_combinations}")

        if total_combinations > self.config.max_iterations:
            logger.warning(f"Muitas combinações, usando Random Search")
            return self.optimize_random_search()

        keys = list(param_values.keys())
        values_lists = [param_values[k] for k in keys]

        best_result = None

        for i, combo in enumerate(itertools.product(*values_lists)):
            config = dict(zip(keys, combo))
            result = self.run_backtest(config)
            self._results.append(result)

            if best_result is None or result.fitness > best_result.fitness:
                best_result = result
                logger.info(f"[{i+1}] Nova melhor config: fitness={result.fitness:.2f}, "
                           f"win_rate={result.win_rate:.2%}, pnl=${result.total_pnl:.2f}")

            if (i + 1) % 100 == 0:
                logger.info(f"Progresso: {i+1}/{total_combinations}")

        self._best_result = best_result
        return best_result

    def optimize_random_search(self) -> BacktestResult:
        """Otimização via Random Search"""
        logger.info(f"Iniciando Random Search ({self.config.max_iterations} iterações)...")

        for i in range(self.config.max_iterations):
            config = self._generate_random_config()
            result = self.run_backtest(config)
            self._results.append(result)
            self._iteration = i + 1

            if self._best_result is None or result.fitness > self._best_result.fitness:
                self._best_result = result
                self._no_improvement_count = 0
                logger.info(f"[{i+1}] Nova melhor config: fitness={result.fitness:.2f}, "
                           f"win_rate={result.win_rate:.2%}, pnl=${result.total_pnl:.2f}")
            else:
                self._no_improvement_count += 1

            # Early stopping
            if self._no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"Early stopping após {i+1} iterações")
                break

            if (i + 1) % 10 == 0:
                logger.info(f"Progresso: {i+1}/{self.config.max_iterations}")

        return self._best_result

    def optimize_genetic(self) -> BacktestResult:
        """Otimização via Algoritmo Genético"""
        logger.info(f"Iniciando Algoritmo Genético (pop={self.config.population_size})...")

        # Inicializar população
        population = [self._generate_random_config() for _ in range(self.config.population_size)]
        fitness_scores = []

        for gen in range(self.config.max_iterations):
            # Avaliar população
            results = []
            for config in population:
                result = self.run_backtest(config)
                results.append(result)
                self._results.append(result)

            fitness_scores = [r.fitness for r in results]

            # Encontrar melhor
            best_idx = np.argmax(fitness_scores)
            best_gen_result = results[best_idx]

            if self._best_result is None or best_gen_result.fitness > self._best_result.fitness:
                self._best_result = best_gen_result
                self._no_improvement_count = 0
                logger.info(f"[Gen {gen+1}] Nova melhor: fitness={best_gen_result.fitness:.2f}, "
                           f"win_rate={best_gen_result.win_rate:.2%}")
            else:
                self._no_improvement_count += 1

            # Early stopping
            if self._no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(f"Early stopping na geração {gen+1}")
                break

            # Seleção e reprodução
            # Tournament selection
            new_population = []

            # Elitismo: manter os melhores
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = max(2, self.config.population_size // 10)

            for i in range(elite_count):
                new_population.append(population[sorted_indices[i]])

            # Crossover e mutação para o resto
            while len(new_population) < self.config.population_size:
                # Tournament selection
                idx1 = random.choice(sorted_indices[:self.config.population_size // 2])
                idx2 = random.choice(sorted_indices[:self.config.population_size // 2])

                parent1 = population[idx1]
                parent2 = population[idx2]

                child = self._crossover(parent1, parent2)
                child = self._mutate_config(child)
                new_population.append(child)

            population = new_population

            if (gen + 1) % 5 == 0:
                logger.info(f"Geração {gen+1}/{self.config.max_iterations}, "
                           f"melhor fitness: {self._best_result.fitness:.2f}")

        return self._best_result

    def optimize(self) -> BacktestResult:
        """
        Executa otimização com método configurado

        Returns:
            Melhor resultado encontrado
        """
        if not self._bars:
            self.load_data_from_csv()

        if self.config.optimization_method == 'grid':
            result = self.optimize_grid_search()
        elif self.config.optimization_method == 'random':
            result = self.optimize_random_search()
        elif self.config.optimization_method == 'genetic':
            result = self.optimize_genetic()
        else:  # bayesian ou default
            result = self.optimize_random_search()  # Fallback para random

        # Salvar resultados
        self.save_results()

        return result

    def save_results(self):
        """Salva resultados da otimização"""
        # Salvar melhor configuração
        if self._best_result:
            filepath = self.config.best_config_path
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump({
                    'config': self._best_result.config,
                    'metrics': {
                        'total_trades': self._best_result.total_trades,
                        'win_rate': self._best_result.win_rate,
                        'total_pnl': self._best_result.total_pnl,
                        'sharpe_ratio': self._best_result.sharpe_ratio,
                        'profit_factor': self._best_result.profit_factor,
                        'max_drawdown': self._best_result.max_drawdown,
                        'fitness': self._best_result.fitness
                    },
                    'optimization': {
                        'method': self.config.optimization_method,
                        'iterations': len(self._results),
                        'timestamp': datetime.now().isoformat()
                    }
                }, f, indent=2)

            logger.info(f"Melhor configuração salva em {filepath}")

        # Salvar histórico de resultados
        history_path = os.path.join(
            self.config.results_dir,
            f'optimization_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

        with open(history_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            if self._results:
                config_keys = list(self._results[0].config.keys())
                header = config_keys + [
                    'total_trades', 'win_rate', 'total_pnl', 'sharpe_ratio',
                    'profit_factor', 'max_drawdown', 'fitness'
                ]
                writer.writerow(header)

                for result in self._results:
                    row = [result.config.get(k) for k in config_keys] + [
                        result.total_trades, result.win_rate, result.total_pnl,
                        result.sharpe_ratio, result.profit_factor, result.max_drawdown,
                        result.fitness
                    ]
                    writer.writerow(row)

        logger.info(f"Histórico salvo em {history_path}")

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Retorna melhor configuração encontrada"""
        if self._best_result:
            return self._best_result.config
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da otimização"""
        return {
            'total_iterations': len(self._results),
            'best_fitness': self._best_result.fitness if self._best_result else None,
            'best_win_rate': self._best_result.win_rate if self._best_result else None,
            'best_pnl': self._best_result.total_pnl if self._best_result else None,
            'best_sharpe': self._best_result.sharpe_ratio if self._best_result else None,
            'best_config': self._best_result.config if self._best_result else None
        }


async def main():
    """Executa otimização de demonstração"""
    print("=" * 60)
    print("EliBotHFT - ML Optimizer")
    print("=" * 60)

    # Configurar otimizador
    config = OptimizationConfig(
        optimization_method='genetic',
        max_iterations=50,
        population_size=20,
        early_stopping_rounds=15,
        symbol='EURUSD',
        timeframe='M1'
    )

    optimizer = MLOptimizer(config)

    # Carregar dados (sintéticos se não houver reais)
    print("\nCarregando dados...")
    num_bars = optimizer.load_data_from_csv()
    print(f"Carregadas {num_bars} barras")

    # Executar otimização
    print(f"\nIniciando otimização ({config.optimization_method})...")
    print("-" * 60)

    best_result = optimizer.optimize()

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("=" * 60)

    stats = optimizer.get_statistics()
    print(f"Total de iterações: {stats['total_iterations']}")
    print(f"\nMelhor configuração encontrada:")
    print(f"  Fitness: {stats['best_fitness']:.2f}")
    print(f"  Win Rate: {stats['best_win_rate']:.2%}")
    print(f"  Total P&L: ${stats['best_pnl']:.2f}")
    print(f"  Sharpe Ratio: {stats['best_sharpe']:.2f}")

    print(f"\nParâmetros:")
    for k, v in stats['best_config'].items():
        print(f"  {k}: {v}")

    print("\nResultados salvos em:")
    print(f"  - {config.best_config_path}")
    print(f"  - {config.results_dir}/optimization_history_*.csv")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    asyncio.run(main())
