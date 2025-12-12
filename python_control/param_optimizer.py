"""
Parameter Optimizer - Otimizador de parâmetros de estratégia
Usa algoritmos de ML para ajustar strategy_params.yaml
"""

import os
import sys
import yaml
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de otimização"""
    parameters: Dict[str, Any]
    score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float


@dataclass
class BacktestResult:
    """Resultado de backtest"""
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    trades: List[Dict] = field(default_factory=list)


class ParameterOptimizer:
    """
    Otimizador de Parâmetros para Estratégias HFT

    Métodos:
    - Grid Search
    - Random Search
    - Bayesian Optimization (opcional)
    - Genetic Algorithm (opcional)
    """

    def __init__(self, config_path: str = None, data_path: str = None):
        """
        Inicializa otimizador

        Args:
            config_path: Caminho para strategy_params.yaml
            data_path: Caminho para dados de mercado
        """
        self.config_path = config_path or 'config/strategy_params.yaml'
        self.data_path = data_path or 'data/market_data'

        # Parâmetros a otimizar com ranges
        self.parameter_space: Dict[str, Dict] = {
            'market_making.spread_threshold_pips': {'min': 0.5, 'max': 3.0, 'step': 0.5},
            'market_making.min_edge_pips': {'min': 0.1, 'max': 1.0, 'step': 0.1},
            'market_making.imbalance_threshold': {'min': 0.1, 'max': 0.5, 'step': 0.1},
            'momentum.lookback_ticks': {'min': 20, 'max': 200, 'step': 20},
            'momentum.threshold_std': {'min': 1.0, 'max': 3.0, 'step': 0.5},
            'momentum.entry_momentum_min': {'min': 0.0001, 'max': 0.001, 'step': 0.0001},
            'mean_reversion.bollinger_period': {'min': 10, 'max': 50, 'step': 5},
            'mean_reversion.bollinger_std': {'min': 1.5, 'max': 3.0, 'step': 0.5},
            'mean_reversion.rsi_oversold': {'min': 20, 'max': 40, 'step': 5},
            'mean_reversion.rsi_overbought': {'min': 60, 'max': 80, 'step': 5},
        }

        # Resultados
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

        # Dados de mercado (cache)
        self._market_data: Dict[str, np.ndarray] = {}

        logger.info("ParameterOptimizer inicializado")

    def load_config(self) -> Dict:
        """Carrega configuração atual"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def save_config(self, config: Dict) -> None:
        """Salva configuração"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuração salva em {self.config_path}")

    def load_market_data(self, symbol: str, start_date: datetime = None,
                        end_date: datetime = None) -> np.ndarray:
        """
        Carrega dados de mercado

        Args:
            symbol: Símbolo
            start_date: Data inicial
            end_date: Data final

        Returns:
            Array com dados OHLCV
        """
        # Tentar carregar de arquivo
        data_file = os.path.join(self.data_path, f'{symbol}_ticks.csv')

        if os.path.exists(data_file):
            import csv
            data = []
            with open(data_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append([
                        float(row.get('timestamp', 0)),
                        float(row.get('bid', 0)),
                        float(row.get('ask', 0)),
                        float(row.get('volume', 0))
                    ])
            return np.array(data)

        # Gerar dados sintéticos para teste
        logger.warning(f"Dados não encontrados para {symbol}, gerando sintéticos")
        return self._generate_synthetic_data(symbol, 10000)

    def _generate_synthetic_data(self, symbol: str, num_points: int) -> np.ndarray:
        """Gera dados sintéticos para backtesting"""
        np.random.seed(42)

        # Parâmetros baseados no símbolo
        if symbol == 'EURUSD':
            base_price = 1.10
            volatility = 0.0002
        elif symbol == 'GBPUSD':
            base_price = 1.27
            volatility = 0.0002
        else:
            base_price = 1.0
            volatility = 0.0003

        # Gerar random walk
        returns = np.random.normal(0, volatility, num_points)
        prices = base_price * np.exp(np.cumsum(returns))

        # Criar spread
        spread = base_price * 0.0001
        bids = prices - spread / 2
        asks = prices + spread / 2

        # Timestamps
        timestamps = np.arange(num_points)

        # Volume aleatório
        volumes = np.random.exponential(100, num_points)

        return np.column_stack([timestamps, bids, asks, volumes])

    def backtest(self, params: Dict, data: np.ndarray) -> BacktestResult:
        """
        Executa backtest com parâmetros específicos

        Args:
            params: Parâmetros da estratégia
            data: Dados de mercado

        Returns:
            Resultado do backtest
        """
        trades = []
        position = 0
        entry_price = 0
        equity = 10000.0
        peak_equity = equity
        max_drawdown = 0

        # Extrair parâmetros
        momentum_threshold = params.get('momentum.entry_momentum_min', 0.0003)
        lookback = int(params.get('momentum.lookback_ticks', 100))
        stop_loss_pct = 0.002
        take_profit_pct = 0.003

        for i in range(lookback, len(data)):
            bid = data[i, 1]
            ask = data[i, 2]
            mid = (bid + ask) / 2

            # Calcular momentum
            past_mid = (data[i - lookback, 1] + data[i - lookback, 2]) / 2
            momentum = (mid - past_mid) / past_mid

            # Lógica de trading simplificada
            if position == 0:
                # Entrada
                if momentum > momentum_threshold:
                    position = 1
                    entry_price = ask
                elif momentum < -momentum_threshold:
                    position = -1
                    entry_price = bid
            else:
                # Saída
                if position == 1:
                    pnl_pct = (bid - entry_price) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        pnl = (bid - entry_price) * 100  # Assumindo 1 lot
                        equity += pnl
                        trades.append({
                            'entry': entry_price,
                            'exit': bid,
                            'pnl': pnl,
                            'side': 'long'
                        })
                        position = 0

                elif position == -1:
                    pnl_pct = (entry_price - ask) / entry_price
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        pnl = (entry_price - ask) * 100
                        equity += pnl
                        trades.append({
                            'entry': entry_price,
                            'exit': ask,
                            'pnl': pnl,
                            'side': 'short'
                        })
                        position = 0

            # Drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calcular métricas
        if not trades:
            return BacktestResult(
                total_pnl=0, sharpe_ratio=0, max_drawdown=max_drawdown,
                win_rate=0, total_trades=0, profit_factor=0
            )

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')

        # Sharpe ratio simplificado
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0

        return BacktestResult(
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            trades=trades
        )

    def _score_result(self, result: BacktestResult) -> float:
        """
        Calcula score composto do resultado

        Considera:
        - Sharpe Ratio (40%)
        - Profit Factor (30%)
        - Win Rate (20%)
        - Drawdown (penalidade) (10%)
        """
        if result.total_trades < 10:
            return -float('inf')  # Poucos trades

        score = (
            result.sharpe_ratio * 0.4 +
            min(result.profit_factor, 5) * 0.3 +  # Cap em 5
            result.win_rate * 100 * 0.2 -
            result.max_drawdown * 100 * 0.1
        )

        return score

    def grid_search(self, symbol: str, param_names: List[str] = None) -> OptimizationResult:
        """
        Otimização por Grid Search

        Args:
            symbol: Símbolo para backtest
            param_names: Nomes dos parâmetros a otimizar

        Returns:
            Melhor resultado encontrado
        """
        if param_names is None:
            param_names = list(self.parameter_space.keys())[:3]  # Limitar a 3

        data = self.load_market_data(symbol)
        logger.info(f"Grid Search iniciado para {symbol} com {len(data)} pontos")

        # Gerar grid
        grids = {}
        for name in param_names:
            if name in self.parameter_space:
                space = self.parameter_space[name]
                grids[name] = np.arange(space['min'], space['max'] + space['step'], space['step'])

        # Iterar sobre todas as combinações
        best_score = -float('inf')
        best_params = {}
        best_result = None
        total_combinations = np.prod([len(g) for g in grids.values()])

        logger.info(f"Total de combinações: {total_combinations}")

        def iterate_grid(params, keys, index):
            nonlocal best_score, best_params, best_result

            if index == len(keys):
                # Executar backtest
                result = self.backtest(params, data)
                score = self._score_result(result)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result

                return

            key = keys[index]
            for value in grids[key]:
                params[key] = value
                iterate_grid(params, keys, index + 1)

        iterate_grid({}, list(grids.keys()), 0)

        if best_result:
            opt_result = OptimizationResult(
                parameters=best_params,
                score=best_score,
                sharpe_ratio=best_result.sharpe_ratio,
                max_drawdown=best_result.max_drawdown,
                win_rate=best_result.win_rate,
                total_trades=best_result.total_trades,
                profit_factor=best_result.profit_factor
            )

            self.results.append(opt_result)
            self.best_result = opt_result

            logger.info(f"Melhor resultado: Score={best_score:.2f}, Sharpe={best_result.sharpe_ratio:.2f}")

            return opt_result

        return None

    def random_search(self, symbol: str, n_iterations: int = 100) -> OptimizationResult:
        """
        Otimização por Random Search

        Args:
            symbol: Símbolo para backtest
            n_iterations: Número de iterações

        Returns:
            Melhor resultado encontrado
        """
        data = self.load_market_data(symbol)
        logger.info(f"Random Search iniciado: {n_iterations} iterações")

        best_score = -float('inf')
        best_params = {}
        best_result = None

        for i in range(n_iterations):
            # Gerar parâmetros aleatórios
            params = {}
            for name, space in self.parameter_space.items():
                if isinstance(space['step'], float):
                    params[name] = np.random.uniform(space['min'], space['max'])
                else:
                    params[name] = np.random.randint(space['min'], space['max'] + 1)

            # Executar backtest
            result = self.backtest(params, data)
            score = self._score_result(result)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_result = result

            if (i + 1) % 10 == 0:
                logger.info(f"Iteração {i+1}/{n_iterations}, Melhor score: {best_score:.2f}")

        if best_result:
            opt_result = OptimizationResult(
                parameters=best_params,
                score=best_score,
                sharpe_ratio=best_result.sharpe_ratio,
                max_drawdown=best_result.max_drawdown,
                win_rate=best_result.win_rate,
                total_trades=best_result.total_trades,
                profit_factor=best_result.profit_factor
            )

            self.results.append(opt_result)
            self.best_result = opt_result

            return opt_result

        return None

    def apply_best_params(self) -> bool:
        """Aplica melhores parâmetros ao arquivo de configuração"""
        if not self.best_result:
            logger.warning("Nenhum resultado de otimização disponível")
            return False

        config = self.load_config()

        # Aplicar parâmetros
        for param_path, value in self.best_result.parameters.items():
            parts = param_path.split('.')
            obj = config

            for part in parts[:-1]:
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]

            obj[parts[-1]] = value

        self.save_config(config)
        logger.info("Melhores parâmetros aplicados à configuração")

        return True

    def report(self) -> str:
        """Gera relatório de otimização"""
        if not self.results:
            return "Nenhum resultado de otimização"

        lines = [
            "=" * 60,
            "RELATÓRIO DE OTIMIZAÇÃO",
            "=" * 60,
            f"\nTotal de testes: {len(self.results)}",
        ]

        if self.best_result:
            lines.extend([
                "\nMELHOR RESULTADO:",
                f"  Score: {self.best_result.score:.2f}",
                f"  Sharpe Ratio: {self.best_result.sharpe_ratio:.2f}",
                f"  Profit Factor: {self.best_result.profit_factor:.2f}",
                f"  Win Rate: {self.best_result.win_rate:.1%}",
                f"  Max Drawdown: {self.best_result.max_drawdown:.2%}",
                f"  Total Trades: {self.best_result.total_trades}",
                "\nPARÂMETROS OTIMIZADOS:"
            ])

            for param, value in self.best_result.parameters.items():
                lines.append(f"  {param}: {value}")

        lines.append("\n" + "=" * 60)

        return '\n'.join(lines)


def main():
    """Executa otimização"""
    optimizer = ParameterOptimizer()

    print("EliBotHFT - Otimizador de Parâmetros")
    print("=" * 40)

    # Random search
    print("\nExecutando Random Search...")
    result = optimizer.random_search('EURUSD', n_iterations=50)

    if result:
        print(optimizer.report())

        # Perguntar se deseja aplicar
        response = input("\nAplicar parâmetros otimizados? (s/n): ")
        if response.lower() == 's':
            optimizer.apply_best_params()
            print("Parâmetros aplicados!")
    else:
        print("Nenhum resultado válido encontrado")


if __name__ == '__main__':
    main()
