"""
Performance Analytics and Metrics for ML Trading System

This module provides comprehensive performance analysis:
1. Trading Performance Metrics (Sharpe, Sortino, Calmar, etc.)
2. Risk Metrics (VaR, CVaR, Max Drawdown)
3. ML Model Performance Metrics
4. Strategy Comparison and Ranking
5. Portfolio Analytics
6. Real-time Performance Monitoring
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    RETURN = auto()
    RISK = auto()
    RISK_ADJUSTED = auto()
    TRADE = auto()
    MODEL = auto()
    PORTFOLIO = auto()


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    symbol: str
    strategy: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    entry_time: float
    exit_time: float
    holding_time_ms: int
    slippage: float = 0.0
    commission: float = 0.0
    market_regime: str = "UNKNOWN"
    signal_confidence: float = 0.0
    was_error: bool = False
    error_type: Optional[str] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    # Time period
    start_time: float
    end_time: float
    duration_days: float

    # Returns
    total_return: float
    annualized_return: float
    daily_returns: List[float]
    monthly_returns: List[float]

    # Risk
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: float
    var_95: float
    cvar_95: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float

    # Trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_time: float
    avg_trades_per_day: float

    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'period': {
                'start': self.start_time,
                'end': self.end_time,
                'duration_days': self.duration_days
            },
            'returns': {
                'total': self.total_return,
                'annualized': self.annualized_return
            },
            'risk': {
                'volatility': self.volatility,
                'downside_volatility': self.downside_volatility,
                'max_drawdown': self.max_drawdown,
                'var_95': self.var_95,
                'cvar_95': self.cvar_95
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'omega_ratio': self.omega_ratio
            },
            'trading': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_holding_time': self.avg_holding_time
            },
            'strategy_breakdown': self.strategy_performance
        }


class ReturnMetrics:
    """Calculate return-based metrics"""

    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        """Total return percentage"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    @staticmethod
    def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Annualized return"""
        if len(returns) == 0:
            return 0.0
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        return (1 + total_return) ** (periods_per_year / n_periods) - 1

    @staticmethod
    def rolling_returns(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling cumulative returns"""
        if len(returns) < window:
            return np.array([])
        rolling = np.array([
            np.prod(1 + returns[i:i+window]) - 1
            for i in range(len(returns) - window + 1)
        ])
        return rolling

    @staticmethod
    def compound_returns(returns: np.ndarray) -> np.ndarray:
        """Compound (cumulative) returns"""
        return np.cumprod(1 + returns) - 1

    @staticmethod
    def log_returns(prices: np.ndarray) -> np.ndarray:
        """Log returns from prices"""
        if len(prices) < 2:
            return np.array([])
        return np.log(prices[1:] / prices[:-1])

    @staticmethod
    def simple_returns(prices: np.ndarray) -> np.ndarray:
        """Simple returns from prices"""
        if len(prices) < 2:
            return np.array([])
        return (prices[1:] - prices[:-1]) / prices[:-1]


class RiskMetrics:
    """Calculate risk metrics"""

    @staticmethod
    def volatility(returns: np.ndarray, annualize: bool = True, periods: int = 252) -> float:
        """Annualized volatility"""
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1)
        if annualize:
            vol *= np.sqrt(periods)
        return vol

    @staticmethod
    def downside_volatility(returns: np.ndarray, threshold: float = 0.0, annualize: bool = True, periods: int = 252) -> float:
        """Downside volatility (semi-deviation)"""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < threshold]
        if len(downside_returns) < 2:
            return 0.0
        vol = np.std(downside_returns, ddof=1)
        if annualize:
            vol *= np.sqrt(periods)
        return vol

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Maximum drawdown and its duration.

        Returns:
            (max_drawdown, peak_idx, trough_idx)
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0

        peak = equity_curve[0]
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        current_peak_idx = 0

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                current_peak_idx = i

            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                peak_idx = current_peak_idx
                trough_idx = i

        return max_dd, peak_idx, trough_idx

    @staticmethod
    def drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
        """Calculate drawdown at each point"""
        if len(equity_curve) < 1:
            return np.array([])

        peak = equity_curve[0]
        drawdowns = []

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdowns.append((peak - value) / peak)

        return np.array(drawdowns)

    @staticmethod
    def var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk"""
        if len(returns) < 10:
            return 0.0
        return -np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) < 10:
            return 0.0
        var = RiskMetrics.var(returns, confidence)
        return -np.mean(returns[returns <= -var])

    @staticmethod
    def tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
        """Ratio of positive tail to negative tail"""
        if len(returns) < 20:
            return 1.0
        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = np.percentile(returns, percentile)
        if abs(left_tail) < 1e-10:
            return 10.0
        return abs(right_tail / left_tail)

    @staticmethod
    def ulcer_index(equity_curve: np.ndarray) -> float:
        """Ulcer Index - measures depth and duration of drawdowns"""
        drawdowns = RiskMetrics.drawdown_series(equity_curve)
        if len(drawdowns) == 0:
            return 0.0
        return np.sqrt(np.mean(drawdowns ** 2))


class RiskAdjustedMetrics:
    """Risk-adjusted return metrics"""

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods: int = 252
    ) -> float:
        """Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods
        vol = np.std(excess_returns, ddof=1)

        if vol == 0:
            return 0.0

        return np.mean(excess_returns) / vol * np.sqrt(periods)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods: int = 252
    ) -> float:
        """Sortino Ratio (using downside deviation)"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods
        downside_vol = RiskMetrics.downside_volatility(excess_returns, 0, False)

        if downside_vol == 0:
            return 0.0

        return np.mean(excess_returns) / downside_vol * np.sqrt(periods)

    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity_curve: np.ndarray, periods: int = 252) -> float:
        """Calmar Ratio (return / max drawdown)"""
        max_dd, _, _ = RiskMetrics.max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0

        ann_return = ReturnMetrics.annualized_return(returns, periods)
        return ann_return / max_dd

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Omega Ratio"""
        if len(returns) < 2:
            return 1.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        sum_losses = np.sum(losses)
        if sum_losses == 0:
            return 10.0  # Max value

        return np.sum(gains) / sum_losses

    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods: int = 252
    ) -> float:
        """Information Ratio"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)

        if tracking_error == 0:
            return 0.0

        return np.mean(active_returns) / tracking_error * np.sqrt(periods)

    @staticmethod
    def treynor_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods: int = 252
    ) -> float:
        """Treynor Ratio"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns)

        if benchmark_var == 0:
            return 0.0

        beta = covariance / benchmark_var

        if beta == 0:
            return 0.0

        excess_return = ReturnMetrics.annualized_return(returns, periods) - risk_free_rate
        return excess_return / beta


class TradingMetrics:
    """Trading-specific metrics"""

    @staticmethod
    def win_rate(trades: List[TradeRecord]) -> float:
        """Percentage of winning trades"""
        if not trades:
            return 0.0
        winning = sum(1 for t in trades if t.pnl > 0)
        return winning / len(trades)

    @staticmethod
    def profit_factor(trades: List[TradeRecord]) -> float:
        """Gross profit / gross loss"""
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def expectancy(trades: List[TradeRecord]) -> float:
        """Expected value per trade"""
        if not trades:
            return 0.0

        win_rate = TradingMetrics.win_rate(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0

        return win_rate * avg_win - (1 - win_rate) * avg_loss

    @staticmethod
    def payoff_ratio(trades: List[TradeRecord]) -> float:
        """Average win / average loss"""
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        if not winning_trades or not losing_trades:
            return 0.0

        avg_win = np.mean([t.pnl for t in winning_trades])
        avg_loss = abs(np.mean([t.pnl for t in losing_trades]))

        if avg_loss == 0:
            return 10.0

        return avg_win / avg_loss

    @staticmethod
    def avg_holding_time(trades: List[TradeRecord]) -> float:
        """Average holding time in milliseconds"""
        if not trades:
            return 0.0
        return np.mean([t.holding_time_ms for t in trades])

    @staticmethod
    def trades_per_day(trades: List[TradeRecord]) -> float:
        """Average trades per day"""
        if not trades:
            return 0.0

        times = [t.entry_time for t in trades]
        if len(times) < 2:
            return len(trades)

        duration_days = (max(times) - min(times)) / (24 * 60 * 60)
        if duration_days == 0:
            return len(trades)

        return len(trades) / duration_days

    @staticmethod
    def consecutive_wins_losses(trades: List[TradeRecord]) -> Tuple[int, int]:
        """Maximum consecutive wins and losses"""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    @staticmethod
    def trade_efficiency(trades: List[TradeRecord]) -> float:
        """
        Trade efficiency - how much of potential profit was captured.
        Uses MAE/MFE concept if available.
        """
        if not trades:
            return 0.0

        # Simplified: ratio of actual PnL to potential based on entry/exit
        efficiencies = []
        for t in trades:
            if t.side == 'long':
                potential = abs(t.exit_price - t.entry_price) / t.entry_price
            else:
                potential = abs(t.entry_price - t.exit_price) / t.entry_price

            if potential > 0:
                actual = abs(t.pnl_pct)
                efficiencies.append(min(1.0, actual / potential))

        return np.mean(efficiencies) if efficiencies else 0.0


class ModelMetrics:
    """ML Model performance metrics"""

    @staticmethod
    def accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Classification accuracy"""
        if len(predictions) != len(actuals):
            return 0.0
        return np.mean(predictions == actuals)

    @staticmethod
    def precision_recall_f1(
        predictions: np.ndarray,
        actuals: np.ndarray,
        positive_class: int = 1
    ) -> Tuple[float, float, float]:
        """Precision, Recall, F1 score"""
        true_positives = np.sum((predictions == positive_class) & (actuals == positive_class))
        predicted_positives = np.sum(predictions == positive_class)
        actual_positives = np.sum(actuals == positive_class)

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    @staticmethod
    def confusion_matrix(
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_classes: int = 2
    ) -> np.ndarray:
        """Compute confusion matrix"""
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        for pred, actual in zip(predictions, actuals):
            if 0 <= pred < n_classes and 0 <= actual < n_classes:
                matrix[actual, pred] += 1
        return matrix

    @staticmethod
    def roc_auc(probabilities: np.ndarray, actuals: np.ndarray) -> float:
        """ROC AUC score"""
        if len(probabilities) != len(actuals) or len(probabilities) < 2:
            return 0.5

        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_actuals = actuals[sorted_indices]

        # Calculate AUC using trapezoidal rule
        n_pos = np.sum(sorted_actuals == 1)
        n_neg = len(sorted_actuals) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = np.cumsum(sorted_actuals == 1) / n_pos
        fpr = np.cumsum(sorted_actuals == 0) / n_neg

        auc = np.trapz(tpr, fpr)
        return auc

    @staticmethod
    def log_loss(probabilities: np.ndarray, actuals: np.ndarray) -> float:
        """Logarithmic loss"""
        if len(probabilities) != len(actuals):
            return float('inf')

        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)

        loss = -np.mean(
            actuals * np.log(probabilities) +
            (1 - actuals) * np.log(1 - probabilities)
        )
        return loss

    @staticmethod
    def brier_score(probabilities: np.ndarray, actuals: np.ndarray) -> float:
        """Brier score for probability calibration"""
        if len(probabilities) != len(actuals):
            return 1.0
        return np.mean((probabilities - actuals) ** 2)

    @staticmethod
    def calibration_curve(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibration curve (reliability diagram).

        Returns:
            (mean_predicted_probs, actual_fractions)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        mean_predicted = []
        fraction_positive = []

        for i in range(n_bins):
            mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
            if np.sum(mask) > 0:
                mean_predicted.append(np.mean(probabilities[mask]))
                fraction_positive.append(np.mean(actuals[mask]))

        return np.array(mean_predicted), np.array(fraction_positive)


class PerformanceAnalyzer:
    """
    Main class for comprehensive performance analysis.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self.daily_returns: List[float] = []
        self.timestamps: List[float] = []

        # Model predictions tracking
        self.predictions: List[Tuple[int, int, float]] = []  # (predicted, actual, probability)

        # Strategy-level tracking
        self.strategy_trades: Dict[str, List[TradeRecord]] = {}
        self.strategy_equity: Dict[str, List[float]] = {}

        logger.info(f"PerformanceAnalyzer initialized with capital={initial_capital}")

    def add_trade(self, trade: TradeRecord):
        """Add a completed trade"""
        self.trades.append(trade)

        # Update equity curve
        new_equity = self.equity_curve[-1] + trade.pnl
        self.equity_curve.append(new_equity)
        self.timestamps.append(trade.exit_time)

        # Update daily returns (simplified)
        pnl_pct = trade.pnl / self.equity_curve[-2] if self.equity_curve[-2] > 0 else 0
        self.daily_returns.append(pnl_pct)

        # Strategy-level tracking
        if trade.strategy not in self.strategy_trades:
            self.strategy_trades[trade.strategy] = []
            self.strategy_equity[trade.strategy] = [self.initial_capital / 10]

        self.strategy_trades[trade.strategy].append(trade)
        strat_equity = self.strategy_equity[trade.strategy][-1] + trade.pnl
        self.strategy_equity[trade.strategy].append(strat_equity)

    def add_prediction(self, predicted: int, actual: int, probability: float):
        """Add a model prediction for tracking"""
        self.predictions.append((predicted, actual, probability))

    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        if not self.trades:
            return self._empty_report()

        # Time period
        start_time = min(t.entry_time for t in self.trades)
        end_time = max(t.exit_time for t in self.trades)
        duration_days = (end_time - start_time) / (24 * 60 * 60)

        # Returns
        returns = np.array(self.daily_returns)
        equity = np.array(self.equity_curve)

        total_return = ReturnMetrics.total_return(equity)
        annualized_return = ReturnMetrics.annualized_return(returns) if len(returns) > 0 else 0

        # Risk
        volatility = RiskMetrics.volatility(returns)
        downside_vol = RiskMetrics.downside_volatility(returns)
        max_dd, peak_idx, trough_idx = RiskMetrics.max_drawdown(equity)
        var_95 = RiskMetrics.var(returns)
        cvar_95 = RiskMetrics.cvar(returns)

        # Max drawdown duration
        if peak_idx < trough_idx and len(self.timestamps) > trough_idx:
            max_dd_duration = (self.timestamps[trough_idx] - self.timestamps[peak_idx]) / (24 * 60 * 60)
        else:
            max_dd_duration = 0

        # Risk-adjusted
        sharpe = RiskAdjustedMetrics.sharpe_ratio(returns)
        sortino = RiskAdjustedMetrics.sortino_ratio(returns)
        calmar = RiskAdjustedMetrics.calmar_ratio(returns, equity)
        omega = RiskAdjustedMetrics.omega_ratio(returns)

        # Trading metrics
        win_rate = TradingMetrics.win_rate(self.trades)
        profit_factor = TradingMetrics.profit_factor(self.trades)
        avg_holding = TradingMetrics.avg_holding_time(self.trades)
        trades_per_day = TradingMetrics.trades_per_day(self.trades)

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0

        # Strategy breakdown
        strategy_performance = {}
        for strategy, trades in self.strategy_trades.items():
            strategy_returns = [t.pnl_pct for t in trades]
            strategy_equity = np.array(self.strategy_equity[strategy])

            strategy_performance[strategy] = {
                'total_trades': len(trades),
                'win_rate': TradingMetrics.win_rate(trades),
                'profit_factor': TradingMetrics.profit_factor(trades),
                'total_pnl': sum(t.pnl for t in trades),
                'sharpe': RiskAdjustedMetrics.sharpe_ratio(np.array(strategy_returns)) if len(strategy_returns) > 1 else 0,
                'max_drawdown': RiskMetrics.max_drawdown(strategy_equity)[0]
            }

        # Monthly returns (simplified)
        monthly_returns = []
        if len(returns) >= 20:
            for i in range(0, len(returns), 20):
                chunk = returns[i:i+20]
                monthly_returns.append(np.sum(chunk))

        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            duration_days=duration_days,
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=list(returns),
            monthly_returns=monthly_returns,
            volatility=volatility,
            downside_volatility=downside_vol,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            information_ratio=0.0,  # Would need benchmark
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_time=avg_holding,
            avg_trades_per_day=trades_per_day,
            strategy_performance=strategy_performance
        )

    def _empty_report(self) -> PerformanceReport:
        """Return empty report when no trades"""
        return PerformanceReport(
            start_time=0, end_time=0, duration_days=0,
            total_return=0, annualized_return=0,
            daily_returns=[], monthly_returns=[],
            volatility=0, downside_volatility=0,
            max_drawdown=0, max_drawdown_duration=0,
            var_95=0, cvar_95=0,
            sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, omega_ratio=0, information_ratio=0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, profit_factor=0,
            avg_win=0, avg_loss=0,
            largest_win=0, largest_loss=0,
            avg_holding_time=0, avg_trades_per_day=0,
            strategy_performance={}
        )

    def get_model_metrics(self) -> Dict[str, float]:
        """Get ML model performance metrics"""
        if not self.predictions:
            return {}

        predictions = np.array([p[0] for p in self.predictions])
        actuals = np.array([p[1] for p in self.predictions])
        probabilities = np.array([p[2] for p in self.predictions])

        precision, recall, f1 = ModelMetrics.precision_recall_f1(predictions, actuals)

        return {
            'accuracy': ModelMetrics.accuracy(predictions, actuals),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': ModelMetrics.roc_auc(probabilities, actuals),
            'log_loss': ModelMetrics.log_loss(probabilities, actuals),
            'brier_score': ModelMetrics.brier_score(probabilities, actuals),
            'n_predictions': len(self.predictions)
        }

    def get_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Performance breakdown by market regime"""
        regime_trades: Dict[str, List[TradeRecord]] = {}

        for trade in self.trades:
            regime = trade.market_regime
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        performance = {}
        for regime, trades in regime_trades.items():
            performance[regime] = {
                'n_trades': len(trades),
                'win_rate': TradingMetrics.win_rate(trades),
                'profit_factor': TradingMetrics.profit_factor(trades),
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades])
            }

        return performance

    def get_time_analysis(self) -> Dict[str, Any]:
        """Analysis by time of day / day of week"""
        from datetime import datetime

        hour_pnl: Dict[int, List[float]] = {h: [] for h in range(24)}
        day_pnl: Dict[int, List[float]] = {d: [] for d in range(7)}

        for trade in self.trades:
            dt = datetime.fromtimestamp(trade.entry_time)
            hour_pnl[dt.hour].append(trade.pnl)
            day_pnl[dt.weekday()].append(trade.pnl)

        return {
            'by_hour': {
                h: {
                    'n_trades': len(pnls),
                    'total_pnl': sum(pnls),
                    'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
                }
                for h, pnls in hour_pnl.items()
            },
            'by_day': {
                d: {
                    'n_trades': len(pnls),
                    'total_pnl': sum(pnls),
                    'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
                }
                for d, pnls in day_pnl.items()
            }
        }

    def compare_strategies(self) -> List[Dict[str, Any]]:
        """Compare all strategies"""
        comparisons = []

        for strategy, trades in self.strategy_trades.items():
            returns = [t.pnl_pct for t in trades]
            equity = self.strategy_equity[strategy]

            comparisons.append({
                'strategy': strategy,
                'n_trades': len(trades),
                'total_pnl': sum(t.pnl for t in trades),
                'win_rate': TradingMetrics.win_rate(trades),
                'profit_factor': TradingMetrics.profit_factor(trades),
                'sharpe': RiskAdjustedMetrics.sharpe_ratio(np.array(returns)) if len(returns) > 1 else 0,
                'max_drawdown': RiskMetrics.max_drawdown(np.array(equity))[0],
                'expectancy': TradingMetrics.expectancy(trades)
            })

        # Sort by Sharpe ratio
        comparisons.sort(key=lambda x: x['sharpe'], reverse=True)

        return comparisons

    def save_report(self, filepath: str):
        """Save performance report to file"""
        report = self.generate_report()
        data = {
            'report': report.to_dict(),
            'model_metrics': self.get_model_metrics(),
            'regime_performance': self.get_regime_performance(),
            'time_analysis': self.get_time_analysis(),
            'strategy_comparison': self.compare_strategies()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance report saved to {filepath}")

    def reset(self):
        """Reset all tracking"""
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []
        self.timestamps = []
        self.predictions = []
        self.strategy_trades = {}
        self.strategy_equity = {}


class RealTimeMonitor:
    """
    Real-time performance monitoring with alerts.
    """

    def __init__(
        self,
        analyzer: PerformanceAnalyzer,
        max_drawdown_alert: float = 0.05,
        min_win_rate_alert: float = 0.3,
        max_consecutive_losses: int = 5
    ):
        self.analyzer = analyzer
        self.max_drawdown_alert = max_drawdown_alert
        self.min_win_rate_alert = min_win_rate_alert
        self.max_consecutive_losses = max_consecutive_losses

        self.alerts: List[Dict[str, Any]] = []
        self.consecutive_losses = 0
        self.last_check_time = time.time()

        logger.info("RealTimeMonitor initialized")

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        new_alerts = []

        # Check drawdown
        equity = np.array(self.analyzer.equity_curve)
        current_dd = 0
        if len(equity) > 1:
            peak = np.max(equity)
            current_dd = (peak - equity[-1]) / peak

            if current_dd > self.max_drawdown_alert:
                new_alerts.append({
                    'type': 'DRAWDOWN',
                    'severity': 'HIGH',
                    'message': f'Current drawdown {current_dd:.2%} exceeds threshold {self.max_drawdown_alert:.2%}',
                    'timestamp': time.time()
                })

        # Check recent win rate
        recent_trades = self.analyzer.trades[-20:] if len(self.analyzer.trades) >= 20 else self.analyzer.trades
        if recent_trades:
            recent_win_rate = TradingMetrics.win_rate(recent_trades)
            if recent_win_rate < self.min_win_rate_alert:
                new_alerts.append({
                    'type': 'WIN_RATE',
                    'severity': 'MEDIUM',
                    'message': f'Recent win rate {recent_win_rate:.2%} below threshold {self.min_win_rate_alert:.2%}',
                    'timestamp': time.time()
                })

        # Check consecutive losses
        if self.analyzer.trades:
            losses = 0
            for trade in reversed(self.analyzer.trades):
                if trade.pnl < 0:
                    losses += 1
                else:
                    break

            if losses >= self.max_consecutive_losses:
                new_alerts.append({
                    'type': 'CONSECUTIVE_LOSSES',
                    'severity': 'HIGH',
                    'message': f'{losses} consecutive losing trades',
                    'timestamp': time.time()
                })

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        equity = self.analyzer.equity_curve
        trades = self.analyzer.trades

        # Current metrics
        current_equity = equity[-1] if equity else self.analyzer.initial_capital
        peak_equity = max(equity) if equity else current_equity
        current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0

        recent_trades = trades[-10:] if len(trades) >= 10 else trades
        recent_win_rate = TradingMetrics.win_rate(recent_trades) if recent_trades else 0

        return {
            'current_equity': current_equity,
            'total_pnl': current_equity - self.analyzer.initial_capital,
            'total_return_pct': (current_equity - self.analyzer.initial_capital) / self.analyzer.initial_capital,
            'current_drawdown': current_dd,
            'total_trades': len(trades),
            'recent_win_rate': recent_win_rate,
            'active_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600]),
            'status': 'OK' if current_dd < self.max_drawdown_alert else 'WARNING'
        }
