"""
Advanced Risk Manager - Sistema Avançado de Gerenciamento de Risco para HFT
Inclui VaR, Kelly Criterion, Sizing Dinâmico e Circuit Breakers Adaptativos
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import time
import threading
import logging
import math
from datetime import datetime, date, timedelta
import numpy as np

from ..oms import Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Níveis de risco do sistema"""
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    HIGH = 'high'
    CRITICAL = 'critical'
    HALT = 'halt'


class AdaptiveAction(Enum):
    """Ações adaptativas baseadas em risco"""
    NONE = 'none'
    REDUCE_SIZE = 'reduce_size'
    WIDEN_STOPS = 'widen_stops'
    PAUSE_ENTRIES = 'pause_entries'
    CLOSE_POSITIONS = 'close_positions'
    FULL_HALT = 'full_halt'


@dataclass
class VaRMetrics:
    """Métricas de Value at Risk"""
    var_95: float = 0.0  # VaR 95%
    var_99: float = 0.0  # VaR 99%
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0
    current_exposure: float = 0.0
    max_var_limit: float = 0.0
    var_utilization: float = 0.0  # % do limite usado


@dataclass
class VolatilityMetrics:
    """Métricas de volatilidade"""
    current: float = 0.0
    fast_ema: float = 0.0  # EMA rápida (para detecção de spikes)
    slow_ema: float = 0.0  # EMA lenta (baseline)
    atr: float = 0.0
    atr_percentile: float = 50.0
    regime: str = 'normal'  # low, normal, high, extreme
    z_score: float = 0.0


@dataclass
class DrawdownMetrics:
    """Métricas de drawdown"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration_seconds: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class PerformanceMetrics:
    """Métricas de performance para adaptive sizing"""
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    kelly_fraction: float = 0.0
    optimal_f: float = 0.0
    expectancy: float = 0.0


@dataclass
class TradeRecord:
    """Registro de trade para análise"""
    timestamp: float
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pips: float
    holding_time_ms: float
    mae_pips: float  # Maximum Adverse Excursion
    mfe_pips: float  # Maximum Favorable Excursion
    strategy: str


@dataclass
class AdvancedRiskLimits:
    """Limites de risco avançados"""
    # Limites de VaR
    max_var_95_percent: float = 2.0  # Max 2% VaR diário
    max_var_99_percent: float = 3.0

    # Limites de Drawdown
    max_daily_drawdown_percent: float = 3.0
    max_weekly_drawdown_percent: float = 5.0
    max_total_drawdown_percent: float = 10.0
    drawdown_recovery_factor: float = 0.5  # Reduz size após DD

    # Limites de Posição
    max_position_lots: float = 1.0
    max_notional_usd: float = 100000.0
    max_open_positions: int = 3
    max_correlated_exposure: float = 1.5  # Max 1.5x em pares correlacionados

    # Limites de Volatilidade
    max_volatility_percentile: float = 90.0  # Pausa em volatilidade extrema
    volatility_scaling: bool = True  # Escalar posição por volatilidade

    # Kelly e Sizing
    kelly_fraction_limit: float = 0.25  # Max 25% Kelly
    min_position_size: float = 0.01
    position_size_step: float = 0.01

    # Rate Limiting
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 100
    max_daily_trades: int = 500

    # Circuit Breakers
    consecutive_losses_limit: int = 5
    cooldown_minutes_base: int = 15
    cooldown_multiplier: float = 1.5  # Aumenta cooldown a cada trigger

    # Slippage
    max_slippage_pips: float = 2.0
    slippage_monitoring_window: int = 20

    # Session Limits
    session_max_loss_percent: float = 2.0
    session_profit_lock_percent: float = 3.0  # Trava lucro após 3%


class AdvancedRiskManager:
    """
    Sistema Avançado de Gerenciamento de Risco para HFT

    Funcionalidades:
    - Value at Risk (VaR) em tempo real
    - Kelly Criterion para position sizing
    - Volatility-based sizing
    - Circuit breakers adaptativos
    - Tracking de MAE/MFE
    - Regime detection
    - Correlation-based limits
    """

    def __init__(self, limits: AdvancedRiskLimits = None, config_path: str = None):
        """
        Inicializa sistema de risco avançado

        Args:
            limits: Limites de risco
            config_path: Caminho para arquivo de configuração
        """
        if config_path:
            self.limits = self._load_config(config_path)
        else:
            self.limits = limits or AdvancedRiskLimits()

        # Estado da conta
        self._account_balance = 10000.0
        self._initial_balance = 10000.0
        self._equity = 10000.0
        self._free_margin = 10000.0

        # Métricas
        self._var = VaRMetrics()
        self._volatility = VolatilityMetrics()
        self._drawdown = DrawdownMetrics()
        self._performance = PerformanceMetrics()

        # Históricos
        self._trade_history: deque = deque(maxlen=500)
        self._pnl_history: deque = deque(maxlen=100)
        self._return_history: deque = deque(maxlen=252)  # ~1 ano de dias
        self._price_history: Dict[str, deque] = {}
        self._volatility_history: deque = deque(maxlen=100)

        # Posições
        self._positions: Dict[str, Position] = {}
        self._position_mae: Dict[str, float] = {}  # MAE por posição
        self._position_mfe: Dict[str, float] = {}  # MFE por posição

        # Rate limiting
        self._order_timestamps: List[float] = []
        self._daily_trade_count = 0
        self._daily_trade_date = date.today()

        # Circuit breaker state
        self._risk_level = RiskLevel.NORMAL
        self._circuit_breaker_active = False
        self._circuit_breaker_until = 0.0
        self._circuit_breaker_count = 0  # Número de vezes ativado
        self._consecutive_losses = 0
        self._last_trade_pnl = 0.0

        # Session tracking
        self._session_start_balance = 0.0
        self._session_high_equity = 0.0
        self._session_realized_pnl = 0.0
        self._profit_locked = False

        # Threading
        self._lock = threading.RLock()

        # Callbacks
        self._on_risk_level_change: List[Callable[[RiskLevel], None]] = []
        self._on_adaptive_action: List[Callable[[AdaptiveAction, str], None]] = []

        logger.info("AdvancedRiskManager initialized")

    def _load_config(self, filepath: str) -> AdvancedRiskLimits:
        """Carrega configuração de arquivo JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            rm = data.get('advanced_risk', data)
            return AdvancedRiskLimits(
                max_var_95_percent=rm.get('max_var_95_percent', 2.0),
                max_var_99_percent=rm.get('max_var_99_percent', 3.0),
                max_daily_drawdown_percent=rm.get('max_daily_drawdown_percent', 3.0),
                max_weekly_drawdown_percent=rm.get('max_weekly_drawdown_percent', 5.0),
                max_total_drawdown_percent=rm.get('max_total_drawdown_percent', 10.0),
                max_position_lots=rm.get('max_position_lots', 1.0),
                max_open_positions=rm.get('max_open_positions', 3),
                kelly_fraction_limit=rm.get('kelly_fraction_limit', 0.25),
                consecutive_losses_limit=rm.get('consecutive_losses_limit', 5),
                cooldown_minutes_base=rm.get('cooldown_minutes_base', 15),
                max_daily_trades=rm.get('max_daily_trades', 500)
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return AdvancedRiskLimits()

    # ==================== ACCOUNT UPDATES ====================

    def update_account(self, balance: float, equity: float, free_margin: float) -> None:
        """Atualiza informações da conta"""
        with self._lock:
            if self._initial_balance == 0:
                self._initial_balance = balance
                self._session_start_balance = balance

            old_equity = self._equity
            self._account_balance = balance
            self._equity = equity
            self._free_margin = free_margin

            # Track session
            if self._session_start_balance == 0:
                self._session_start_balance = balance

            if equity > self._session_high_equity:
                self._session_high_equity = equity

            # Update drawdown
            self._update_drawdown()

            # Check profit lock
            self._check_profit_lock()

            # Update risk level
            self._evaluate_risk_level()

    def update_position(self, position: Position, current_price: float) -> None:
        """Atualiza posição e rastreia MAE/MFE"""
        with self._lock:
            symbol = position.symbol
            self._positions[symbol] = position

            # Calcular P&L não realizado em pips
            pip_size = 0.0001 if 'JPY' not in symbol else 0.01

            if position.side == 'long':
                pnl_pips = (current_price - position.entry_price) / pip_size
            else:
                pnl_pips = (position.entry_price - current_price) / pip_size

            # Update MAE (Maximum Adverse Excursion)
            if symbol not in self._position_mae:
                self._position_mae[symbol] = 0.0
            if pnl_pips < self._position_mae[symbol]:
                self._position_mae[symbol] = pnl_pips

            # Update MFE (Maximum Favorable Excursion)
            if symbol not in self._position_mfe:
                self._position_mfe[symbol] = 0.0
            if pnl_pips > self._position_mfe[symbol]:
                self._position_mfe[symbol] = pnl_pips

    def remove_position(self, symbol: str) -> Tuple[float, float]:
        """Remove posição e retorna MAE/MFE"""
        with self._lock:
            self._positions.pop(symbol, None)
            mae = self._position_mae.pop(symbol, 0.0)
            mfe = self._position_mfe.pop(symbol, 0.0)
            return mae, mfe

    # ==================== TRADE RECORDING ====================

    def record_trade(self, trade: TradeRecord) -> None:
        """Registra trade completado"""
        with self._lock:
            self._trade_history.append(trade)
            self._pnl_history.append(trade.pnl)

            # Daily count
            if date.today() != self._daily_trade_date:
                self._daily_trade_count = 0
                self._daily_trade_date = date.today()
            self._daily_trade_count += 1

            # Session P&L
            self._session_realized_pnl += trade.pnl

            # Consecutive losses tracking
            self._last_trade_pnl = trade.pnl
            if trade.pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            # Check circuit breaker
            if self._consecutive_losses >= self.limits.consecutive_losses_limit:
                self._trigger_circuit_breaker()

            # Update performance metrics
            self._update_performance_metrics()

            # Add to return history
            if self._account_balance > 0:
                ret = trade.pnl / self._account_balance
                self._return_history.append(ret)

            # Update VaR
            self._update_var()

    def record_trade_result(self, symbol: str, side: str, entry_price: float,
                           exit_price: float, pnl: float, holding_time_ms: float,
                           strategy: str = '') -> None:
        """Versão simplificada para registrar resultado de trade"""
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01

        if side == 'long':
            pnl_pips = (exit_price - entry_price) / pip_size
        else:
            pnl_pips = (entry_price - exit_price) / pip_size

        mae, mfe = self.remove_position(symbol)

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pips=pnl_pips,
            holding_time_ms=holding_time_ms,
            mae_pips=mae,
            mfe_pips=mfe,
            strategy=strategy
        )

        self.record_trade(trade)

    # ==================== RISK CALCULATIONS ====================

    def _update_drawdown(self) -> None:
        """Atualiza métricas de drawdown"""
        if self._drawdown.peak_equity == 0:
            self._drawdown.peak_equity = self._equity

        # Update peak
        if self._equity > self._drawdown.peak_equity:
            self._drawdown.peak_equity = self._equity
            self._drawdown.drawdown_duration_seconds = 0

        # Current drawdown
        if self._drawdown.peak_equity > 0:
            self._drawdown.current_drawdown = (
                (self._drawdown.peak_equity - self._equity) / self._drawdown.peak_equity
            ) * 100

        # Max drawdown
        if self._drawdown.current_drawdown > self._drawdown.max_drawdown:
            self._drawdown.max_drawdown = self._drawdown.current_drawdown
            self._drawdown.trough_equity = self._equity

        # Recovery factor
        total_profit = sum(t.pnl for t in self._trade_history if t.pnl > 0)
        if self._drawdown.max_drawdown > 0 and self._initial_balance > 0:
            max_dd_usd = (self._drawdown.max_drawdown / 100) * self._initial_balance
            if max_dd_usd > 0:
                self._drawdown.recovery_factor = total_profit / max_dd_usd

    def _update_var(self) -> None:
        """Calcula Value at Risk"""
        if len(self._return_history) < 20:
            return

        returns = np.array(list(self._return_history))

        # VaR paramétrico (assumindo distribuição normal)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        # Z-scores para níveis de confiança
        z_95 = 1.645
        z_99 = 2.326

        # VaR como percentual do portfolio
        self._var.var_95 = abs(mean_ret - z_95 * std_ret) * 100
        self._var.var_99 = abs(mean_ret - z_99 * std_ret) * 100

        # CVaR (Expected Shortfall) - média das perdas além do VaR
        sorted_returns = np.sort(returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_99_idx = int(len(sorted_returns) * 0.01)

        if var_95_idx > 0:
            self._var.cvar_95 = abs(np.mean(sorted_returns[:var_95_idx])) * 100
        if var_99_idx > 0:
            self._var.cvar_99 = abs(np.mean(sorted_returns[:var_99_idx])) * 100

        # VaR utilization
        self._var.max_var_limit = self.limits.max_var_95_percent
        if self._var.max_var_limit > 0:
            self._var.var_utilization = (self._var.var_95 / self._var.max_var_limit) * 100

    def _update_performance_metrics(self) -> None:
        """Atualiza métricas de performance"""
        if len(self._trade_history) < 10:
            return

        trades = list(self._trade_history)

        # Win rate
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        if trades:
            self._performance.win_rate = len(wins) / len(trades)

        # Average win/loss
        if wins:
            self._performance.avg_win = np.mean([t.pnl for t in wins])
        if losses:
            self._performance.avg_loss = abs(np.mean([t.pnl for t in losses]))

        # Profit factor
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        if total_losses > 0:
            self._performance.profit_factor = total_wins / total_losses

        # Expectancy
        self._performance.expectancy = (
            self._performance.win_rate * self._performance.avg_win -
            (1 - self._performance.win_rate) * self._performance.avg_loss
        )

        # Kelly Criterion
        self._calculate_kelly()

        # Sharpe Ratio (anualizado)
        if len(self._return_history) >= 20:
            returns = np.array(list(self._return_history))
            if np.std(returns) > 0:
                # Assumindo ~252 dias de trading
                self._performance.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )

    def _calculate_kelly(self) -> None:
        """Calcula Kelly Criterion"""
        w = self._performance.win_rate
        r = self._performance.avg_win / self._performance.avg_loss if self._performance.avg_loss > 0 else 1

        # Kelly = W - (1-W)/R
        if r > 0:
            kelly = w - ((1 - w) / r)
            # Limitar Kelly
            self._performance.kelly_fraction = max(0, min(kelly, self.limits.kelly_fraction_limit))

            # Optimal f (mais conservador)
            self._performance.optimal_f = self._performance.kelly_fraction * 0.5

    # ==================== VOLATILITY ====================

    def update_volatility(self, symbol: str, price: float) -> None:
        """Atualiza métricas de volatilidade"""
        with self._lock:
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=100)

            self._price_history[symbol].append(price)

            if len(self._price_history[symbol]) < 20:
                return

            prices = np.array(list(self._price_history[symbol]))
            returns = np.diff(prices) / prices[:-1]

            # Volatilidade atual (desvio padrão dos retornos)
            self._volatility.current = np.std(returns) * 100

            # EMAs de volatilidade
            alpha_fast = 2 / (10 + 1)
            alpha_slow = 2 / (50 + 1)

            if self._volatility.fast_ema == 0:
                self._volatility.fast_ema = self._volatility.current
                self._volatility.slow_ema = self._volatility.current
            else:
                self._volatility.fast_ema = alpha_fast * self._volatility.current + (1 - alpha_fast) * self._volatility.fast_ema
                self._volatility.slow_ema = alpha_slow * self._volatility.current + (1 - alpha_slow) * self._volatility.slow_ema

            # ATR simplificado
            if len(prices) >= 14:
                high_low = np.max(prices[-14:]) - np.min(prices[-14:])
                self._volatility.atr = high_low

            # Histórico para percentil
            self._volatility_history.append(self._volatility.current)

            # Percentil
            if len(self._volatility_history) >= 20:
                sorted_vol = sorted(self._volatility_history)
                idx = sorted_vol.index(self._volatility.current) if self._volatility.current in sorted_vol else len(sorted_vol) // 2
                self._volatility.atr_percentile = (idx / len(sorted_vol)) * 100

            # Regime detection
            self._detect_volatility_regime()

            # Z-score da volatilidade
            if len(self._volatility_history) >= 20:
                mean_vol = np.mean(list(self._volatility_history))
                std_vol = np.std(list(self._volatility_history))
                if std_vol > 0:
                    self._volatility.z_score = (self._volatility.current - mean_vol) / std_vol

    def _detect_volatility_regime(self) -> None:
        """Detecta regime de volatilidade"""
        percentile = self._volatility.atr_percentile

        if percentile < 20:
            self._volatility.regime = 'low'
        elif percentile < 60:
            self._volatility.regime = 'normal'
        elif percentile < 90:
            self._volatility.regime = 'high'
        else:
            self._volatility.regime = 'extreme'

    # ==================== RISK LEVEL & CIRCUIT BREAKER ====================

    def _evaluate_risk_level(self) -> None:
        """Avalia nível de risco atual"""
        old_level = self._risk_level

        # Check multiple risk factors
        risk_score = 0

        # Drawdown
        dd_pct = self._drawdown.current_drawdown
        if dd_pct > self.limits.max_daily_drawdown_percent:
            risk_score += 3
        elif dd_pct > self.limits.max_daily_drawdown_percent * 0.7:
            risk_score += 2
        elif dd_pct > self.limits.max_daily_drawdown_percent * 0.5:
            risk_score += 1

        # VaR utilization
        if self._var.var_utilization > 100:
            risk_score += 2
        elif self._var.var_utilization > 80:
            risk_score += 1

        # Volatility regime
        if self._volatility.regime == 'extreme':
            risk_score += 2
        elif self._volatility.regime == 'high':
            risk_score += 1

        # Consecutive losses
        if self._consecutive_losses >= self.limits.consecutive_losses_limit:
            risk_score += 3
        elif self._consecutive_losses >= self.limits.consecutive_losses_limit - 2:
            risk_score += 1

        # Circuit breaker
        if self._circuit_breaker_active:
            risk_score += 5

        # Determine level
        if risk_score >= 6 or self._circuit_breaker_active:
            self._risk_level = RiskLevel.HALT
        elif risk_score >= 4:
            self._risk_level = RiskLevel.CRITICAL
        elif risk_score >= 3:
            self._risk_level = RiskLevel.HIGH
        elif risk_score >= 1:
            self._risk_level = RiskLevel.ELEVATED
        else:
            self._risk_level = RiskLevel.NORMAL

        # Notify if changed
        if self._risk_level != old_level:
            logger.warning(f"Risk level changed: {old_level.value} -> {self._risk_level.value}")
            for cb in self._on_risk_level_change:
                try:
                    cb(self._risk_level)
                except Exception as e:
                    logger.error(f"Error in risk level callback: {e}")

    def _trigger_circuit_breaker(self) -> None:
        """Ativa circuit breaker"""
        self._circuit_breaker_count += 1

        # Cooldown aumenta a cada trigger
        cooldown = self.limits.cooldown_minutes_base * (
            self.limits.cooldown_multiplier ** (self._circuit_breaker_count - 1)
        )

        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + (cooldown * 60)

        logger.warning(f"Circuit breaker activated! Cooldown: {cooldown:.0f} minutes")

        # Emit action
        for cb in self._on_adaptive_action:
            try:
                cb(AdaptiveAction.FULL_HALT, f"Circuit breaker #{self._circuit_breaker_count}")
            except Exception as e:
                logger.error(f"Error in action callback: {e}")

    def _check_profit_lock(self) -> None:
        """Verifica e ativa profit lock"""
        if self._session_start_balance > 0:
            session_return = (
                (self._equity - self._session_start_balance) / self._session_start_balance
            ) * 100

            if session_return >= self.limits.session_profit_lock_percent:
                self._profit_locked = True
                logger.info(f"Profit locked at {session_return:.2f}%")

    # ==================== PRE-TRADE CHECKS ====================

    def check_pre_trade(self, order: Order) -> Tuple[bool, str]:
        """
        Verificação de risco pre-trade completa

        Returns:
            Tuple (passed, message)
        """
        # Circuit breaker check
        if self._circuit_breaker_active:
            if time.time() < self._circuit_breaker_until:
                remaining = (self._circuit_breaker_until - time.time()) / 60
                return False, f"Circuit breaker active. Remaining: {remaining:.1f} min"
            else:
                self._circuit_breaker_active = False
                logger.info("Circuit breaker deactivated")

        # Risk level check
        if self._risk_level == RiskLevel.HALT:
            return False, "Risk level: HALT - No trading allowed"

        if self._risk_level == RiskLevel.CRITICAL and order.quantity > self.limits.min_position_size:
            return False, "Risk level: CRITICAL - Only minimum size allowed"

        # Profit lock check
        if self._profit_locked:
            return False, "Profit locked - No new entries today"

        # Drawdown check
        if self._drawdown.current_drawdown >= self.limits.max_daily_drawdown_percent:
            return False, f"Max daily drawdown reached: {self._drawdown.current_drawdown:.2f}%"

        # VaR check
        if self._var.var_utilization > 100:
            return False, f"VaR limit exceeded: {self._var.var_utilization:.1f}%"

        # Volatility check
        if self._volatility.atr_percentile > self.limits.max_volatility_percentile:
            return False, f"Volatility too high: {self._volatility.atr_percentile:.1f} percentile"

        # Position count check
        open_positions = len([p for p in self._positions.values() if p.size > 0])
        if open_positions >= self.limits.max_open_positions:
            return False, f"Max positions reached: {open_positions}"

        # Position size check
        if order.quantity > self.limits.max_position_lots:
            return False, f"Position size {order.quantity} exceeds max {self.limits.max_position_lots}"

        # Rate limiting
        rate_ok, rate_msg = self._check_rate_limit()
        if not rate_ok:
            return False, rate_msg

        # Daily trade limit
        if self._daily_trade_count >= self.limits.max_daily_trades:
            return False, f"Daily trade limit reached: {self._daily_trade_count}"

        # Register order timestamp
        self._order_timestamps.append(time.time())

        return True, "Pre-trade check passed"

    def _check_rate_limit(self) -> Tuple[bool, str]:
        """Verifica rate limiting"""
        current_time = time.time()

        # Clean old timestamps
        self._order_timestamps = [t for t in self._order_timestamps if current_time - t < 60]

        # Per second
        recent_second = sum(1 for t in self._order_timestamps if current_time - t < 1)
        if recent_second >= self.limits.max_orders_per_second:
            return False, f"Rate limit: {self.limits.max_orders_per_second}/sec exceeded"

        # Per minute
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, f"Rate limit: {self.limits.max_orders_per_minute}/min exceeded"

        return True, "OK"

    # ==================== POSITION SIZING ====================

    def calculate_position_size(self, symbol: str, stop_loss_pips: float,
                                 risk_per_trade_percent: float = None) -> float:
        """
        Calcula tamanho de posição otimizado

        Considera:
        - Risk per trade
        - Kelly criterion
        - Volatility scaling
        - Drawdown adjustment
        - Risk level adjustment
        """
        with self._lock:
            if risk_per_trade_percent is None:
                risk_per_trade_percent = 1.0  # Default 1%

            # Base position from risk
            risk_amount = self._account_balance * (risk_per_trade_percent / 100)

            pip_value = 10  # Aproximado para 1 lot standard
            pip_size = 0.0001 if 'JPY' not in symbol else 0.01

            if stop_loss_pips > 0:
                base_size = risk_amount / (stop_loss_pips * pip_value)
            else:
                base_size = self.limits.min_position_size

            # Kelly adjustment
            if self._performance.kelly_fraction > 0:
                kelly_size = self._account_balance * self._performance.optimal_f / (stop_loss_pips * pip_value)
                # Use menor entre risk-based e kelly
                base_size = min(base_size, kelly_size)

            # Volatility scaling
            if self.limits.volatility_scaling and self._volatility.slow_ema > 0:
                vol_ratio = self._volatility.slow_ema / self._volatility.current if self._volatility.current > 0 else 1
                vol_ratio = max(0.5, min(1.5, vol_ratio))  # Limitar ajuste
                base_size *= vol_ratio

            # Drawdown adjustment
            if self._drawdown.current_drawdown > 0:
                dd_factor = 1 - (self._drawdown.current_drawdown / self.limits.max_daily_drawdown_percent) * self.limits.drawdown_recovery_factor
                dd_factor = max(0.25, dd_factor)  # Mínimo 25%
                base_size *= dd_factor

            # Risk level adjustment
            if self._risk_level == RiskLevel.HIGH:
                base_size *= 0.5
            elif self._risk_level == RiskLevel.ELEVATED:
                base_size *= 0.75

            # Apply limits
            base_size = max(self.limits.min_position_size, base_size)
            base_size = min(self.limits.max_position_lots, base_size)

            # Round to step
            base_size = round(base_size / self.limits.position_size_step) * self.limits.position_size_step

            return base_size

    # ==================== CALLBACKS ====================

    def register_risk_level_callback(self, callback: Callable[[RiskLevel], None]) -> None:
        """Registra callback para mudanças de nível de risco"""
        self._on_risk_level_change.append(callback)

    def register_action_callback(self, callback: Callable[[AdaptiveAction, str], None]) -> None:
        """Registra callback para ações adaptativas"""
        self._on_adaptive_action.append(callback)

    # ==================== RESET ====================

    def reset_daily(self) -> None:
        """Reset estatísticas diárias"""
        with self._lock:
            self._consecutive_losses = 0
            self._daily_trade_count = 0
            self._daily_trade_date = date.today()
            self._session_start_balance = self._equity
            self._session_high_equity = self._equity
            self._session_realized_pnl = 0.0
            self._profit_locked = False
            self._drawdown.drawdown_duration_seconds = 0

            # Reset circuit breaker count after day
            self._circuit_breaker_count = max(0, self._circuit_breaker_count - 1)

            logger.info("Daily risk stats reset")

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker manualmente"""
        with self._lock:
            self._circuit_breaker_active = False
            self._circuit_breaker_until = 0
            self._consecutive_losses = 0
            self._risk_level = RiskLevel.NORMAL
            logger.info("Circuit breaker reset manually")

    # ==================== PROPERTIES ====================

    @property
    def risk_level(self) -> RiskLevel:
        """Nível de risco atual"""
        return self._risk_level

    @property
    def is_trading_allowed(self) -> bool:
        """Verifica se trading é permitido"""
        return (
            self._risk_level not in (RiskLevel.HALT, RiskLevel.CRITICAL) and
            not self._circuit_breaker_active and
            not self._profit_locked
        )

    @property
    def var_metrics(self) -> VaRMetrics:
        """Métricas de VaR"""
        return self._var

    @property
    def volatility_metrics(self) -> VolatilityMetrics:
        """Métricas de volatilidade"""
        return self._volatility

    @property
    def drawdown_metrics(self) -> DrawdownMetrics:
        """Métricas de drawdown"""
        return self._drawdown

    @property
    def performance_metrics(self) -> PerformanceMetrics:
        """Métricas de performance"""
        return self._performance

    @property
    def stats(self) -> dict:
        """Estatísticas completas"""
        return {
            'account': {
                'balance': self._account_balance,
                'equity': self._equity,
                'free_margin': self._free_margin
            },
            'risk_level': self._risk_level.value,
            'circuit_breaker_active': self._circuit_breaker_active,
            'trading_allowed': self.is_trading_allowed,
            'var': {
                'var_95': self._var.var_95,
                'var_99': self._var.var_99,
                'cvar_95': self._var.cvar_95,
                'utilization': self._var.var_utilization
            },
            'volatility': {
                'current': self._volatility.current,
                'regime': self._volatility.regime,
                'percentile': self._volatility.atr_percentile
            },
            'drawdown': {
                'current': self._drawdown.current_drawdown,
                'max': self._drawdown.max_drawdown,
                'recovery_factor': self._drawdown.recovery_factor
            },
            'performance': {
                'win_rate': self._performance.win_rate,
                'profit_factor': self._performance.profit_factor,
                'sharpe_ratio': self._performance.sharpe_ratio,
                'kelly_fraction': self._performance.kelly_fraction,
                'expectancy': self._performance.expectancy
            },
            'session': {
                'realized_pnl': self._session_realized_pnl,
                'trade_count': self._daily_trade_count,
                'consecutive_losses': self._consecutive_losses,
                'profit_locked': self._profit_locked
            }
        }
