"""
Risk Manager - Gerenciamento de risco para HFT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
import logging
from datetime import datetime, date

from ..oms import Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class RiskViolation(Enum):
    """Tipos de violação de risco"""
    MAX_POSITION_SIZE = 'max_position_size'
    MAX_DAILY_LOSS = 'max_daily_loss'
    MAX_DRAWDOWN = 'max_drawdown'
    MAX_ORDERS_PER_SECOND = 'max_orders_per_second'
    MAX_ORDERS_PER_MINUTE = 'max_orders_per_minute'
    INSUFFICIENT_MARGIN = 'insufficient_margin'
    CIRCUIT_BREAKER = 'circuit_breaker'
    SYMBOL_NOT_ALLOWED = 'symbol_not_allowed'
    OUTSIDE_TRADING_HOURS = 'outside_trading_hours'


@dataclass
class RiskLimits:
    """Limites de risco"""
    # Perda máxima
    max_daily_loss_usd: float = 500.0
    max_daily_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0

    # Posições
    max_position_size_lots: float = 1.0
    max_open_positions: int = 3
    risk_per_trade_percent: float = 1.0

    # Ordens
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 60
    max_slippage_pips: float = 2.0

    # Margem
    min_free_margin_percent: float = 50.0
    leverage: int = 500

    # Circuit breaker
    consecutive_losses_limit: int = 5
    cooldown_minutes: int = 30

    # Stop/Take
    stop_loss_pips: float = 20.0
    take_profit_pips: float = 15.0

    @classmethod
    def from_json(cls, filepath: str) -> 'RiskLimits':
        """Carrega limites de arquivo JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        rm = data.get('risk_management', data)
        return cls(
            max_daily_loss_usd=rm.get('max_daily_loss_usd', 500.0),
            max_daily_loss_percent=rm.get('max_daily_loss_percent', 5.0),
            max_drawdown_percent=rm.get('max_drawdown_percent', 10.0),
            max_position_size_lots=rm.get('max_position_size_lots', 1.0),
            max_open_positions=rm.get('max_open_positions', 3),
            risk_per_trade_percent=rm.get('risk_per_trade_percent', 1.0),
            max_orders_per_second=rm.get('max_orders_per_second', 5),
            max_orders_per_minute=rm.get('max_orders_per_minute', 60),
            max_slippage_pips=rm.get('max_slippage_pips', 2.0),
            min_free_margin_percent=rm.get('min_free_margin_percent', 50.0),
            leverage=rm.get('leverage', 500),
            consecutive_losses_limit=rm.get('consecutive_losses_limit', 5),
            cooldown_minutes=rm.get('cooldown_minutes', 30),
            stop_loss_pips=rm.get('stop_loss_pips', 20.0),
            take_profit_pips=rm.get('take_profit_pips', 15.0)
        )


@dataclass
class RiskCheck:
    """Resultado de verificação de risco"""
    passed: bool
    violation: Optional[RiskViolation] = None
    message: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self):
        return self.passed


@dataclass
class DailyStats:
    """Estatísticas diárias"""
    date: date = field(default_factory=date.today)
    starting_balance: float = 0.0
    current_balance: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0


class RiskManager:
    """
    Gerenciador de Risco para HFT

    Funcionalidades:
    - Verificação pre-trade
    - Limites de posição
    - Circuit breaker
    - Tracking de P&L
    - Rate limiting
    """

    def __init__(self, limits: RiskLimits = None, config_path: str = None):
        """
        Inicializa gerenciador de risco

        Args:
            limits: Limites de risco
            config_path: Caminho para arquivo de configuração
        """
        if config_path:
            self.limits = RiskLimits.from_json(config_path)
        else:
            self.limits = limits or RiskLimits()

        # Estado
        self._positions: Dict[str, Position] = {}
        self._daily_stats = DailyStats()
        self._account_balance = 10000.0  # Balanço inicial padrão
        self._free_margin = self._account_balance

        # Rate limiting
        self._order_timestamps: List[float] = []
        self._lock = threading.Lock()

        # Circuit breaker
        self._circuit_breaker_active = False
        self._circuit_breaker_until = 0.0

        # Símbolos permitidos
        self._allowed_symbols: set = {'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'}

        # Callbacks
        self._on_violation: List[Callable[[RiskViolation, str], None]] = []

        logger.info("RiskManager inicializado")

    def register_violation_callback(self, callback: Callable[[RiskViolation, str], None]) -> None:
        """Registra callback para violações"""
        self._on_violation.append(callback)

    def _emit_violation(self, violation: RiskViolation, message: str) -> None:
        """Emite violação"""
        logger.warning(f"Risk violation: {violation.value} - {message}")
        for cb in self._on_violation:
            try:
                cb(violation, message)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")

    def update_account(self, balance: float, free_margin: float) -> None:
        """
        Atualiza informações da conta

        Args:
            balance: Balanço atual
            free_margin: Margem livre
        """
        with self._lock:
            old_balance = self._account_balance
            self._account_balance = balance
            self._free_margin = free_margin

            # Atualizar daily stats
            if self._daily_stats.starting_balance == 0:
                self._daily_stats.starting_balance = balance
                self._daily_stats.peak_balance = balance

            self._daily_stats.current_balance = balance
            self._daily_stats.realized_pnl = balance - self._daily_stats.starting_balance

            # Peak e drawdown
            if balance > self._daily_stats.peak_balance:
                self._daily_stats.peak_balance = balance

            drawdown = (self._daily_stats.peak_balance - balance) / self._daily_stats.peak_balance
            if drawdown > self._daily_stats.max_drawdown:
                self._daily_stats.max_drawdown = drawdown

    def update_position(self, position: Position) -> None:
        """Atualiza posição"""
        with self._lock:
            self._positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove posição"""
        with self._lock:
            self._positions.pop(symbol, None)

    def record_trade_result(self, pnl: float) -> None:
        """
        Registra resultado de trade

        Args:
            pnl: P&L do trade
        """
        with self._lock:
            self._daily_stats.trades_count += 1

            if pnl >= 0:
                self._daily_stats.wins += 1
                self._daily_stats.consecutive_losses = 0
            else:
                self._daily_stats.losses += 1
                self._daily_stats.consecutive_losses += 1

            # Verificar circuit breaker
            if self._daily_stats.consecutive_losses >= self.limits.consecutive_losses_limit:
                self._activate_circuit_breaker()

    def _activate_circuit_breaker(self) -> None:
        """Ativa circuit breaker"""
        self._circuit_breaker_active = True
        self._circuit_breaker_until = time.time() + (self.limits.cooldown_minutes * 60)

        message = f"Circuit breaker ativado por {self.limits.cooldown_minutes} minutos"
        self._emit_violation(RiskViolation.CIRCUIT_BREAKER, message)
        logger.warning(message)

    def check_pre_trade(self, order: Order) -> RiskCheck:
        """
        Verificação de risco pre-trade

        Args:
            order: Ordem a verificar

        Returns:
            Resultado da verificação
        """
        # Circuit breaker
        if self._circuit_breaker_active:
            if time.time() < self._circuit_breaker_until:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.CIRCUIT_BREAKER,
                    message="Circuit breaker ativo"
                )
            else:
                self._circuit_breaker_active = False

        # Símbolo permitido
        if order.symbol not in self._allowed_symbols:
            return RiskCheck(
                passed=False,
                violation=RiskViolation.SYMBOL_NOT_ALLOWED,
                message=f"Símbolo {order.symbol} não permitido"
            )

        # Rate limiting
        rate_check = self._check_rate_limit()
        if not rate_check:
            return rate_check

        # Perda diária
        loss_check = self._check_daily_loss()
        if not loss_check:
            return loss_check

        # Drawdown
        dd_check = self._check_drawdown()
        if not dd_check:
            return dd_check

        # Tamanho de posição
        size_check = self._check_position_size(order)
        if not size_check:
            return size_check

        # Número de posições
        pos_check = self._check_position_count()
        if not pos_check:
            return pos_check

        # Margem
        margin_check = self._check_margin(order)
        if not margin_check:
            return margin_check

        # Registrar ordem
        with self._lock:
            self._order_timestamps.append(time.time())

        return RiskCheck(passed=True, message="Pre-trade check passed")

    def _check_rate_limit(self) -> RiskCheck:
        """Verifica rate limiting"""
        current_time = time.time()

        with self._lock:
            # Limpar timestamps antigos
            self._order_timestamps = [
                t for t in self._order_timestamps
                if current_time - t < 60
            ]

            # Verificar por segundo
            recent_second = sum(1 for t in self._order_timestamps if current_time - t < 1)
            if recent_second >= self.limits.max_orders_per_second:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.MAX_ORDERS_PER_SECOND,
                    message=f"Limite de {self.limits.max_orders_per_second} ordens/segundo"
                )

            # Verificar por minuto
            if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.MAX_ORDERS_PER_MINUTE,
                    message=f"Limite de {self.limits.max_orders_per_minute} ordens/minuto"
                )

        return RiskCheck(passed=True)

    def _check_daily_loss(self) -> RiskCheck:
        """Verifica perda diária"""
        with self._lock:
            daily_loss = -self._daily_stats.realized_pnl if self._daily_stats.realized_pnl < 0 else 0

            # USD
            if daily_loss >= self.limits.max_daily_loss_usd:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.MAX_DAILY_LOSS,
                    message=f"Perda diária {daily_loss:.2f} >= limite {self.limits.max_daily_loss_usd}"
                )

            # Percentual
            if self._daily_stats.starting_balance > 0:
                loss_pct = (daily_loss / self._daily_stats.starting_balance) * 100
                if loss_pct >= self.limits.max_daily_loss_percent:
                    return RiskCheck(
                        passed=False,
                        violation=RiskViolation.MAX_DAILY_LOSS,
                        message=f"Perda diária {loss_pct:.2f}% >= limite {self.limits.max_daily_loss_percent}%"
                    )

        return RiskCheck(passed=True)

    def _check_drawdown(self) -> RiskCheck:
        """Verifica drawdown"""
        with self._lock:
            dd_pct = self._daily_stats.max_drawdown * 100

            if dd_pct >= self.limits.max_drawdown_percent:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.MAX_DRAWDOWN,
                    message=f"Drawdown {dd_pct:.2f}% >= limite {self.limits.max_drawdown_percent}%"
                )

        return RiskCheck(passed=True)

    def _check_position_size(self, order: Order) -> RiskCheck:
        """Verifica tamanho de posição"""
        # Converter quantidade para lots se necessário
        order_lots = order.quantity

        if order_lots > self.limits.max_position_size_lots:
            return RiskCheck(
                passed=False,
                violation=RiskViolation.MAX_POSITION_SIZE,
                message=f"Tamanho {order_lots} > limite {self.limits.max_position_size_lots} lots"
            )

        return RiskCheck(passed=True)

    def _check_position_count(self) -> RiskCheck:
        """Verifica número de posições abertas"""
        with self._lock:
            open_positions = len([p for p in self._positions.values() if p.quantity > 0])

            if open_positions >= self.limits.max_open_positions:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.MAX_POSITION_SIZE,
                    message=f"Posições abertas {open_positions} >= limite {self.limits.max_open_positions}"
                )

        return RiskCheck(passed=True)

    def _check_margin(self, order: Order) -> RiskCheck:
        """Verifica margem disponível"""
        with self._lock:
            # Se não temos info da conta ainda, permitir trade
            if self._account_balance <= 0:
                return RiskCheck(passed=True, message="Sem info de conta, permitindo trade")

            margin_pct = (self._free_margin / self._account_balance) * 100

            if margin_pct < self.limits.min_free_margin_percent:
                return RiskCheck(
                    passed=False,
                    violation=RiskViolation.INSUFFICIENT_MARGIN,
                    message=f"Margem livre {margin_pct:.1f}% < mínimo {self.limits.min_free_margin_percent}%"
                )

        return RiskCheck(passed=True)

    def calculate_position_size(self, symbol: str, stop_loss_pips: float,
                               current_price: float) -> float:
        """
        Calcula tamanho de posição baseado em risco

        Args:
            symbol: Símbolo
            stop_loss_pips: Stop loss em pips
            current_price: Preço atual

        Returns:
            Tamanho em lots
        """
        risk_amount = self._account_balance * (self.limits.risk_per_trade_percent / 100)

        # Pip value aproximado (simplificado)
        pip_value = 10  # Para 1 lot padrão

        if stop_loss_pips > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            position_size = 0.01  # Mínimo

        # Limitar ao máximo
        position_size = min(position_size, self.limits.max_position_size_lots)

        # Arredondar para 2 casas
        return round(position_size, 2)

    def reset_daily_stats(self) -> None:
        """Reseta estatísticas diárias"""
        with self._lock:
            self._daily_stats = DailyStats(
                starting_balance=self._account_balance,
                current_balance=self._account_balance,
                peak_balance=self._account_balance
            )
            self._order_timestamps.clear()
            self._circuit_breaker_active = False

        logger.info("Estatísticas diárias resetadas")

    def add_allowed_symbol(self, symbol: str) -> None:
        """Adiciona símbolo permitido"""
        self._allowed_symbols.add(symbol)

    def remove_allowed_symbol(self, symbol: str) -> None:
        """Remove símbolo permitido"""
        self._allowed_symbols.discard(symbol)

    @property
    def daily_stats(self) -> DailyStats:
        """Estatísticas diárias"""
        return self._daily_stats

    @property
    def is_trading_allowed(self) -> bool:
        """Verifica se trading está permitido"""
        if self._circuit_breaker_active:
            return time.time() >= self._circuit_breaker_until

        return (
            self._check_daily_loss().passed and
            self._check_drawdown().passed
        )

    @property
    def stats(self) -> dict:
        """Estatísticas completas"""
        return {
            'account_balance': self._account_balance,
            'free_margin': self._free_margin,
            'daily_pnl': self._daily_stats.realized_pnl,
            'trades_today': self._daily_stats.trades_count,
            'win_rate': self._daily_stats.wins / max(self._daily_stats.trades_count, 1),
            'consecutive_losses': self._daily_stats.consecutive_losses,
            'max_drawdown_pct': self._daily_stats.max_drawdown * 100,
            'circuit_breaker_active': self._circuit_breaker_active,
            'trading_allowed': self.is_trading_allowed,
            'open_positions': len(self._positions)
        }
