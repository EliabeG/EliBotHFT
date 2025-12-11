"""
Base Strategy - Classe base para estratégias de trading
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

from ..book import OrderBook, Quote

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Tipo de sinal"""
    BUY = 'buy'
    SELL = 'sell'
    CLOSE_LONG = 'close_long'
    CLOSE_SHORT = 'close_short'
    HOLD = 'hold'


class SignalStrength(Enum):
    """Força do sinal"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class Signal:
    """Sinal de trading"""
    signal_type: SignalType
    symbol: str
    price: float
    strength: SignalStrength = SignalStrength.MODERATE
    size: float = 0.0  # 0 = usar tamanho padrão
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    strategy_name: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5  # 0.0 a 1.0

    @property
    def is_entry(self) -> bool:
        """Verifica se é sinal de entrada"""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)

    @property
    def is_exit(self) -> bool:
        """Verifica se é sinal de saída"""
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)


@dataclass
class MarketState:
    """Estado do mercado para a estratégia"""
    symbol: str
    quote: Quote
    book: OrderBook
    timestamp: float = field(default_factory=time.time)

    # Dados adicionais
    tick_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)

    # Indicadores calculados
    volatility: float = 0.0
    momentum: float = 0.0
    trend: float = 0.0  # -1 a 1


@dataclass
class Position:
    """Posição atual"""
    symbol: str
    side: str  # 'long' ou 'short'
    size: float
    entry_price: float
    entry_time: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class BaseStrategy(ABC):
    """
    Classe base abstrata para estratégias de trading

    Todas as estratégias devem herdar desta classe e implementar
    os métodos abstratos.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Inicializa estratégia

        Args:
            name: Nome da estratégia
            config: Configuração da estratégia
        """
        self.name = name
        self.config = config or {}
        self.enabled = True

        # Estado
        self._positions: Dict[str, Position] = {}
        self._last_signal: Optional[Signal] = None
        self._last_signal_time = 0.0

        # Cooldown entre sinais
        self.signal_cooldown_ms = self.config.get('signal_cooldown_ms', 100)

        # Estatísticas
        self._signals_generated = 0
        self._trades_executed = 0
        self._wins = 0
        self._losses = 0

        logger.info(f"Estratégia {name} inicializada")

    @abstractmethod
    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """
        Processa tick de mercado

        Args:
            state: Estado atual do mercado

        Returns:
            Sinal de trading ou None
        """
        pass

    @abstractmethod
    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """
        Processa atualização de cotação

        Args:
            quote: Nova cotação

        Returns:
            Sinal de trading ou None
        """
        pass

    def on_fill(self, order_id: str, fill_price: float, fill_size: float) -> None:
        """
        Callback quando ordem é executada

        Args:
            order_id: ID da ordem
            fill_price: Preço de execução
            fill_size: Volume executado
        """
        logger.info(f"{self.name}: Fill order {order_id} @ {fill_price} x {fill_size}")

    def on_position_update(self, position: Position) -> None:
        """
        Callback quando posição é atualizada

        Args:
            position: Nova posição
        """
        self._positions[position.symbol] = position

    def on_trade_closed(self, symbol: str, pnl: float) -> None:
        """
        Callback quando trade é fechado

        Args:
            symbol: Símbolo
            pnl: P&L realizado
        """
        self._trades_executed += 1
        if pnl >= 0:
            self._wins += 1
        else:
            self._losses += 1

        logger.info(f"{self.name}: Trade fechado {symbol}, PnL: {pnl:.2f}")

    def can_generate_signal(self) -> bool:
        """Verifica se pode gerar novo sinal (cooldown)"""
        elapsed = (time.time() - self._last_signal_time) * 1000
        return elapsed >= self.signal_cooldown_ms

    def _emit_signal(self, signal: Signal) -> Signal:
        """Emite sinal e atualiza estado"""
        signal.strategy_name = self.name
        self._last_signal = signal
        self._last_signal_time = time.time()
        self._signals_generated += 1

        logger.debug(f"{self.name}: Sinal {signal.signal_type.value} @ {signal.price}")
        return signal

    def get_position(self, symbol: str) -> Optional[Position]:
        """Obtém posição atual"""
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Verifica se há posição aberta"""
        pos = self._positions.get(symbol)
        return pos is not None and pos.size > 0

    def update_config(self, config: Dict[str, Any]) -> None:
        """Atualiza configuração em tempo real"""
        self.config.update(config)
        logger.info(f"{self.name}: Configuração atualizada")

    def enable(self) -> None:
        """Habilita estratégia"""
        self.enabled = True
        logger.info(f"{self.name}: Habilitada")

    def disable(self) -> None:
        """Desabilita estratégia"""
        self.enabled = False
        logger.info(f"{self.name}: Desabilitada")

    @property
    def win_rate(self) -> float:
        """Taxa de acerto"""
        total = self._wins + self._losses
        if total > 0:
            return self._wins / total
        return 0.0

    @property
    def stats(self) -> dict:
        """Estatísticas da estratégia"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'signals_generated': self._signals_generated,
            'trades_executed': self._trades_executed,
            'wins': self._wins,
            'losses': self._losses,
            'win_rate': self.win_rate
        }

    def reset_stats(self) -> None:
        """Reseta estatísticas"""
        self._signals_generated = 0
        self._trades_executed = 0
        self._wins = 0
        self._losses = 0
