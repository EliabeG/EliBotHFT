"""
Order Flow Strategy - Estratégia baseada em Fluxo de Ordens para HFT
Análise de microestrutura de mercado usando Level 2 data
"""

from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote, OrderBook, BookSide

logger = logging.getLogger(__name__)


class OrderFlowSignal(Enum):
    """Tipo de sinal de order flow"""
    ABSORPTION = 'absorption'  # Grande ordem absorvendo liquidez
    IMBALANCE = 'imbalance'  # Desequilíbrio bid/ask
    STACKING = 'stacking'  # Ordens empilhadas em um nível
    PULLING = 'pulling'  # Ordens sendo canceladas
    SWEEPING = 'sweep'  # Varredura de múltiplos níveis
    ICEBERG = 'iceberg'  # Detecção de ordem iceberg


@dataclass
class OrderFlowMetrics:
    """Métricas de fluxo de ordens"""
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    imbalance: float = 0.0
    delta: float = 0.0  # Diferença entre compras e vendas
    cumulative_delta: float = 0.0
    absorption_ratio: float = 0.0
    aggressor_ratio: float = 0.0  # Ratio de market orders
    large_order_detected: bool = False
    sweep_detected: bool = False


@dataclass
class VolumeProfile:
    """Perfil de volume por nível de preço"""
    price: float
    volume: float
    buy_volume: float
    sell_volume: float
    order_count: int
    is_poc: bool = False  # Point of Control


class OrderFlowStrategy(BaseStrategy):
    """
    Estratégia de Order Flow para HFT

    Características:
    - Análise de imbalance em tempo real
    - Detecção de absorção de ordens
    - Tracking de delta cumulativo
    - Detecção de iceberg orders
    - Identificação de níveis de volume
    - Otimizada para EURUSD
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('OrderFlow', config)

        # Parâmetros de imbalance
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.4)
        self.strong_imbalance_threshold = self.config.get('strong_imbalance_threshold', 0.6)
        self.imbalance_levels = self.config.get('imbalance_levels', 5)

        # Parâmetros de delta
        self.delta_threshold = self.config.get('delta_threshold', 0.3)
        self.cumulative_delta_period = self.config.get('cumulative_delta_period', 50)

        # Parâmetros de detecção de ordens grandes
        self.large_order_multiplier = self.config.get('large_order_multiplier', 3.0)
        self.sweep_detection_levels = self.config.get('sweep_detection_levels', 3)

        # Pip configuration
        self.pip_size = self.config.get('pip_size', 0.0001)

        # Parâmetros de risco
        self.stop_loss_pips = self.config.get('stop_loss_pips', 8)
        self.take_profit_pips = self.config.get('take_profit_pips', 5)
        self.trailing_stop_pips = self.config.get('trailing_stop_pips', 3)

        # Filtros
        self.min_book_depth = self.config.get('min_book_depth', 5)
        self.max_spread_pips = self.config.get('max_spread_pips', 1.5)
        self.confirmation_ticks = self.config.get('confirmation_ticks', 2)

        # Históricos
        self._imbalance_history: deque = deque(maxlen=50)
        self._delta_history: deque = deque(maxlen=self.cumulative_delta_period)
        self._bid_volume_history: deque = deque(maxlen=50)
        self._ask_volume_history: deque = deque(maxlen=50)
        self._trade_flow_history: deque = deque(maxlen=100)
        self._book_snapshots: deque = deque(maxlen=10)

        # Estado
        self._metrics = OrderFlowMetrics()
        self._volume_profile: Dict[float, VolumeProfile] = {}
        self._avg_bid_volume = 0.0
        self._avg_ask_volume = 0.0
        self._signal_confirmation_count = 0
        self._pending_signal_type: Optional[SignalType] = None

        # Trailing stop tracking
        self._trailing_high: Dict[str, float] = {}
        self._trailing_low: Dict[str, float] = {}

        # Métricas de performance
        self._signals_by_type: Dict[str, int] = {
            'imbalance': 0,
            'absorption': 0,
            'sweep': 0,
            'delta': 0
        }

        logger.info(f"OrderFlow strategy initialized")

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e analisa fluxo de ordens"""
        if not self.enabled:
            return None

        quote = state.quote
        book = state.book
        symbol = state.symbol

        # Validar dados
        if not self._validate_market_data(quote, book):
            return None

        # Atualizar métricas de order flow
        self._update_order_flow_metrics(quote, book)

        # Atualizar trailing stops
        self._update_trailing_stops(symbol, quote)

        # Verificar saída primeiro
        exit_signal = self._check_exit_conditions(symbol, quote)
        if exit_signal:
            return exit_signal

        # Processar sinal pendente
        if self._pending_signal_type:
            return self._process_pending_signal(symbol, quote)

        # Analisar sinais de order flow
        return self._analyze_order_flow(symbol, quote, book)

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa atualização de cotação"""
        return None

    def _validate_market_data(self, quote: Quote, book: OrderBook) -> bool:
        """Valida qualidade dos dados de mercado"""
        if quote.bid_price <= 0 or quote.ask_price <= 0:
            return False

        spread_pips = quote.spread / self.pip_size
        if spread_pips > self.max_spread_pips:
            return False

        if book.bid_depth < self.min_book_depth or book.ask_depth < self.min_book_depth:
            return False

        return True

    def _update_order_flow_metrics(self, quote: Quote, book: OrderBook) -> None:
        """Atualiza todas as métricas de order flow"""
        # Volumes por lado
        bids = book.get_bids(self.imbalance_levels)
        asks = book.get_asks(self.imbalance_levels)

        bid_volume = sum(level.size for level in bids)
        ask_volume = sum(level.size for level in asks)

        self._bid_volume_history.append(bid_volume)
        self._ask_volume_history.append(ask_volume)

        # Médias de volume
        if len(self._bid_volume_history) >= 10:
            self._avg_bid_volume = np.mean(list(self._bid_volume_history))
            self._avg_ask_volume = np.mean(list(self._ask_volume_history))

        # Imbalance
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            self._metrics.imbalance = (bid_volume - ask_volume) / total_volume
        else:
            self._metrics.imbalance = 0.0

        self._imbalance_history.append(self._metrics.imbalance)

        # Delta (simplificado - usando mudança no book como proxy)
        self._metrics.bid_volume = bid_volume
        self._metrics.ask_volume = ask_volume
        self._metrics.delta = bid_volume - ask_volume
        self._delta_history.append(self._metrics.delta)

        # Delta cumulativo
        if self._delta_history:
            self._metrics.cumulative_delta = sum(self._delta_history)

        # Detecção de ordem grande
        self._metrics.large_order_detected = (
            bid_volume > self._avg_bid_volume * self.large_order_multiplier or
            ask_volume > self._avg_ask_volume * self.large_order_multiplier
        )

        # Detecção de absorção
        self._update_absorption_detection(quote, book)

        # Detecção de sweep
        self._update_sweep_detection(book)

        # Salvar snapshot do book
        self._book_snapshots.append({
            'timestamp': time.time(),
            'bids': [(l.price, l.size) for l in bids],
            'asks': [(l.price, l.size) for l in asks],
            'imbalance': self._metrics.imbalance
        })

    def _update_absorption_detection(self, quote: Quote, book: OrderBook) -> None:
        """Detecta absorção de ordens em um nível"""
        if len(self._book_snapshots) < 2:
            self._metrics.absorption_ratio = 0.0
            return

        # Comparar com snapshot anterior
        prev_snapshot = self._book_snapshots[-1]
        current_bid_vol = self._metrics.bid_volume
        current_ask_vol = self._metrics.ask_volume

        prev_bid_vol = sum(size for _, size in prev_snapshot.get('bids', []))
        prev_ask_vol = sum(size for _, size in prev_snapshot.get('asks', []))

        # Absorção = preço não moveu mas volume foi consumido
        if prev_bid_vol > 0 and prev_ask_vol > 0:
            bid_absorption = max(0, prev_bid_vol - current_bid_vol) / prev_bid_vol
            ask_absorption = max(0, prev_ask_vol - current_ask_vol) / prev_ask_vol

            # Ratio: positivo = bids absorvendo (bullish), negativo = asks absorvendo (bearish)
            self._metrics.absorption_ratio = bid_absorption - ask_absorption
        else:
            self._metrics.absorption_ratio = 0.0

    def _update_sweep_detection(self, book: OrderBook) -> None:
        """Detecta varredura de múltiplos níveis"""
        if len(self._book_snapshots) < 2:
            self._metrics.sweep_detected = False
            return

        prev_snapshot = self._book_snapshots[-1]
        prev_bid_levels = len(prev_snapshot.get('bids', []))
        prev_ask_levels = len(prev_snapshot.get('asks', []))

        current_bid_levels = book.bid_depth
        current_ask_levels = book.ask_depth

        # Sweep detectado se múltiplos níveis foram consumidos
        bid_levels_consumed = prev_bid_levels - current_bid_levels
        ask_levels_consumed = prev_ask_levels - current_ask_levels

        self._metrics.sweep_detected = (
            bid_levels_consumed >= self.sweep_detection_levels or
            ask_levels_consumed >= self.sweep_detection_levels
        )

    def _analyze_order_flow(self, symbol: str, quote: Quote, book: OrderBook) -> Optional[Signal]:
        """Analisa sinais de order flow"""
        if not self.can_generate_signal():
            return None

        if self.has_position(symbol):
            return None

        signal_type = None
        signal_reason = None
        confidence = 0.0

        # 1. Análise de Imbalance forte
        if abs(self._metrics.imbalance) > self.strong_imbalance_threshold:
            if self._metrics.imbalance > 0:
                signal_type = SignalType.BUY
                signal_reason = 'strong_imbalance'
                confidence = min(1.0, abs(self._metrics.imbalance))
            else:
                signal_type = SignalType.SELL
                signal_reason = 'strong_imbalance'
                confidence = min(1.0, abs(self._metrics.imbalance))

        # 2. Análise de Absorção
        elif abs(self._metrics.absorption_ratio) > 0.5:
            if self._metrics.absorption_ratio > 0:
                # Bids absorvendo = compradores fortes
                signal_type = SignalType.BUY
                signal_reason = 'absorption'
                confidence = min(1.0, 0.5 + abs(self._metrics.absorption_ratio) * 0.5)
            else:
                signal_type = SignalType.SELL
                signal_reason = 'absorption'
                confidence = min(1.0, 0.5 + abs(self._metrics.absorption_ratio) * 0.5)

        # 3. Análise de Sweep
        elif self._metrics.sweep_detected and self._metrics.large_order_detected:
            # Sweep com ordem grande = movimento forte
            if self._metrics.imbalance > 0:
                signal_type = SignalType.BUY
                signal_reason = 'sweep'
                confidence = 0.8
            elif self._metrics.imbalance < 0:
                signal_type = SignalType.SELL
                signal_reason = 'sweep'
                confidence = 0.8

        # 4. Análise de Delta Cumulativo
        elif len(self._delta_history) >= self.cumulative_delta_period:
            avg_delta = np.mean(list(self._delta_history))
            if avg_delta > self._avg_bid_volume * self.delta_threshold:
                signal_type = SignalType.BUY
                signal_reason = 'cumulative_delta'
                confidence = min(1.0, 0.5 + (avg_delta / self._avg_bid_volume) * 0.3)
            elif avg_delta < -self._avg_ask_volume * self.delta_threshold:
                signal_type = SignalType.SELL
                signal_reason = 'cumulative_delta'
                confidence = min(1.0, 0.5 + (abs(avg_delta) / self._avg_ask_volume) * 0.3)

        # Se temos sinal, iniciar confirmação
        if signal_type:
            self._pending_signal_type = signal_type
            self._signal_confirmation_count = 1
            self._pending_signal_reason = signal_reason
            self._pending_signal_confidence = confidence
            return None  # Aguardar confirmação

        return None

    def _process_pending_signal(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Processa confirmação de sinal pendente"""
        # Verificar se sinal ainda é válido
        signal_still_valid = False

        if self._pending_signal_type == SignalType.BUY:
            signal_still_valid = self._metrics.imbalance > 0.1 or self._metrics.absorption_ratio > 0.2
        else:
            signal_still_valid = self._metrics.imbalance < -0.1 or self._metrics.absorption_ratio < -0.2

        if signal_still_valid:
            self._signal_confirmation_count += 1
        else:
            # Sinal invalidado
            self._pending_signal_type = None
            self._signal_confirmation_count = 0
            return None

        # Verificar se atingiu confirmação
        if self._signal_confirmation_count >= self.confirmation_ticks:
            signal_type = self._pending_signal_type
            confidence = self._pending_signal_confidence
            reason = self._pending_signal_reason

            # Reset
            self._pending_signal_type = None
            self._signal_confirmation_count = 0

            # Incrementar contagem por tipo
            self._signals_by_type[reason] = self._signals_by_type.get(reason, 0) + 1

            mid = quote.mid_price

            if signal_type == SignalType.BUY:
                stop_loss = mid - (self.stop_loss_pips * self.pip_size)
                take_profit = mid + (self.take_profit_pips * self.pip_size)

                return self._emit_signal(Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    price=quote.ask_price,
                    strength=self._get_strength(confidence),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    metadata={
                        'strategy': 'order_flow',
                        'signal_type': reason,
                        'imbalance': self._metrics.imbalance,
                        'absorption_ratio': self._metrics.absorption_ratio,
                        'cumulative_delta': self._metrics.cumulative_delta,
                        'sweep_detected': self._metrics.sweep_detected
                    }
                ))

            else:
                stop_loss = mid + (self.stop_loss_pips * self.pip_size)
                take_profit = mid - (self.take_profit_pips * self.pip_size)

                return self._emit_signal(Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    price=quote.bid_price,
                    strength=self._get_strength(confidence),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    metadata={
                        'strategy': 'order_flow',
                        'signal_type': reason,
                        'imbalance': self._metrics.imbalance,
                        'absorption_ratio': self._metrics.absorption_ratio,
                        'cumulative_delta': self._metrics.cumulative_delta,
                        'sweep_detected': self._metrics.sweep_detected
                    }
                ))

        return None

    def _update_trailing_stops(self, symbol: str, quote: Quote) -> None:
        """Atualiza trailing stops"""
        position = self.get_position(symbol)
        if not position:
            return

        mid = quote.mid_price
        pip_value = self.pip_size

        if position.side == 'long':
            # Atualizar trailing high
            current_high = self._trailing_high.get(symbol, position.entry_price)
            if mid > current_high:
                self._trailing_high[symbol] = mid
        else:
            # Atualizar trailing low
            current_low = self._trailing_low.get(symbol, position.entry_price)
            if mid < current_low:
                self._trailing_low[symbol] = mid

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída"""
        position = self.get_position(symbol)
        if not position:
            return None

        current_price = quote.mid_price
        pip_value = self.pip_size

        # Calcular P&L em pips
        if position.side == 'long':
            pnl_pips = (current_price - position.entry_price) / pip_value
        else:
            pnl_pips = (position.entry_price - current_price) / pip_value

        # Trailing stop
        trailing_distance = self.trailing_stop_pips * pip_value

        if position.side == 'long':
            trailing_high = self._trailing_high.get(symbol, position.entry_price)
            trailing_stop = trailing_high - trailing_distance

            if current_price < trailing_stop and pnl_pips > 0.5:
                self._trailing_high.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    price=current_price,
                    strength=SignalStrength.STRONG,
                    confidence=0.9,
                    metadata={'reason': 'trailing_stop', 'pnl_pips': pnl_pips}
                ))

        else:
            trailing_low = self._trailing_low.get(symbol, position.entry_price)
            trailing_stop = trailing_low + trailing_distance

            if current_price > trailing_stop and pnl_pips > 0.5:
                self._trailing_low.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=symbol,
                    price=current_price,
                    strength=SignalStrength.STRONG,
                    confidence=0.9,
                    metadata={'reason': 'trailing_stop', 'pnl_pips': pnl_pips}
                ))

        # Reversão de order flow
        if position.side == 'long' and self._metrics.imbalance < -self.imbalance_threshold:
            if pnl_pips > 0.5:  # Só fecha se no lucro
                self._trailing_high.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    price=current_price,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={'reason': 'flow_reversal', 'pnl_pips': pnl_pips, 'imbalance': self._metrics.imbalance}
                ))

        if position.side == 'short' and self._metrics.imbalance > self.imbalance_threshold:
            if pnl_pips > 0.5:
                self._trailing_low.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=symbol,
                    price=current_price,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={'reason': 'flow_reversal', 'pnl_pips': pnl_pips, 'imbalance': self._metrics.imbalance}
                ))

        return None

    def _get_strength(self, confidence: float) -> SignalStrength:
        """Converte confiança em força do sinal"""
        if confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            return SignalStrength.STRONG
        elif confidence >= 0.4:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def metrics(self) -> dict:
        """Métricas de order flow atuais"""
        return {
            'imbalance': self._metrics.imbalance,
            'bid_volume': self._metrics.bid_volume,
            'ask_volume': self._metrics.ask_volume,
            'delta': self._metrics.delta,
            'cumulative_delta': self._metrics.cumulative_delta,
            'absorption_ratio': self._metrics.absorption_ratio,
            'large_order_detected': self._metrics.large_order_detected,
            'sweep_detected': self._metrics.sweep_detected,
            'avg_bid_volume': self._avg_bid_volume,
            'avg_ask_volume': self._avg_ask_volume,
            'signals_by_type': self._signals_by_type
        }
