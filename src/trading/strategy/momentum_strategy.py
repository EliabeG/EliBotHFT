"""
Momentum Strategy - Estratégia de momentum para HFT
"""

from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Estratégia de Momentum para HFT

    Detecta movimentos direcionais fortes e entra na direção do momentum.
    Usa múltiplos timeframes de momentum para confirmação.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('Momentum', config)

        # Timeframes de análise
        self.short_window = self.config.get('short_window', 10)
        self.medium_window = self.config.get('medium_window', 30)
        self.long_window = self.config.get('long_window', 100)

        # Thresholds
        self.entry_threshold = self.config.get('entry_threshold', 0.0003)
        self.exit_threshold = self.config.get('exit_threshold', 0.0001)
        self.confirmation_ratio = self.config.get('confirmation_ratio', 0.7)

        # Stop/Take (em % do preço)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.002)  # 0.2%
        self.take_profit_pct = self.config.get('take_profit_pct', 0.003)  # 0.3%
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 0.001)  # 0.1%

        # Volume filter
        self.volume_filter = self.config.get('volume_filter', True)
        self.min_volume_ratio = self.config.get('min_volume_ratio', 1.2)

        # Históricos
        self._price_history: deque = deque(maxlen=self.long_window)
        self._volume_history: deque = deque(maxlen=self.long_window)
        self._tick_count = 0

        # Indicadores
        self._short_momentum = 0.0
        self._medium_momentum = 0.0
        self._long_momentum = 0.0
        self._rsi = 50.0
        self._trend_strength = 0.0

        # Trailing stop
        self._trailing_highs: Dict[str, float] = {}
        self._trailing_lows: Dict[str, float] = {}

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e gera sinal"""
        if not self.enabled:
            return None

        quote = state.quote
        mid = quote.mid_price

        if mid <= 0:
            return None

        # Atualizar histórico
        self._price_history.append(mid)
        self._tick_count += 1

        # Atualizar indicadores
        self._update_indicators()

        # Gerenciar trailing stop
        symbol = state.symbol
        self._update_trailing_stop(symbol, mid)

        # Verificar condições
        signal = self._check_entry_conditions(symbol, quote)

        if signal is None:
            signal = self._check_exit_conditions(symbol, quote)

        return signal

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa cotação"""
        return None

    def _update_indicators(self) -> None:
        """Atualiza indicadores técnicos"""
        prices = list(self._price_history)

        if len(prices) < self.short_window:
            return

        # Momentum em diferentes timeframes
        if len(prices) >= self.short_window:
            self._short_momentum = (prices[-1] - prices[-self.short_window]) / prices[-self.short_window]

        if len(prices) >= self.medium_window:
            self._medium_momentum = (prices[-1] - prices[-self.medium_window]) / prices[-self.medium_window]

        if len(prices) >= self.long_window:
            self._long_momentum = (prices[-1] - prices[-self.long_window]) / prices[-self.long_window]

        # RSI simplificado
        self._rsi = self._calculate_rsi(prices, 14)

        # Força da tendência (baseada na concordância dos momentums)
        momentums = [self._short_momentum, self._medium_momentum, self._long_momentum]
        positive = sum(1 for m in momentums if m > 0)
        negative = sum(1 for m in momentums if m < 0)

        if positive > negative:
            self._trend_strength = positive / 3
        elif negative > positive:
            self._trend_strength = -negative / 3
        else:
            self._trend_strength = 0

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calcula RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _update_trailing_stop(self, symbol: str, price: float) -> None:
        """Atualiza trailing stop"""
        position = self.get_position(symbol)
        if not position:
            return

        if position.side == 'long':
            # Atualizar high
            current_high = self._trailing_highs.get(symbol, price)
            if price > current_high:
                self._trailing_highs[symbol] = price
        else:
            # Atualizar low
            current_low = self._trailing_lows.get(symbol, price)
            if price < current_low:
                self._trailing_lows[symbol] = price

    def _check_entry_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de entrada"""
        if not self.can_generate_signal():
            return None

        # Já tem posição
        if self.has_position(symbol):
            return None

        # Momentum insuficiente
        if abs(self._short_momentum) < self.entry_threshold:
            return None

        # Confirmação de tendência
        if abs(self._trend_strength) < self.confirmation_ratio:
            return None

        mid = quote.mid_price

        # Sinal de compra
        if (self._short_momentum > self.entry_threshold and
            self._medium_momentum > 0 and
            self._rsi < 70):

            confidence = min(1.0, abs(self._trend_strength) + abs(self._short_momentum) * 50)

            return self._emit_signal(Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=quote.ask_price,
                strength=self._get_strength(confidence),
                stop_loss=mid * (1 - self.stop_loss_pct),
                take_profit=mid * (1 + self.take_profit_pct),
                confidence=confidence,
                metadata={
                    'short_momentum': self._short_momentum,
                    'medium_momentum': self._medium_momentum,
                    'trend_strength': self._trend_strength,
                    'rsi': self._rsi
                }
            ))

        # Sinal de venda
        if (self._short_momentum < -self.entry_threshold and
            self._medium_momentum < 0 and
            self._rsi > 30):

            confidence = min(1.0, abs(self._trend_strength) + abs(self._short_momentum) * 50)

            return self._emit_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=quote.bid_price,
                strength=self._get_strength(confidence),
                stop_loss=mid * (1 + self.stop_loss_pct),
                take_profit=mid * (1 - self.take_profit_pct),
                confidence=confidence,
                metadata={
                    'short_momentum': self._short_momentum,
                    'medium_momentum': self._medium_momentum,
                    'trend_strength': self._trend_strength,
                    'rsi': self._rsi
                }
            ))

        return None

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída"""
        position = self.get_position(symbol)
        if not position:
            return None

        mid = quote.mid_price

        # Trailing stop
        if position.side == 'long':
            trailing_high = self._trailing_highs.get(symbol, position.entry_price)
            trailing_stop = trailing_high * (1 - self.trailing_stop_pct)

            if mid < trailing_stop:
                self._trailing_highs.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.STRONG,
                    confidence=0.9,
                    metadata={'reason': 'trailing_stop', 'trigger_price': trailing_stop}
                ))

            # Momentum reversal
            if self._short_momentum < -self.exit_threshold:
                self._trailing_highs.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={'reason': 'momentum_reversal', 'momentum': self._short_momentum}
                ))

        else:  # Short
            trailing_low = self._trailing_lows.get(symbol, position.entry_price)
            trailing_stop = trailing_low * (1 + self.trailing_stop_pct)

            if mid > trailing_stop:
                self._trailing_lows.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.STRONG,
                    confidence=0.9,
                    metadata={'reason': 'trailing_stop', 'trigger_price': trailing_stop}
                ))

            # Momentum reversal
            if self._short_momentum > self.exit_threshold:
                self._trailing_lows.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={'reason': 'momentum_reversal', 'momentum': self._short_momentum}
                ))

        return None

    def _get_strength(self, confidence: float) -> SignalStrength:
        """Determina força do sinal"""
        if confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            return SignalStrength.STRONG
        elif confidence >= 0.4:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def indicators(self) -> dict:
        """Indicadores atuais"""
        return {
            'short_momentum': self._short_momentum,
            'medium_momentum': self._medium_momentum,
            'long_momentum': self._long_momentum,
            'rsi': self._rsi,
            'trend_strength': self._trend_strength,
            'tick_count': self._tick_count
        }
