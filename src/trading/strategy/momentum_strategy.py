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

        # Thresholds - valores menores para EURUSD (menos volátil)
        self.entry_threshold = self.config.get('entry_threshold', 0.0001)  # 0.01% - mais sensível
        self.exit_threshold = self.config.get('exit_threshold', 0.00005)  # 0.005%
        self.confirmation_ratio = self.config.get('confirmation_ratio', 0.5)  # 50% confirmação

        # Stop/Take em PIPS (mais preciso para HFT)
        # Para EURUSD: 1 pip = 0.0001, 20 pontos = 2 pips
        self.stop_loss_pips = self.config.get('stop_loss_pips', 30)  # 30 pips = 300 pontos
        self.take_profit_pips = self.config.get('take_profit_pips', 5)  # 5 pips = 50 pontos (~$0.50)
        self.trailing_stop_pips = self.config.get('trailing_stop_pips', 2)  # 2 pips = 20 pontos

        # Lucro mínimo antes de ativar trailing stop (para cobrir comissão)
        # Com 0.01 lot: 1 pip = $0.10, comissão ~$0.03/lado = $0.06 total
        # Precisa de pelo menos 1 pip de lucro para cobrir comissão
        self.min_profit_to_trail_pips = self.config.get('min_profit_to_trail_pips', 1)  # 1 pip mínimo

        # Pip value por símbolo (pode ser configurado)
        self._pip_values = self.config.get('pip_values', {
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDJPY': 0.01,
            'default': 0.0001
        })

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

    def _get_pip_value(self, symbol: str) -> float:
        """Retorna o valor de 1 pip para o símbolo"""
        return self._pip_values.get(symbol, self._pip_values.get('default', 0.0001))

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
        """Atualiza trailing stop - só ativa após lucro mínimo"""
        position = self.get_position(symbol)
        if not position:
            return

        pip_value = self._get_pip_value(symbol)
        min_profit_distance = self.min_profit_to_trail_pips * pip_value

        if position.side == 'long':
            # Só começa a rastrear trailing high após lucro mínimo
            current_profit = price - position.entry_price
            if current_profit >= min_profit_distance:
                current_high = self._trailing_highs.get(symbol, price)
                if price > current_high:
                    self._trailing_highs[symbol] = price
                    logger.debug(f"[{symbol}] Trailing high atualizado: {price:.5f} (profit: {current_profit/pip_value:.1f} pips)")
        else:
            # Short: lucro é quando preço cai
            current_profit = position.entry_price - price
            if current_profit >= min_profit_distance:
                current_low = self._trailing_lows.get(symbol, price)
                if price < current_low:
                    self._trailing_lows[symbol] = price
                    logger.debug(f"[{symbol}] Trailing low atualizado: {price:.5f} (profit: {current_profit/pip_value:.1f} pips)")

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
        pip_value = self._get_pip_value(symbol)

        # Calcular SL/TP em valor absoluto (baseado em pips)
        sl_distance = self.stop_loss_pips * pip_value
        tp_distance = self.take_profit_pips * pip_value

        # Sinal de compra
        if (self._short_momentum > self.entry_threshold and
            self._medium_momentum > 0 and
            self._rsi < 70):

            confidence = min(1.0, abs(self._trend_strength) + abs(self._short_momentum) * 50)

            stop_loss = mid - sl_distance
            take_profit = mid + tp_distance

            logger.debug(f"[{symbol}] BUY signal: mid={mid:.5f}, SL={stop_loss:.5f} ({self.stop_loss_pips} pips), TP={take_profit:.5f} ({self.take_profit_pips} pips)")

            return self._emit_signal(Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=quote.ask_price,
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'short_momentum': self._short_momentum,
                    'medium_momentum': self._medium_momentum,
                    'trend_strength': self._trend_strength,
                    'rsi': self._rsi,
                    'sl_pips': self.stop_loss_pips,
                    'tp_pips': self.take_profit_pips
                }
            ))

        # Sinal de venda
        if (self._short_momentum < -self.entry_threshold and
            self._medium_momentum < 0 and
            self._rsi > 30):

            confidence = min(1.0, abs(self._trend_strength) + abs(self._short_momentum) * 50)

            stop_loss = mid + sl_distance
            take_profit = mid - tp_distance

            logger.debug(f"[{symbol}] SELL signal: mid={mid:.5f}, SL={stop_loss:.5f} ({self.stop_loss_pips} pips), TP={take_profit:.5f} ({self.take_profit_pips} pips)")

            return self._emit_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=quote.bid_price,
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'short_momentum': self._short_momentum,
                    'medium_momentum': self._medium_momentum,
                    'trend_strength': self._trend_strength,
                    'rsi': self._rsi,
                    'sl_pips': self.stop_loss_pips,
                    'tp_pips': self.take_profit_pips
                }
            ))

        return None

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída - só fecha no positivo"""
        position = self.get_position(symbol)
        if not position:
            return None

        mid = quote.mid_price
        pip_value = self._get_pip_value(symbol)
        trailing_distance = self.trailing_stop_pips * pip_value
        min_profit_distance = self.min_profit_to_trail_pips * pip_value

        # Trailing stop
        if position.side == 'long':
            # Só verifica trailing stop se já tiver trailing high registrado
            trailing_high = self._trailing_highs.get(symbol)

            if trailing_high is not None:
                trailing_stop = trailing_high - trailing_distance
                profit_pips = (mid - position.entry_price) / pip_value

                # SÓ fecha se ainda estiver no lucro (acima do mínimo)
                if mid < trailing_stop and profit_pips >= 0.5:  # Mínimo 0.5 pips de lucro
                    logger.info(f"[{symbol}] TRAILING STOP triggered: high={trailing_high:.5f}, stop={trailing_stop:.5f}, profit={profit_pips:.1f} pips")
                    self._trailing_highs.pop(symbol, None)
                    return self._emit_signal(Signal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        price=mid,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        metadata={
                            'reason': 'trailing_stop',
                            'trigger_price': trailing_stop,
                            'trailing_pips': self.trailing_stop_pips,
                            'profit_pips': profit_pips
                        }
                    ))

            # Momentum reversal - só sai se estiver no lucro
            profit_pips = (mid - position.entry_price) / pip_value
            if self._short_momentum < -self.exit_threshold and profit_pips >= 0.5:
                logger.info(f"[{symbol}] MOMENTUM REVERSAL exit: profit={profit_pips:.1f} pips")
                self._trailing_highs.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={
                        'reason': 'momentum_reversal',
                        'momentum': self._short_momentum,
                        'profit_pips': profit_pips
                    }
                ))

        else:  # Short
            # Só verifica trailing stop se já tiver trailing low registrado
            trailing_low = self._trailing_lows.get(symbol)

            if trailing_low is not None:
                trailing_stop = trailing_low + trailing_distance
                profit_pips = (position.entry_price - mid) / pip_value

                # SÓ fecha se ainda estiver no lucro
                if mid > trailing_stop and profit_pips >= 0.5:
                    logger.info(f"[{symbol}] TRAILING STOP triggered: low={trailing_low:.5f}, stop={trailing_stop:.5f}, profit={profit_pips:.1f} pips")
                    self._trailing_lows.pop(symbol, None)
                    return self._emit_signal(Signal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=symbol,
                        price=mid,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        metadata={
                            'reason': 'trailing_stop',
                            'trigger_price': trailing_stop,
                            'trailing_pips': self.trailing_stop_pips,
                            'profit_pips': profit_pips
                        }
                    ))

            # Momentum reversal - só sai se estiver no lucro
            profit_pips = (position.entry_price - mid) / pip_value
            if self._short_momentum > self.exit_threshold and profit_pips >= 0.5:
                logger.info(f"[{symbol}] MOMENTUM REVERSAL exit: profit={profit_pips:.1f} pips")
                self._trailing_lows.pop(symbol, None)
                return self._emit_signal(Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=symbol,
                    price=mid,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={
                        'reason': 'momentum_reversal',
                        'momentum': self._short_momentum,
                        'profit_pips': profit_pips
                    }
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
