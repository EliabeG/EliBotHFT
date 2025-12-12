"""
Breakout Strategy - Estratégia de Rompimento de Volatilidade para HFT
Baseada em ATR e detecção de níveis de suporte/resistência
"""

from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote, OrderBook

logger = logging.getLogger(__name__)


class BreakoutType(Enum):
    """Tipo de breakout"""
    RESISTANCE_BREAK = 'resistance_break'
    SUPPORT_BREAK = 'support_break'
    VOLATILITY_EXPANSION = 'volatility_expansion'
    RANGE_BREAK = 'range_break'


@dataclass
class PriceLevel:
    """Nível de preço (suporte/resistência)"""
    price: float
    strength: int = 1  # Número de toques
    last_touch: float = 0.0
    level_type: str = 'support'  # 'support' ou 'resistance'


@dataclass
class ATRState:
    """Estado do Average True Range"""
    current: float = 0.0
    fast: float = 0.0  # ATR rápido
    slow: float = 0.0  # ATR lento
    ratio: float = 1.0  # fast / slow (expansão de volatilidade)
    percentile: float = 50.0  # Percentil histórico


class BreakoutStrategy(BaseStrategy):
    """
    Estratégia de Breakout para HFT

    Características:
    - Detecta rompimentos de níveis de suporte/resistência
    - Usa ATR para filtrar falsos breakouts
    - Identifica expansão de volatilidade
    - Confirmação por volume/momentum
    - Otimizada para EURUSD
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('Breakout', config)

        # Parâmetros de ATR
        self.atr_fast_period = self.config.get('atr_fast_period', 10)
        self.atr_slow_period = self.config.get('atr_slow_period', 50)
        self.atr_multiplier = self.config.get('atr_multiplier', 1.5)

        # Parâmetros de níveis
        self.level_lookback = self.config.get('level_lookback', 100)
        self.level_tolerance_pips = self.config.get('level_tolerance_pips', 2.0)
        self.min_level_touches = self.config.get('min_level_touches', 2)

        # Pip configuration
        self.pip_size = self.config.get('pip_size', 0.0001)

        # Parâmetros de breakout
        self.breakout_confirmation_ticks = self.config.get('breakout_confirmation_ticks', 3)
        self.min_breakout_distance_pips = self.config.get('min_breakout_distance_pips', 1.0)
        self.volatility_expansion_threshold = self.config.get('volatility_expansion_threshold', 1.5)

        # Parâmetros de risco
        self.stop_loss_atr_multiplier = self.config.get('stop_loss_atr_multiplier', 1.5)
        self.take_profit_atr_multiplier = self.config.get('take_profit_atr_multiplier', 2.0)
        self.max_stop_loss_pips = self.config.get('max_stop_loss_pips', 20)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.5)

        # Filtros
        self.min_volume_increase = self.config.get('min_volume_increase', 1.2)
        self.momentum_confirmation = self.config.get('momentum_confirmation', True)

        # Históricos
        self._high_history: deque = deque(maxlen=self.level_lookback)
        self._low_history: deque = deque(maxlen=self.level_lookback)
        self._close_history: deque = deque(maxlen=self.level_lookback)
        self._tr_history: deque = deque(maxlen=self.atr_slow_period)
        self._atr_history: deque = deque(maxlen=100)
        self._volume_history: deque = deque(maxlen=20)

        # Estado
        self._atr = ATRState()
        self._support_levels: List[PriceLevel] = []
        self._resistance_levels: List[PriceLevel] = []
        self._last_high = 0.0
        self._last_low = float('inf')
        self._breakout_in_progress = False
        self._breakout_direction = 0  # 1 = bullish, -1 = bearish
        self._breakout_confirmation_count = 0
        self._breakout_level = 0.0

        # Métricas
        self._breakouts_detected = 0
        self._false_breakouts = 0
        self._successful_breakouts = 0

        logger.info(f"Breakout strategy initialized")

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e detecta breakouts"""
        if not self.enabled:
            return None

        quote = state.quote
        book = state.book
        symbol = state.symbol

        if quote.bid_price <= 0 or quote.ask_price <= 0:
            return None

        # Simular OHLC a partir do tick
        mid = quote.mid_price
        self._update_ohlc(mid)

        # Atualizar ATR
        self._update_atr()

        # Atualizar níveis de suporte/resistência
        self._update_price_levels()

        # Atualizar volume
        self._update_volume(quote)

        # Verificar saída primeiro
        exit_signal = self._check_exit_conditions(symbol, quote)
        if exit_signal:
            return exit_signal

        # Verificar breakout em andamento
        if self._breakout_in_progress:
            return self._process_breakout_confirmation(symbol, quote)

        # Detectar novo breakout
        return self._detect_breakout(symbol, quote, book)

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa atualização de cotação"""
        return None

    def _update_ohlc(self, price: float) -> None:
        """Atualiza dados OHLC simulados"""
        self._close_history.append(price)

        # Atualizar high/low da janela
        if len(self._close_history) > 1:
            window = list(self._close_history)[-20:]  # Janela de 20 ticks
            self._high_history.append(max(window))
            self._low_history.append(min(window))
        else:
            self._high_history.append(price)
            self._low_history.append(price)

    def _update_atr(self) -> None:
        """Calcula ATR (Average True Range)"""
        if len(self._close_history) < 2:
            return

        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        high = self._high_history[-1] if self._high_history else self._close_history[-1]
        low = self._low_history[-1] if self._low_history else self._close_history[-1]
        prev_close = self._close_history[-2]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        self._tr_history.append(tr)

        if len(self._tr_history) >= self.atr_fast_period:
            # ATR rápido
            self._atr.fast = np.mean(list(self._tr_history)[-self.atr_fast_period:])

        if len(self._tr_history) >= self.atr_slow_period:
            # ATR lento
            self._atr.slow = np.mean(list(self._tr_history))

        # ATR atual (média dos dois)
        if self._atr.fast > 0 and self._atr.slow > 0:
            self._atr.current = (self._atr.fast + self._atr.slow) / 2
            self._atr.ratio = self._atr.fast / self._atr.slow
        elif self._atr.fast > 0:
            self._atr.current = self._atr.fast
            self._atr.ratio = 1.0

        # Histórico de ATR para percentil
        if self._atr.current > 0:
            self._atr_history.append(self._atr.current)

        # Calcular percentil
        if len(self._atr_history) >= 20:
            sorted_atr = sorted(self._atr_history)
            idx = sorted_atr.index(self._atr.current) if self._atr.current in sorted_atr else len(sorted_atr) // 2
            self._atr.percentile = (idx / len(sorted_atr)) * 100

    def _update_price_levels(self) -> None:
        """Identifica níveis de suporte e resistência"""
        if len(self._close_history) < self.level_lookback:
            return

        prices = list(self._close_history)
        highs = list(self._high_history) if self._high_history else prices
        lows = list(self._low_history) if self._low_history else prices

        tolerance = self.level_tolerance_pips * self.pip_size

        # Encontrar pivots (máximos e mínimos locais)
        window = 5

        # Resistências (máximos locais)
        self._resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                # Verificar se já existe nível próximo
                level_exists = False
                for level in self._resistance_levels:
                    if abs(level.price - highs[i]) < tolerance:
                        level.strength += 1
                        level.last_touch = time.time()
                        level_exists = True
                        break

                if not level_exists:
                    self._resistance_levels.append(PriceLevel(
                        price=highs[i],
                        strength=1,
                        last_touch=time.time(),
                        level_type='resistance'
                    ))

        # Suportes (mínimos locais)
        self._support_levels = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                level_exists = False
                for level in self._support_levels:
                    if abs(level.price - lows[i]) < tolerance:
                        level.strength += 1
                        level.last_touch = time.time()
                        level_exists = True
                        break

                if not level_exists:
                    self._support_levels.append(PriceLevel(
                        price=lows[i],
                        strength=1,
                        last_touch=time.time(),
                        level_type='support'
                    ))

        # Filtrar níveis fracos
        self._resistance_levels = [l for l in self._resistance_levels if l.strength >= self.min_level_touches]
        self._support_levels = [l for l in self._support_levels if l.strength >= self.min_level_touches]

        # Ordenar por proximidade do preço atual
        current_price = prices[-1]
        self._resistance_levels.sort(key=lambda l: l.price - current_price if l.price > current_price else float('inf'))
        self._support_levels.sort(key=lambda l: current_price - l.price if l.price < current_price else float('inf'))

    def _update_volume(self, quote: Quote) -> None:
        """Atualiza histórico de volume"""
        total_volume = quote.bid_size + quote.ask_size
        self._volume_history.append(total_volume)

    def _detect_breakout(self, symbol: str, quote: Quote, book: OrderBook) -> Optional[Signal]:
        """Detecta início de breakout"""
        if not self.can_generate_signal():
            return None

        if self.has_position(symbol):
            return None

        current_price = quote.mid_price

        # Verificar expansão de volatilidade
        if self._atr.ratio < self.volatility_expansion_threshold:
            return None

        # Verificar níveis de resistência
        for level in self._resistance_levels[:3]:  # Top 3 mais próximos
            breakout_distance = (current_price - level.price) / self.pip_size

            if breakout_distance > self.min_breakout_distance_pips:
                # Breakout de resistência detectado
                self._breakout_in_progress = True
                self._breakout_direction = 1
                self._breakout_confirmation_count = 1
                self._breakout_level = level.price
                self._breakouts_detected += 1

                logger.debug(f"Resistance breakout detected at {level.price:.5f}")
                return None  # Aguardar confirmação

        # Verificar níveis de suporte
        for level in self._support_levels[:3]:
            breakout_distance = (level.price - current_price) / self.pip_size

            if breakout_distance > self.min_breakout_distance_pips:
                # Breakout de suporte detectado
                self._breakout_in_progress = True
                self._breakout_direction = -1
                self._breakout_confirmation_count = 1
                self._breakout_level = level.price
                self._breakouts_detected += 1

                logger.debug(f"Support breakout detected at {level.price:.5f}")
                return None  # Aguardar confirmação

        return None

    def _process_breakout_confirmation(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Processa confirmação de breakout"""
        current_price = quote.mid_price

        # Verificar se breakout ainda é válido
        if self._breakout_direction == 1:  # Bullish
            if current_price > self._breakout_level:
                self._breakout_confirmation_count += 1
            else:
                # Breakout falhou
                self._breakout_in_progress = False
                self._false_breakouts += 1
                return None

        else:  # Bearish
            if current_price < self._breakout_level:
                self._breakout_confirmation_count += 1
            else:
                # Breakout falhou
                self._breakout_in_progress = False
                self._false_breakouts += 1
                return None

        # Verificar se atingiu confirmação necessária
        if self._breakout_confirmation_count >= self.breakout_confirmation_ticks:
            self._breakout_in_progress = False

            # Verificar volume (confirmação adicional)
            if len(self._volume_history) >= 10:
                recent_volume = np.mean(list(self._volume_history)[-5:])
                avg_volume = np.mean(list(self._volume_history))
                if recent_volume < avg_volume * self.min_volume_increase:
                    return None  # Volume insuficiente

            # Calcular SL/TP baseado em ATR
            atr_pips = self._atr.current / self.pip_size if self._atr.current > 0 else 10

            stop_loss_pips = min(atr_pips * self.stop_loss_atr_multiplier, self.max_stop_loss_pips)
            take_profit_pips = atr_pips * self.take_profit_atr_multiplier

            # Verificar risk/reward
            if take_profit_pips / stop_loss_pips < self.min_risk_reward:
                take_profit_pips = stop_loss_pips * self.min_risk_reward

            confidence = min(1.0, 0.5 + (self._atr.ratio - 1) * 0.5)

            if self._breakout_direction == 1:  # Bullish breakout
                stop_loss = current_price - (stop_loss_pips * self.pip_size)
                take_profit = current_price + (take_profit_pips * self.pip_size)

                self._successful_breakouts += 1

                return self._emit_signal(Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    price=quote.ask_price,
                    strength=self._get_strength(confidence),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    metadata={
                        'strategy': 'breakout',
                        'breakout_type': BreakoutType.RESISTANCE_BREAK.value,
                        'breakout_level': self._breakout_level,
                        'atr': self._atr.current,
                        'atr_ratio': self._atr.ratio,
                        'atr_percentile': self._atr.percentile,
                        'stop_loss_pips': stop_loss_pips,
                        'take_profit_pips': take_profit_pips
                    }
                ))

            else:  # Bearish breakout
                stop_loss = current_price + (stop_loss_pips * self.pip_size)
                take_profit = current_price - (take_profit_pips * self.pip_size)

                self._successful_breakouts += 1

                return self._emit_signal(Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    price=quote.bid_price,
                    strength=self._get_strength(confidence),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    metadata={
                        'strategy': 'breakout',
                        'breakout_type': BreakoutType.SUPPORT_BREAK.value,
                        'breakout_level': self._breakout_level,
                        'atr': self._atr.current,
                        'atr_ratio': self._atr.ratio,
                        'atr_percentile': self._atr.percentile,
                        'stop_loss_pips': stop_loss_pips,
                        'take_profit_pips': take_profit_pips
                    }
                ))

        return None

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída"""
        position = self.get_position(symbol)
        if not position:
            return None

        current_price = quote.mid_price

        # Calcular P&L em pips
        if position.side == 'long':
            pnl_pips = (current_price - position.entry_price) / self.pip_size
        else:
            pnl_pips = (position.entry_price - current_price) / self.pip_size

        # Stop Loss / Take Profit são gerenciados pelo broker
        # Aqui fazemos trailing stop baseado em ATR

        atr_pips = self._atr.current / self.pip_size if self._atr.current > 0 else 10
        trailing_distance = atr_pips * 1.0  # 1x ATR trailing

        if pnl_pips > trailing_distance:
            # Verificar se reverteu mais que o trailing
            if position.side == 'long':
                high_since_entry = max(list(self._close_history)[-20:]) if self._close_history else current_price
                drawdown_pips = (high_since_entry - current_price) / self.pip_size

                if drawdown_pips > trailing_distance:
                    return self._emit_signal(Signal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=symbol,
                        price=current_price,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        metadata={'reason': 'atr_trailing_stop', 'pnl_pips': pnl_pips}
                    ))

            else:
                low_since_entry = min(list(self._close_history)[-20:]) if self._close_history else current_price
                drawdown_pips = (current_price - low_since_entry) / self.pip_size

                if drawdown_pips > trailing_distance:
                    return self._emit_signal(Signal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=symbol,
                        price=current_price,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        metadata={'reason': 'atr_trailing_stop', 'pnl_pips': pnl_pips}
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
    def indicators(self) -> dict:
        """Indicadores atuais"""
        return {
            'atr_current': self._atr.current,
            'atr_fast': self._atr.fast,
            'atr_slow': self._atr.slow,
            'atr_ratio': self._atr.ratio,
            'atr_percentile': self._atr.percentile,
            'resistance_levels': len(self._resistance_levels),
            'support_levels': len(self._support_levels),
            'breakout_in_progress': self._breakout_in_progress,
            'breakout_direction': self._breakout_direction,
            'breakouts_detected': self._breakouts_detected,
            'false_breakouts': self._false_breakouts,
            'successful_breakouts': self._successful_breakouts
        }
