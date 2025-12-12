"""
Scalping Strategy - Estratégia de scalping de alta frequência
"""

from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote, OrderBook

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """
    Estratégia de Scalping para HFT

    Características:
    - Opera em micro movimentos de preço
    - Aproveita spread bid-ask
    - Tempo de holding muito curto
    - Alta taxa de trades
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('Scalping', config)

        # Parâmetros da estratégia
        self.min_spread_pips = self.config.get('min_spread_pips', 0.5)
        self.max_spread_pips = self.config.get('max_spread_pips', 3.0)
        self.min_edge_pips = self.config.get('min_edge_pips', 0.3)
        self.pip_size = self.config.get('pip_size', 0.0001)

        # Order book imbalance
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.3)

        # Momentum
        self.momentum_window = self.config.get('momentum_window', 20)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.0002)

        # Stop/Take
        self.stop_loss_pips = self.config.get('stop_loss_pips', 5)
        self.take_profit_pips = self.config.get('take_profit_pips', 3)

        # Histórico de preços
        self._price_history: deque = deque(maxlen=self.momentum_window)
        self._tick_times: deque = deque(maxlen=100)

        # Estado
        self._last_mid = 0.0
        self._momentum = 0.0
        self._volatility = 0.0

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e gera sinal"""
        if not self.enabled:
            return None

        quote = state.quote
        book = state.book

        # Atualizar histórico
        mid = quote.mid_price
        if mid > 0:
            self._price_history.append(mid)
            self._tick_times.append(time.time())

        # Calcular métricas
        self._update_metrics()

        # Verificar condições de entrada
        signal = self._check_entry_conditions(quote, book)

        # Verificar condições de saída
        if signal is None:
            signal = self._check_exit_conditions(state.symbol, quote)

        return signal

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa atualização de cotação"""
        if not self.enabled:
            return None

        # Versão simplificada sem order book
        mid = quote.mid_price
        if mid > 0:
            self._price_history.append(mid)

        return None  # Precisa de tick completo para sinal

    def _update_metrics(self) -> None:
        """Atualiza métricas calculadas"""
        if len(self._price_history) < 2:
            return

        prices = list(self._price_history)

        # Momentum (retorno recente)
        if len(prices) >= 5 and prices[-5] != 0:
            self._momentum = (prices[-1] - prices[-5]) / prices[-5]

        # Volatilidade (desvio padrão dos retornos)
        if len(prices) >= 10:
            # Evitar divisão por zero
            prices_arr = np.array(prices[:-1])
            if np.all(prices_arr != 0):
                returns = np.diff(prices) / prices_arr
                self._volatility = np.std(returns) if len(returns) > 0 else 0

    def _check_entry_conditions(self, quote: Quote, book: OrderBook) -> Optional[Signal]:
        """Verifica condições de entrada"""
        if not self.can_generate_signal():
            return None

        spread_pips = quote.spread / self.pip_size

        # Spread muito alto ou muito baixo
        if spread_pips > self.max_spread_pips or spread_pips < self.min_spread_pips:
            return None

        # Calcular imbalance
        imbalance = book.get_imbalance()

        # Decisão de entrada
        signal = None

        # Imbalance favorável para compra
        if imbalance > self.imbalance_threshold and self._momentum > 0:
            confidence = min(1.0, abs(imbalance) + abs(self._momentum) * 100)

            signal = Signal(
                signal_type=SignalType.BUY,
                symbol=book.symbol,
                price=quote.ask_price,
                strength=self._get_signal_strength(confidence),
                stop_loss=quote.ask_price - (self.stop_loss_pips * self.pip_size),
                take_profit=quote.ask_price + (self.take_profit_pips * self.pip_size),
                confidence=confidence,
                metadata={
                    'imbalance': imbalance,
                    'momentum': self._momentum,
                    'spread_pips': spread_pips
                }
            )

        # Imbalance favorável para venda
        elif imbalance < -self.imbalance_threshold and self._momentum < 0:
            confidence = min(1.0, abs(imbalance) + abs(self._momentum) * 100)

            signal = Signal(
                signal_type=SignalType.SELL,
                symbol=book.symbol,
                price=quote.bid_price,
                strength=self._get_signal_strength(confidence),
                stop_loss=quote.bid_price + (self.stop_loss_pips * self.pip_size),
                take_profit=quote.bid_price - (self.take_profit_pips * self.pip_size),
                confidence=confidence,
                metadata={
                    'imbalance': imbalance,
                    'momentum': self._momentum,
                    'spread_pips': spread_pips
                }
            )

        if signal:
            return self._emit_signal(signal)

        return None

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída"""
        position = self.get_position(symbol)
        if not position:
            return None

        current_price = quote.mid_price

        # Calcular P&L
        if position.side == 'long':
            pnl_pips = (current_price - position.entry_price) / self.pip_size
        else:
            pnl_pips = (position.entry_price - current_price) / self.pip_size

        # Stop loss atingido
        if pnl_pips <= -self.stop_loss_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.VERY_STRONG,
                confidence=1.0,
                metadata={'reason': 'stop_loss', 'pnl_pips': pnl_pips}
            ))

        # Take profit atingido
        if pnl_pips >= self.take_profit_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.STRONG,
                confidence=1.0,
                metadata={'reason': 'take_profit', 'pnl_pips': pnl_pips}
            ))

        # Momentum reverteu
        if position.side == 'long' and self._momentum < -self.momentum_threshold:
            return self._emit_signal(Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                metadata={'reason': 'momentum_reversal', 'momentum': self._momentum}
            ))

        if position.side == 'short' and self._momentum > self.momentum_threshold:
            return self._emit_signal(Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                metadata={'reason': 'momentum_reversal', 'momentum': self._momentum}
            ))

        return None

    def _get_signal_strength(self, confidence: float) -> SignalStrength:
        """Converte confiança em força do sinal"""
        if confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            return SignalStrength.STRONG
        elif confidence >= 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    @property
    def metrics(self) -> dict:
        """Métricas atuais"""
        return {
            'momentum': self._momentum,
            'volatility': self._volatility,
            'price_history_size': len(self._price_history)
        }
