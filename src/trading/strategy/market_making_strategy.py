"""
Market Making Strategy - Estratégia de Market Making para HFT
Captura spread bid-ask com gerenciamento de inventário
"""

from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote, OrderBook, BookSide

logger = logging.getLogger(__name__)


@dataclass
class InventoryState:
    """Estado do inventário para market making"""
    position: float = 0.0  # Posição atual em lots
    target: float = 0.0  # Posição alvo
    max_inventory: float = 1.0  # Máximo permitido
    skew_factor: float = 0.0  # Fator de ajuste de preços


class MarketMakingStrategy(BaseStrategy):
    """
    Estratégia de Market Making para HFT

    Características:
    - Captura spread bid-ask
    - Gerenciamento ativo de inventário
    - Ajuste dinâmico de quotes baseado em risco
    - Proteção contra seleção adversa
    - Otimizado para EURUSD
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('MarketMaking', config)

        # Parâmetros de spread
        self.min_spread_pips = self.config.get('min_spread_pips', 0.3)
        self.target_spread_pips = self.config.get('target_spread_pips', 0.5)
        self.max_spread_pips = self.config.get('max_spread_pips', 2.0)

        # Pip configuration
        self.pip_size = self.config.get('pip_size', 0.0001)  # EURUSD

        # Parâmetros de inventário
        self.max_inventory_lots = self.config.get('max_inventory_lots', 0.5)
        self.inventory_skew_factor = self.config.get('inventory_skew_factor', 0.5)
        self.target_inventory = self.config.get('target_inventory', 0.0)  # Neutro

        # Proteção contra seleção adversa
        self.volatility_multiplier = self.config.get('volatility_multiplier', 1.5)
        self.min_edge_pips = self.config.get('min_edge_pips', 0.1)

        # Parâmetros de risco
        self.stop_loss_pips = self.config.get('stop_loss_pips', 10)
        self.take_profit_pips = self.config.get('take_profit_pips', 3)
        self.max_holding_time_ms = self.config.get('max_holding_time_ms', 30000)

        # Filtros de mercado
        self.min_book_depth = self.config.get('min_book_depth', 3)
        self.max_imbalance = self.config.get('max_imbalance', 0.7)  # Evita mercado unilateral

        # Estado do inventário
        self._inventory = InventoryState(max_inventory=self.max_inventory_lots)

        # Históricos
        self._price_history: deque = deque(maxlen=100)
        self._spread_history: deque = deque(maxlen=50)
        self._trade_history: deque = deque(maxlen=100)

        # Métricas calculadas
        self._volatility = 0.0
        self._avg_spread = 0.0
        self._fair_value = 0.0
        self._microprice = 0.0

        # Estatísticas de performance
        self._spread_captured = 0.0
        self._inventory_cost = 0.0
        self._adverse_selection_cost = 0.0

        logger.info(f"MarketMaking strategy initialized for EURUSD")

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e decide ação de market making"""
        if not self.enabled:
            return None

        quote = state.quote
        book = state.book
        symbol = state.symbol

        # Validar dados
        if not self._validate_market_data(quote, book):
            return None

        # Atualizar métricas
        self._update_metrics(quote, book)

        # Atualizar estado do inventário
        self._update_inventory_state(symbol)

        # Verificar se deve fechar posição (timeout ou risco)
        exit_signal = self._check_exit_conditions(symbol, quote)
        if exit_signal:
            return exit_signal

        # Calcular fair value e microprice
        self._fair_value = self._calculate_fair_value(quote, book)
        self._microprice = self._calculate_microprice(quote, book)

        # Decidir se deve entrar
        return self._generate_market_making_signal(symbol, quote, book)

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa atualização de cotação"""
        return None  # Precisa de tick completo com book

    def _validate_market_data(self, quote: Quote, book: OrderBook) -> bool:
        """Valida qualidade dos dados de mercado"""
        # Verificar preços válidos
        if quote.bid_price <= 0 or quote.ask_price <= 0:
            return False

        # Verificar spread razoável
        spread_pips = quote.spread / self.pip_size
        if spread_pips > self.max_spread_pips * 3:  # Spread muito alto = mercado instável
            return False

        # Verificar profundidade do book
        if book.bid_depth < self.min_book_depth or book.ask_depth < self.min_book_depth:
            return False

        return True

    def _update_metrics(self, quote: Quote, book: OrderBook) -> None:
        """Atualiza métricas de mercado"""
        mid = quote.mid_price
        if mid <= 0:
            return

        self._price_history.append(mid)
        self._spread_history.append(quote.spread)

        # Volatilidade (desvio padrão dos retornos)
        if len(self._price_history) >= 20:
            prices = list(self._price_history)
            returns = np.diff(prices) / prices[:-1]
            self._volatility = np.std(returns) * 100  # Em percentual

        # Spread médio
        if self._spread_history:
            self._avg_spread = np.mean(list(self._spread_history))

    def _update_inventory_state(self, symbol: str) -> None:
        """Atualiza estado do inventário"""
        position = self.get_position(symbol)
        if position:
            self._inventory.position = position.size if position.side == 'long' else -position.size
        else:
            self._inventory.position = 0.0

        # Calcular fator de skew baseado no inventário
        # Quanto mais inventário, mais agressivo para reduzir
        inventory_ratio = self._inventory.position / self._inventory.max_inventory
        self._inventory.skew_factor = inventory_ratio * self.inventory_skew_factor

    def _calculate_fair_value(self, quote: Quote, book: OrderBook) -> float:
        """Calcula fair value baseado no book"""
        # Usar VWAP dos primeiros níveis
        bid_vwap = book.get_vwap(BookSide.BID, 0.1) or quote.bid_price
        ask_vwap = book.get_vwap(BookSide.ASK, 0.1) or quote.ask_price

        # Fair value é média ponderada pelo volume
        total_bid_vol = sum(level.size for level in book.get_bids(5))
        total_ask_vol = sum(level.size for level in book.get_asks(5))

        if total_bid_vol + total_ask_vol > 0:
            weight = total_bid_vol / (total_bid_vol + total_ask_vol)
            return bid_vwap * weight + ask_vwap * (1 - weight)

        return quote.mid_price

    def _calculate_microprice(self, quote: Quote, book: OrderBook) -> float:
        """
        Calcula microprice (previsão de curto prazo)
        Microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        """
        if quote.bid_size + quote.ask_size > 0:
            return (quote.bid_price * quote.ask_size +
                   quote.ask_price * quote.bid_size) / (quote.bid_size + quote.ask_size)
        return quote.mid_price

    def _generate_market_making_signal(self, symbol: str, quote: Quote,
                                        book: OrderBook) -> Optional[Signal]:
        """Gera sinal de market making"""
        if not self.can_generate_signal():
            return None

        # Já tem posição máxima?
        if abs(self._inventory.position) >= self._inventory.max_inventory:
            return None

        # Verificar imbalance - evitar entrar em mercado unilateral
        imbalance = book.get_imbalance()
        if abs(imbalance) > self.max_imbalance:
            return None

        spread_pips = quote.spread / self.pip_size

        # Só opera se spread for favorável
        if spread_pips < self.min_spread_pips:
            return None

        # Calcular preços de quote ajustados pelo inventário
        skew = self._inventory.skew_factor * self.pip_size

        # Decisão baseada em microprice vs fair value
        price_signal = self._microprice - self._fair_value

        # Ajustar por volatilidade
        vol_adjustment = self._volatility * self.volatility_multiplier * self.pip_size

        # Sinal de compra: microprice > fair value (pressão compradora)
        # E inventário não está muito long
        if (price_signal > vol_adjustment and
            self._inventory.position < self._inventory.max_inventory * 0.5 and
            imbalance > 0.1):  # Mais bids que asks

            entry_price = quote.bid_price  # Comprar no bid

            # Ajustar SL/TP
            stop_loss = entry_price - (self.stop_loss_pips * self.pip_size)
            take_profit = entry_price + (self.take_profit_pips * self.pip_size)

            confidence = min(1.0, 0.5 + abs(imbalance) + (spread_pips / 10))

            return self._emit_signal(Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=quote.ask_price,  # Executa no ask
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'strategy': 'market_making',
                    'microprice': self._microprice,
                    'fair_value': self._fair_value,
                    'imbalance': imbalance,
                    'spread_pips': spread_pips,
                    'inventory': self._inventory.position,
                    'volatility': self._volatility
                }
            ))

        # Sinal de venda: microprice < fair value (pressão vendedora)
        # E inventário não está muito short
        if (price_signal < -vol_adjustment and
            self._inventory.position > -self._inventory.max_inventory * 0.5 and
            imbalance < -0.1):  # Mais asks que bids

            entry_price = quote.ask_price  # Vender no ask

            stop_loss = entry_price + (self.stop_loss_pips * self.pip_size)
            take_profit = entry_price - (self.take_profit_pips * self.pip_size)

            confidence = min(1.0, 0.5 + abs(imbalance) + (spread_pips / 10))

            return self._emit_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=quote.bid_price,  # Executa no bid
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'strategy': 'market_making',
                    'microprice': self._microprice,
                    'fair_value': self._fair_value,
                    'imbalance': imbalance,
                    'spread_pips': spread_pips,
                    'inventory': self._inventory.position,
                    'volatility': self._volatility
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

        # Stop Loss
        if pnl_pips <= -self.stop_loss_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            self._adverse_selection_cost += abs(pnl_pips) * self.pip_size

            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.VERY_STRONG,
                confidence=1.0,
                metadata={'reason': 'stop_loss', 'pnl_pips': pnl_pips}
            ))

        # Take Profit
        if pnl_pips >= self.take_profit_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            self._spread_captured += pnl_pips * self.pip_size

            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.STRONG,
                confidence=1.0,
                metadata={'reason': 'take_profit', 'pnl_pips': pnl_pips}
            ))

        # Timeout - fechar se posição está aberta por muito tempo
        holding_time_ms = (time.time() - position.entry_time) * 1000
        if holding_time_ms > self.max_holding_time_ms and pnl_pips > 0:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT

            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.MODERATE,
                confidence=0.8,
                metadata={'reason': 'timeout', 'pnl_pips': pnl_pips, 'holding_ms': holding_time_ms}
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
        """Métricas da estratégia"""
        return {
            'volatility': self._volatility,
            'avg_spread': self._avg_spread,
            'fair_value': self._fair_value,
            'microprice': self._microprice,
            'inventory': self._inventory.position,
            'skew_factor': self._inventory.skew_factor,
            'spread_captured': self._spread_captured,
            'adverse_selection_cost': self._adverse_selection_cost
        }
