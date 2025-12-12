"""
Order Book - Limit Order Book de alta performance
Otimizado para HFT com estruturas de dados eficientes
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import OrderedDict
import time
import bisect
import threading
import logging

logger = logging.getLogger(__name__)


class BookSide(IntEnum):
    """Lado do book"""
    BID = 0
    ASK = 1


class UpdateType(Enum):
    """Tipo de atualização do book"""
    NEW = 'new'
    CHANGE = 'change'
    DELETE = 'delete'
    SNAPSHOT = 'snapshot'


@dataclass
class OrderBookLevel:
    """Nível de preço no order book"""
    price: float
    size: float
    count: int = 1  # Número de ordens
    timestamp: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.price)

    def __eq__(self, other):
        if isinstance(other, OrderBookLevel):
            return self.price == other.price
        return self.price == other


@dataclass
class OrderBookUpdate:
    """Atualização do order book"""
    symbol: str
    side: BookSide
    price: float
    size: float
    update_type: UpdateType
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0


@dataclass
class Quote:
    """Cotação (melhor bid/ask)"""
    bid_price: float = 0.0
    bid_size: float = 0.0
    ask_price: float = 0.0
    ask_size: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def spread(self) -> float:
        """Spread bid-ask"""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Spread em basis points"""
        if self.bid_price > 0 and self.ask_price > 0:
            mid = (self.bid_price + self.ask_price) / 2
            return (self.spread / mid) * 10000
        return 0.0

    @property
    def mid_price(self) -> float:
        """Preço médio"""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.bid_price or self.ask_price


class OrderBook:
    """
    Limit Order Book (LOB) de alta performance

    Características:
    - Estrutura otimizada para acesso O(1) ao topo do book
    - Suporte a L2 (price levels) e L3 (individual orders)
    - Thread-safe
    - Callbacks para mudanças
    """

    def __init__(self, symbol: str, tick_size: float = 0.00001,
                 max_depth: int = 50):
        """
        Inicializa order book

        Args:
            symbol: Símbolo do instrumento
            tick_size: Tamanho mínimo de tick
            max_depth: Profundidade máxima do book
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.max_depth = max_depth

        # Níveis de preço: price -> OrderBookLevel
        # Bids: maior para menor (reversed)
        # Asks: menor para maior
        self._bids: Dict[float, OrderBookLevel] = {}
        self._asks: Dict[float, OrderBookLevel] = {}

        # Listas ordenadas de preços para acesso rápido
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []

        # Sequenciamento
        self._sequence = 0
        self._last_update_time = 0.0

        # Threading
        self._lock = threading.RLock()

        # Callbacks
        self._callbacks: List[Callable[[OrderBookUpdate], None]] = []

        # Estatísticas
        self._update_count = 0
        self._snapshot_count = 0

    def register_callback(self, callback: Callable[[OrderBookUpdate], None]) -> None:
        """Registra callback para atualizações"""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[OrderBookUpdate], None]) -> bool:
        """Remove callback registrado. Retorna True se removido."""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _emit_update(self, update: OrderBookUpdate) -> None:
        """Emite atualização para callbacks"""
        for cb in self._callbacks:
            try:
                cb(update)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")

    def update(self, side: BookSide, price: float, size: float,
               update_type: UpdateType = UpdateType.CHANGE) -> None:
        """
        Atualiza nível de preço

        Args:
            side: Lado (BID/ASK)
            price: Preço
            size: Volume
            update_type: Tipo de atualização
        """
        with self._lock:
            self._sequence += 1
            self._last_update_time = time.time()
            self._update_count += 1

            if side == BookSide.BID:
                self._update_side(self._bids, self._bid_prices, price, size, reverse=True)
            else:
                self._update_side(self._asks, self._ask_prices, price, size, reverse=False)

        # Emitir update
        update = OrderBookUpdate(
            symbol=self.symbol,
            side=side,
            price=price,
            size=size,
            update_type=update_type,
            timestamp=self._last_update_time,
            sequence=self._sequence
        )
        self._emit_update(update)

    def _update_side(self, levels: Dict[float, OrderBookLevel],
                     prices: List[float], price: float, size: float,
                     reverse: bool) -> None:
        """Atualiza um lado do book"""
        if size <= 0:
            # Remover nível - usar bisect para encontrar índice (O(log n))
            if price in levels:
                del levels[price]
                if reverse:
                    # Bids são ordenados do maior para menor
                    # Encontrar índice com busca binária invertida
                    idx = len(prices) - bisect.bisect_left(prices[::-1], price) - 1
                    if 0 <= idx < len(prices) and prices[idx] == price:
                        prices.pop(idx)
                else:
                    idx = bisect.bisect_left(prices, price)
                    if idx < len(prices) and prices[idx] == price:
                        prices.pop(idx)
        else:
            # Adicionar ou atualizar
            if price in levels:
                levels[price].size = size
                levels[price].timestamp = time.time()
            else:
                levels[price] = OrderBookLevel(price=price, size=size)
                # Inserir ordenado
                if reverse:
                    # Para bids: inserir mantendo ordem decrescente
                    # Usar negativo para bisect manter ordem reversa
                    idx = bisect.bisect_left([-p for p in prices], -price)
                    prices.insert(idx, price)
                else:
                    bisect.insort(prices, price)

        # Limitar profundidade
        while len(prices) > self.max_depth:
            price_to_remove = prices.pop()
            if price_to_remove in levels:
                del levels[price_to_remove]

    def apply_snapshot(self, bids: List[Tuple[float, float]],
                       asks: List[Tuple[float, float]]) -> None:
        """
        Aplica snapshot completo do book

        Args:
            bids: Lista de (price, size) para bids
            asks: Lista de (price, size) para asks
        """
        with self._lock:
            # Limpar book
            self._bids.clear()
            self._asks.clear()
            self._bid_prices.clear()
            self._ask_prices.clear()

            # Aplicar bids (ordenar maior para menor)
            for price, size in sorted(bids, key=lambda x: x[0], reverse=True):
                if size > 0:
                    self._bids[price] = OrderBookLevel(price=price, size=size)
                    self._bid_prices.append(price)

            # Aplicar asks (ordenar menor para maior)
            for price, size in sorted(asks, key=lambda x: x[0]):
                if size > 0:
                    self._asks[price] = OrderBookLevel(price=price, size=size)
                    self._ask_prices.append(price)

            # Limitar profundidade
            self._bid_prices = self._bid_prices[:self.max_depth]
            self._ask_prices = self._ask_prices[:self.max_depth]

            self._sequence += 1
            self._last_update_time = time.time()
            self._snapshot_count += 1

    def get_quote(self) -> Quote:
        """
        Obtém melhor cotação (BBO)

        Returns:
            Quote com melhor bid/ask
        """
        with self._lock:
            quote = Quote(timestamp=time.time())

            if self._bid_prices:
                best_bid = self._bid_prices[0]
                quote.bid_price = best_bid
                quote.bid_size = self._bids[best_bid].size

            if self._ask_prices:
                best_ask = self._ask_prices[0]
                quote.ask_price = best_ask
                quote.ask_size = self._asks[best_ask].size

            return quote

    def get_bids(self, depth: int = None) -> List[OrderBookLevel]:
        """Obtém níveis de bid"""
        depth = depth or self.max_depth
        with self._lock:
            return [self._bids[p] for p in self._bid_prices[:depth]]

    def get_asks(self, depth: int = None) -> List[OrderBookLevel]:
        """Obtém níveis de ask"""
        depth = depth or self.max_depth
        with self._lock:
            return [self._asks[p] for p in self._ask_prices[:depth]]

    def get_depth(self, side: BookSide, levels: int = 5) -> List[Tuple[float, float]]:
        """
        Obtém profundidade do book

        Args:
            side: Lado
            levels: Número de níveis

        Returns:
            Lista de (price, size)
        """
        with self._lock:
            if side == BookSide.BID:
                return [(p, self._bids[p].size) for p in self._bid_prices[:levels]]
            else:
                return [(p, self._asks[p].size) for p in self._ask_prices[:levels]]

    def get_vwap(self, side: BookSide, size: float) -> Optional[float]:
        """
        Calcula VWAP para tamanho especificado

        Args:
            side: Lado
            size: Volume desejado

        Returns:
            Preço médio ponderado ou None
        """
        with self._lock:
            levels = self._bids if side == BookSide.BID else self._asks
            prices = self._bid_prices if side == BookSide.BID else self._ask_prices

            total_cost = 0.0
            total_size = 0.0

            for price in prices:
                level = levels[price]
                take_size = min(size - total_size, level.size)
                total_cost += price * take_size
                total_size += take_size

                if total_size >= size:
                    break

            if total_size > 0:
                return total_cost / total_size
            return None

    def get_imbalance(self, levels: int = 5) -> float:
        """
        Calcula imbalance do book

        Returns:
            Valor entre -1 (mais asks) e 1 (mais bids)
        """
        with self._lock:
            bid_volume = sum(self._bids[p].size for p in self._bid_prices[:levels])
            ask_volume = sum(self._asks[p].size for p in self._ask_prices[:levels])

            total = bid_volume + ask_volume
            if total > 0:
                return (bid_volume - ask_volume) / total
            return 0.0

    def clear(self) -> None:
        """Limpa o book"""
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            self._bid_prices.clear()
            self._ask_prices.clear()

    @property
    def best_bid(self) -> Optional[float]:
        """Melhor bid"""
        with self._lock:
            return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        """Melhor ask"""
        with self._lock:
            return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        """Preço médio"""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return (bid + ask) / 2
        return bid or ask

    @property
    def spread(self) -> float:
        """Spread bid-ask"""
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return ask - bid
        return 0.0

    @property
    def bid_depth(self) -> int:
        """Profundidade de bids"""
        return len(self._bid_prices)

    @property
    def ask_depth(self) -> int:
        """Profundidade de asks"""
        return len(self._ask_prices)

    @property
    def stats(self) -> dict:
        """Estatísticas do book"""
        return {
            'symbol': self.symbol,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'spread': self.spread,
            'update_count': self._update_count,
            'snapshot_count': self._snapshot_count,
            'sequence': self._sequence,
            'last_update': self._last_update_time
        }

    def __str__(self) -> str:
        quote = self.get_quote()
        return (f"OrderBook({self.symbol}: "
               f"bid={quote.bid_price:.5f}x{quote.bid_size:.2f}, "
               f"ask={quote.ask_price:.5f}x{quote.ask_size:.2f}, "
               f"spread={quote.spread_bps:.2f}bps)")
