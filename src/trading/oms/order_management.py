"""
Order Management System - Sistema de gerenciamento de ordens HFT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
import threading
import asyncio
import logging

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Status da ordem"""
    PENDING = 'pending'
    NEW = 'new'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    PENDING_CANCEL = 'pending_cancel'
    PENDING_REPLACE = 'pending_replace'


class OrderType(Enum):
    """Tipo de ordem"""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


class TimeInForce(Enum):
    """Tempo de validade"""
    IOC = 'ioc'  # Immediate or Cancel
    FOK = 'fok'  # Fill or Kill
    GTC = 'gtc'  # Good Till Cancel
    DAY = 'day'  # Day order


class Side(Enum):
    """Lado da ordem"""
    BUY = 'buy'
    SELL = 'sell'


@dataclass
class Order:
    """Representação de uma ordem"""
    id: str = ''
    client_order_id: str = ''
    symbol: str = ''
    side: Side = Side.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    time_in_force: TimeInForce = TimeInForce.IOC
    status: OrderStatus = OrderStatus.PENDING

    # Execução
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0

    # Timestamps
    created_at: float = field(default_factory=time.time)
    submitted_at: float = 0.0
    filled_at: float = 0.0
    cancelled_at: float = 0.0

    # Metadados
    strategy_name: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Latência
    submit_latency_ns: int = 0
    fill_latency_ns: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.client_order_id:
            self.client_order_id = f"ELI_{int(time.time()*1000)}_{self.id[:8]}"

    @property
    def remaining_quantity(self) -> float:
        """Quantidade restante"""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Verifica se ordem está ativa"""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        )

    @property
    def is_done(self) -> bool:
        """Verifica se ordem está finalizada"""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        )

    @property
    def fill_ratio(self) -> float:
        """Razão de preenchimento"""
        if self.quantity > 0:
            return self.filled_quantity / self.quantity
        return 0.0


@dataclass
class Fill:
    """Registro de execução"""
    order_id: str
    fill_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: float
    liquidity: str = 'taker'  # maker ou taker


@dataclass
class Position:
    """Posição aberta"""
    symbol: str
    side: str  # 'long' ou 'short'
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    open_time: float = field(default_factory=time.time)

    def update_pnl(self, current_price: float) -> None:
        """Atualiza P&L não realizado"""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.quantity


class OrderManagementSystem:
    """
    Sistema de Gerenciamento de Ordens para HFT

    Responsabilidades:
    - Criação e validação de ordens
    - Envio para execução
    - Tracking de status
    - Gerenciamento de posições
    - Histórico de fills
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa OMS

        Args:
            config: Configuração do OMS
        """
        self.config = config or {}

        # Ordens
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []

        # Posições
        self._positions: Dict[str, Position] = {}

        # Fills
        self._fills: List[Fill] = []

        # Callbacks
        self._on_order_update: List[Callable[[Order], None]] = []
        self._on_fill: List[Callable[[Fill], None]] = []
        self._on_position_update: List[Callable[[Position], None]] = []

        # Executor de ordens (a ser injetado)
        self._order_executor: Optional[Callable] = None

        # Lock
        self._lock = threading.Lock()

        # Estatísticas
        self._orders_submitted = 0
        self._orders_filled = 0
        self._orders_cancelled = 0
        self._orders_rejected = 0
        self._total_volume = 0.0
        self._total_commission = 0.0

        logger.info("OMS inicializado")

    def set_order_executor(self, executor: Callable) -> None:
        """Define executor de ordens"""
        self._order_executor = executor

    def register_callback(self, event: str, callback: Callable) -> None:
        """Registra callback"""
        if event == 'on_order_update':
            self._on_order_update.append(callback)
        elif event == 'on_fill':
            self._on_fill.append(callback)
        elif event == 'on_position_update':
            self._on_position_update.append(callback)

    def _emit(self, event: str, data: Any) -> None:
        """Emite evento"""
        callbacks = {
            'on_order_update': self._on_order_update,
            'on_fill': self._on_fill,
            'on_position_update': self._on_position_update
        }.get(event, [])

        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Erro no callback: {e}")

    def create_order(self, symbol: str, side: Side, quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: float = 0.0, stop_price: float = 0.0,
                    time_in_force: TimeInForce = TimeInForce.IOC,
                    strategy_name: str = '', metadata: Dict = None) -> Order:
        """
        Cria nova ordem

        Args:
            symbol: Símbolo
            side: Lado (BUY/SELL)
            quantity: Quantidade
            order_type: Tipo de ordem
            price: Preço limite
            stop_price: Preço de stop
            time_in_force: Validade
            strategy_name: Nome da estratégia
            metadata: Metadados adicionais

        Returns:
            Ordem criada
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_name=strategy_name,
            metadata=metadata or {}
        )

        with self._lock:
            self._orders[order.id] = order

        logger.debug(f"Ordem criada: {order.client_order_id} {side.value} {quantity} {symbol}")
        return order

    async def submit_order(self, order: Order) -> bool:
        """
        Submete ordem para execução

        Args:
            order: Ordem a submeter

        Returns:
            True se submetida com sucesso
        """
        start_ns = time.perf_counter_ns()

        with self._lock:
            if order.id not in self._orders:
                logger.error(f"Ordem não encontrada: {order.id}")
                return False

            order.status = OrderStatus.PENDING
            order.submitted_at = time.time()
            self._active_orders[order.id] = order

        self._orders_submitted += 1

        # Executar ordem
        if self._order_executor:
            try:
                success = await self._order_executor(order)
                order.submit_latency_ns = time.perf_counter_ns() - start_ns

                if success:
                    order.status = OrderStatus.NEW
                    logger.debug(f"Ordem submetida: {order.client_order_id} "
                               f"latency={order.submit_latency_ns/1000:.2f}µs")
                else:
                    order.status = OrderStatus.REJECTED
                    self._orders_rejected += 1

                self._emit('on_order_update', order)
                return success

            except Exception as e:
                logger.error(f"Erro ao submeter ordem: {e}")
                order.status = OrderStatus.REJECTED
                self._orders_rejected += 1
                return False
        else:
            # Simulação (para testes)
            order.status = OrderStatus.NEW
            order.submit_latency_ns = time.perf_counter_ns() - start_ns
            self._emit('on_order_update', order)
            return True

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancela ordem

        Args:
            order_id: ID da ordem

        Returns:
            True se cancelamento iniciado
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.error(f"Ordem não encontrada: {order_id}")
                return False

            if not order.is_active:
                logger.warning(f"Ordem não está ativa: {order_id}")
                return False

            order.status = OrderStatus.PENDING_CANCEL

        # TODO: Implementar cancelamento real via executor
        logger.info(f"Cancelamento solicitado: {order.client_order_id}")
        return True

    async def cancel_all_orders(self, symbol: str = None) -> int:
        """
        Cancela todas as ordens ativas

        Args:
            symbol: Símbolo específico (opcional)

        Returns:
            Número de ordens canceladas
        """
        cancelled = 0

        with self._lock:
            orders_to_cancel = [
                o for o in self._active_orders.values()
                if o.is_active and (symbol is None or o.symbol == symbol)
            ]

        for order in orders_to_cancel:
            if await self.cancel_order(order.id):
                cancelled += 1

        logger.info(f"Canceladas {cancelled} ordens")
        return cancelled

    def on_execution_report(self, order_id: str, status: OrderStatus,
                           filled_qty: float = 0, avg_price: float = 0,
                           commission: float = 0) -> None:
        """
        Processa relatório de execução

        Args:
            order_id: ID da ordem
            status: Novo status
            filled_qty: Quantidade preenchida
            avg_price: Preço médio
            commission: Comissão
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"Ordem não encontrada: {order_id}")
                return

            old_status = order.status
            order.status = status
            order.filled_quantity = filled_qty
            order.avg_price = avg_price
            order.commission = commission

            # Atualizar estatísticas
            if status == OrderStatus.FILLED:
                order.filled_at = time.time()
                order.fill_latency_ns = int((order.filled_at - order.submitted_at) * 1e9)
                self._orders_filled += 1
                self._total_volume += filled_qty * avg_price
                self._total_commission += commission

                # Remover de ativos
                self._active_orders.pop(order_id, None)
                self._order_history.append(order)

                # Criar fill
                fill = Fill(
                    order_id=order_id,
                    fill_id=str(uuid.uuid4()),
                    symbol=order.symbol,
                    side=order.side,
                    quantity=filled_qty,
                    price=avg_price,
                    commission=commission,
                    timestamp=time.time()
                )
                self._fills.append(fill)
                self._emit('on_fill', fill)

                # Atualizar posição
                self._update_position(order)

            elif status == OrderStatus.CANCELLED:
                order.cancelled_at = time.time()
                self._orders_cancelled += 1
                self._active_orders.pop(order_id, None)

            elif status == OrderStatus.REJECTED:
                self._orders_rejected += 1
                self._active_orders.pop(order_id, None)

        self._emit('on_order_update', order)
        logger.debug(f"Ordem {order.client_order_id}: {old_status.value} -> {status.value}")

    def _update_position(self, order: Order) -> None:
        """Atualiza posição após fill"""
        symbol = order.symbol
        position = self._positions.get(symbol)

        filled_qty = order.filled_quantity
        avg_price = order.avg_price

        if position is None:
            # Nova posição
            side = 'long' if order.side == Side.BUY else 'short'
            position = Position(
                symbol=symbol,
                side=side,
                quantity=filled_qty,
                avg_entry_price=avg_price
            )
            self._positions[symbol] = position

        else:
            # Atualizar posição existente
            if order.side == Side.BUY:
                if position.side == 'long':
                    # Aumentar long
                    total_qty = position.quantity + filled_qty
                    position.avg_entry_price = (
                        (position.avg_entry_price * position.quantity + avg_price * filled_qty)
                        / total_qty
                    )
                    position.quantity = total_qty
                else:
                    # Reduzir short
                    position.quantity -= filled_qty
                    if position.quantity <= 0:
                        # Fechar e inverter
                        remaining = abs(position.quantity)
                        if remaining > 0:
                            position.side = 'long'
                            position.quantity = remaining
                            position.avg_entry_price = avg_price
                        else:
                            del self._positions[symbol]
                            position = None

            else:  # SELL
                if position.side == 'short':
                    # Aumentar short
                    total_qty = position.quantity + filled_qty
                    position.avg_entry_price = (
                        (position.avg_entry_price * position.quantity + avg_price * filled_qty)
                        / total_qty
                    )
                    position.quantity = total_qty
                else:
                    # Reduzir long
                    position.quantity -= filled_qty
                    if position.quantity <= 0:
                        remaining = abs(position.quantity)
                        if remaining > 0:
                            position.side = 'short'
                            position.quantity = remaining
                            position.avg_entry_price = avg_price
                        else:
                            del self._positions[symbol]
                            position = None

        if position:
            self._emit('on_position_update', position)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Obtém ordem por ID"""
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Obtém ordens ativas"""
        with self._lock:
            orders = list(self._active_orders.values())
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders

    def get_position(self, symbol: str) -> Optional[Position]:
        """Obtém posição"""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Obtém todas as posições"""
        return dict(self._positions)

    def get_fills(self, symbol: str = None, limit: int = 100) -> List[Fill]:
        """Obtém histórico de fills"""
        fills = self._fills[-limit:]
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        return fills

    @property
    def stats(self) -> dict:
        """Estatísticas do OMS"""
        return {
            'orders_submitted': self._orders_submitted,
            'orders_filled': self._orders_filled,
            'orders_cancelled': self._orders_cancelled,
            'orders_rejected': self._orders_rejected,
            'active_orders': len(self._active_orders),
            'open_positions': len(self._positions),
            'total_volume': self._total_volume,
            'total_commission': self._total_commission,
            'fill_rate': self._orders_filled / max(self._orders_submitted, 1)
        }
