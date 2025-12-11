"""
FIX Engine - Motor FIX completo para HFT
Gerencia sessões, sequenciamento e reconexão
"""

import asyncio
import time
import threading
from typing import Dict, Optional, Callable, List, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

from .fix_message import FIXMessage, FIXMessageType, FIXField
from .fix_parser import FIXParser
from ..network import TCPClient

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Estados da sessão FIX"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LOGGED_IN = "logged_in"
    LOGGING_OUT = "logging_out"
    ERROR = "error"


@dataclass
class SessionConfig:
    """Configuração da sessão FIX"""
    begin_string: str = 'FIX.4.4'
    sender_comp_id: str = ''
    target_comp_id: str = ''
    host: str = ''
    port: int = 0
    username: str = ''
    password: str = ''
    heartbeat_interval: int = 30
    use_ssl: bool = True
    reset_on_logon: bool = True
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class SessionStats:
    """Estatísticas da sessão"""
    messages_sent: int = 0
    messages_received: int = 0
    heartbeats_sent: int = 0
    heartbeats_received: int = 0
    logon_time: float = 0.0
    last_sent_time: float = 0.0
    last_received_time: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    reconnect_count: int = 0


class FIXEngine:
    """
    Motor FIX de alta performance
    Gerencia sessão completa com sequenciamento e heartbeat
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self._state = SessionState.DISCONNECTED
        self._stats = SessionStats()

        # Sequenciamento
        self._outgoing_seq_num = 1
        self._incoming_seq_num = 1
        self._message_store: Dict[int, FIXMessage] = {}

        # Network
        self._client: Optional[TCPClient] = None
        self._parser = FIXParser()

        # Threading/Async
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_logon': [],
            'on_logout': [],
            'on_message': [],
            'on_execution_report': [],
            'on_market_data': [],
            'on_error': [],
            'on_heartbeat': []
        }

        # Pending requests
        self._pending_requests: Dict[str, asyncio.Future] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Registra callback para eventos"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emite evento"""
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def connect(self) -> bool:
        """
        Conecta à sessão FIX
        """
        if self._state != SessionState.DISCONNECTED:
            logger.warning("Sessão já conectada ou conectando")
            return False

        self._state = SessionState.CONNECTING

        try:
            # Criar cliente TCP
            self._client = TCPClient(
                self.config.host,
                self.config.port,
                self.config.use_ssl
            )

            # Conectar
            if not await self._client.connect_async():
                self._state = SessionState.ERROR
                return False

            self._state = SessionState.CONNECTED

            # Iniciar receiver
            self._running = True
            self._receiver_task = asyncio.create_task(self._receiver_loop())

            # Enviar Logon
            return await self._send_logon()

        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            self._state = SessionState.ERROR
            self._emit('on_error', str(e))
            return False

    async def disconnect(self) -> None:
        """
        Desconecta da sessão
        """
        if self._state not in (SessionState.LOGGED_IN, SessionState.CONNECTED):
            return

        self._state = SessionState.LOGGING_OUT
        self._running = False

        try:
            # Enviar Logout
            await self._send_logout()

            # Cancelar tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._receiver_task:
                self._receiver_task.cancel()

            # Fechar conexão
            if self._client:
                await self._client.disconnect_async()

        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")

        self._state = SessionState.DISCONNECTED
        self._emit('on_logout')

    async def _send_logon(self) -> bool:
        """Envia mensagem de Logon"""
        msg = FIXMessage.logon(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            heartbeat_int=self.config.heartbeat_interval,
            username=self.config.username,
            password=self.config.password
        )

        return await self._send_message(msg)

    async def _send_logout(self, text: str = None) -> bool:
        """Envia mensagem de Logout"""
        msg = FIXMessage.logout(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            text=text
        )

        return await self._send_message(msg)

    async def _send_heartbeat(self, test_req_id: str = None) -> bool:
        """Envia Heartbeat"""
        msg = FIXMessage.heartbeat(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            test_req_id=test_req_id
        )

        self._stats.heartbeats_sent += 1
        return await self._send_message(msg)

    async def _send_message(self, msg: FIXMessage) -> bool:
        """
        Envia mensagem FIX
        """
        if not self._client or not self._client.is_connected:
            logger.error("Cliente não conectado")
            return False

        try:
            # Build e enviar
            data = msg.build()
            start_time = time.perf_counter()

            if not await self._client.send_async(data):
                return False

            latency = time.perf_counter() - start_time

            # Atualizar stats
            with self._lock:
                self._stats.messages_sent += 1
                self._stats.last_sent_time = time.time()
                self._stats.latencies.append(latency)
                self._message_store[msg.msg_seq_num] = msg

            logger.debug(f"Enviado: {msg.msg_type} seq={msg.msg_seq_num}")
            return True

        except Exception as e:
            logger.error(f"Erro ao enviar: {e}")
            return False

    async def _receiver_loop(self) -> None:
        """Loop de recepção de mensagens"""
        while self._running:
            try:
                data = await self._client.recv_async()
                if not data:
                    if self._running:
                        logger.warning("Conexão fechada pelo servidor")
                        await self._handle_disconnect()
                    break

                # Parsear mensagens
                self._parser.feed(data)

                for msg in self._parser.parse_all():
                    await self._handle_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no receiver: {e}")
                if self._running:
                    await asyncio.sleep(0.1)

    async def _handle_message(self, msg: FIXMessage) -> None:
        """Processa mensagem recebida"""
        with self._lock:
            self._stats.messages_received += 1
            self._stats.last_received_time = time.time()
            self._incoming_seq_num = msg.msg_seq_num + 1

        msg_type = msg.msg_type

        # Logon
        if msg_type == FIXMessageType.LOGON.value:
            self._state = SessionState.LOGGED_IN
            self._stats.logon_time = time.time()
            logger.info("Logon bem-sucedido")

            # Iniciar heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._emit('on_logon')

        # Logout
        elif msg_type == FIXMessageType.LOGOUT.value:
            logger.info("Logout recebido")
            await self.disconnect()

        # Heartbeat
        elif msg_type == FIXMessageType.HEARTBEAT.value:
            self._stats.heartbeats_received += 1
            self._emit('on_heartbeat')

        # Test Request
        elif msg_type == FIXMessageType.TEST_REQUEST.value:
            test_req_id = msg.get_field(112)
            await self._send_heartbeat(test_req_id)

        # Execution Report
        elif msg_type == FIXMessageType.EXECUTION_REPORT.value:
            self._emit('on_execution_report', msg)

        # Market Data
        elif msg_type in (FIXMessageType.MARKET_DATA_SNAPSHOT.value,
                         FIXMessageType.MARKET_DATA_INCREMENTAL.value):
            self._emit('on_market_data', msg)

        # Reject
        elif msg_type == FIXMessageType.REJECT.value:
            logger.warning(f"Reject recebido: {msg.get_field(FIXField.TEXT)}")
            self._emit('on_error', f"Reject: {msg.get_field(FIXField.TEXT)}")

        # Emitir evento genérico
        self._emit('on_message', msg)

    async def _heartbeat_loop(self) -> None:
        """Loop de heartbeat"""
        while self._running and self._state == SessionState.LOGGED_IN:
            await asyncio.sleep(self.config.heartbeat_interval)
            if self._running:
                await self._send_heartbeat()

    async def _handle_disconnect(self) -> None:
        """Trata desconexão inesperada"""
        if self._state == SessionState.LOGGING_OUT:
            return

        self._state = SessionState.ERROR
        self._stats.reconnect_count += 1

        logger.warning(f"Desconexão detectada. Tentativa de reconexão {self._stats.reconnect_count}")

        # Tentar reconectar
        for attempt in range(self.config.max_reconnect_attempts):
            await asyncio.sleep(self.config.reconnect_interval * (2 ** attempt))

            if await self.connect():
                logger.info("Reconectado com sucesso")
                return

        logger.error("Falha ao reconectar após máximo de tentativas")
        self._emit('on_error', "Reconexão falhou")

    def _get_next_seq_num(self) -> int:
        """Obtém próximo sequence number"""
        with self._lock:
            seq = self._outgoing_seq_num
            self._outgoing_seq_num += 1
            return seq

    # ==================== Trading Methods ====================

    async def send_new_order(self, cl_ord_id: str, symbol: str, side: int,
                            qty: float, ord_type: int, price: float = None,
                            stop_px: float = None, time_in_force: int = 3) -> bool:
        """
        Envia nova ordem
        """
        if self._state != SessionState.LOGGED_IN:
            logger.error("Sessão não está logada")
            return False

        from .fix_message import Side, OrdType, TimeInForce

        msg = FIXMessage.new_order_single(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            cl_ord_id=cl_ord_id,
            symbol=symbol,
            side=Side(side),
            order_qty=qty,
            ord_type=OrdType(ord_type),
            price=price,
            stop_px=stop_px,
            time_in_force=TimeInForce(time_in_force)
        )

        return await self._send_message(msg)

    async def cancel_order(self, cl_ord_id: str, orig_cl_ord_id: str,
                          symbol: str, side: int) -> bool:
        """
        Cancela ordem existente
        """
        if self._state != SessionState.LOGGED_IN:
            return False

        from .fix_message import Side

        msg = FIXMessage.order_cancel_request(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            cl_ord_id=cl_ord_id,
            orig_cl_ord_id=orig_cl_ord_id,
            symbol=symbol,
            side=Side(side)
        )

        return await self._send_message(msg)

    async def subscribe_market_data(self, symbols: List[str],
                                   market_depth: int = 0) -> bool:
        """
        Inscreve para receber market data
        """
        if self._state != SessionState.LOGGED_IN:
            return False

        md_req_id = f"MD_{int(time.time()*1000)}"

        msg = FIXMessage.market_data_request(
            sender=self.config.sender_comp_id,
            target=self.config.target_comp_id,
            seq_num=self._get_next_seq_num(),
            md_req_id=md_req_id,
            symbols=symbols,
            subscription_type=1,
            market_depth=market_depth
        )

        return await self._send_message(msg)

    # ==================== Properties ====================

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def is_logged_in(self) -> bool:
        return self._state == SessionState.LOGGED_IN

    @property
    def avg_latency_ms(self) -> float:
        if not self._stats.latencies:
            return 0.0
        return sum(self._stats.latencies) / len(self._stats.latencies) * 1000
