"""
TCP Client - Cliente TCP de baixa latência para HFT
Otimizado para Windows 11
"""

import asyncio
import socket
import ssl
import time
import struct
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


class TCPClientState(Enum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 3


@dataclass
class TCPStats:
    """Estatísticas de conexão TCP"""
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    send_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    recv_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    connection_time_ms: float = 0.0
    last_activity: float = 0.0
    errors: int = 0


class TCPClient:
    """
    Cliente TCP otimizado para baixa latência
    Suporta conexões síncronas e assíncronas
    """

    def __init__(self, host: str, port: int, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl

        self._socket: Optional[socket.socket] = None
        self._ssl_socket: Optional[ssl.SSLSocket] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

        self._state = TCPClientState.DISCONNECTED
        self._stats = TCPStats()
        self._lock = threading.Lock()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_data': [],
            'on_error': []
        }

        # Buffer de recepção
        self._recv_buffer = bytearray()
        self._recv_buffer_size = 65536

        # Configurações de socket
        self._socket_options = {
            'TCP_NODELAY': True,
            'SO_KEEPALIVE': True,
            'SO_RCVBUF': 262144,  # 256KB
            'SO_SNDBUF': 262144,  # 256KB
            'TCP_QUICKACK': True  # Linux only
        }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Registra callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args) -> None:
        """Emite evento"""
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _configure_socket(self, sock: socket.socket) -> None:
        """Configura socket para baixa latência"""
        # Desabilita Nagle's algorithm
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Keep-alive
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Buffers maiores
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                       self._socket_options['SO_RCVBUF'])
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                       self._socket_options['SO_SNDBUF'])

        # TCP_QUICKACK (Linux only)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except (AttributeError, OSError):
            pass

        # Windows: Loopback fast path
        try:
            sock.ioctl(socket.SIO_LOOPBACK_FAST_PATH, True)
        except (AttributeError, OSError):
            pass

    def connect_sync(self, timeout: float = 5.0) -> bool:
        """
        Conexão síncrona (blocking)
        """
        try:
            self._state = TCPClientState.CONNECTING
            start_time = time.perf_counter()

            # Criar socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._configure_socket(self._socket)
            self._socket.settimeout(timeout)

            # SSL wrap se necessário
            if self.use_ssl:
                context = ssl.create_default_context()
                self._ssl_socket = context.wrap_socket(
                    self._socket,
                    server_hostname=self.host
                )
                self._ssl_socket.connect((self.host, self.port))
            else:
                self._socket.connect((self.host, self.port))

            connection_time = (time.perf_counter() - start_time) * 1000
            self._stats.connection_time_ms = connection_time
            self._stats.last_activity = time.time()

            self._state = TCPClientState.CONNECTED
            logger.info(f"TCP conectado a {self.host}:{self.port} em {connection_time:.2f}ms")
            self._emit('on_connect')

            return True

        except Exception as e:
            self._state = TCPClientState.ERROR
            self._stats.errors += 1
            logger.error(f"Erro ao conectar: {e}")
            self._emit('on_error', str(e))
            return False

    async def connect_async(self, timeout: float = 5.0) -> bool:
        """
        Conexão assíncrona
        """
        try:
            self._state = TCPClientState.CONNECTING
            start_time = time.perf_counter()

            ssl_context = None
            if self.use_ssl:
                ssl_context = ssl.create_default_context()

            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.host,
                    self.port,
                    ssl=ssl_context
                ),
                timeout=timeout
            )

            # Configurar socket subjacente
            sock = self._writer.get_extra_info('socket')
            if sock:
                self._configure_socket(sock)

            connection_time = (time.perf_counter() - start_time) * 1000
            self._stats.connection_time_ms = connection_time
            self._stats.last_activity = time.time()

            self._state = TCPClientState.CONNECTED
            logger.info(f"TCP async conectado a {self.host}:{self.port} em {connection_time:.2f}ms")
            self._emit('on_connect')

            return True

        except asyncio.TimeoutError:
            self._state = TCPClientState.ERROR
            self._stats.errors += 1
            logger.error(f"Timeout ao conectar a {self.host}:{self.port}")
            return False
        except Exception as e:
            self._state = TCPClientState.ERROR
            self._stats.errors += 1
            logger.error(f"Erro ao conectar: {e}")
            self._emit('on_error', str(e))
            return False

    def send_sync(self, data: bytes) -> bool:
        """
        Envio síncrono
        """
        if self._state != TCPClientState.CONNECTED:
            return False

        try:
            start_time = time.perf_counter()

            sock = self._ssl_socket if self.use_ssl else self._socket
            total_sent = 0

            while total_sent < len(data):
                sent = sock.send(data[total_sent:])
                if sent == 0:
                    raise RuntimeError("Conexão fechada")
                total_sent += sent

            latency = time.perf_counter() - start_time

            with self._lock:
                self._stats.bytes_sent += len(data)
                self._stats.packets_sent += 1
                self._stats.send_latencies.append(latency)
                self._stats.last_activity = time.time()

            return True

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Erro ao enviar: {e}")
            self._emit('on_error', str(e))
            return False

    async def send_async(self, data: bytes) -> bool:
        """
        Envio assíncrono
        """
        if self._state != TCPClientState.CONNECTED or not self._writer:
            return False

        try:
            start_time = time.perf_counter()

            self._writer.write(data)
            await self._writer.drain()

            latency = time.perf_counter() - start_time

            with self._lock:
                self._stats.bytes_sent += len(data)
                self._stats.packets_sent += 1
                self._stats.send_latencies.append(latency)
                self._stats.last_activity = time.time()

            return True

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Erro ao enviar: {e}")
            self._emit('on_error', str(e))
            return False

    def recv_sync(self, size: int = None) -> Optional[bytes]:
        """
        Recepção síncrona
        """
        if self._state != TCPClientState.CONNECTED:
            return None

        size = size or self._recv_buffer_size

        try:
            start_time = time.perf_counter()

            sock = self._ssl_socket if self.use_ssl else self._socket
            data = sock.recv(size)

            if not data:
                self._state = TCPClientState.DISCONNECTED
                self._emit('on_disconnect')
                return None

            latency = time.perf_counter() - start_time

            with self._lock:
                self._stats.bytes_received += len(data)
                self._stats.packets_received += 1
                self._stats.recv_latencies.append(latency)
                self._stats.last_activity = time.time()

            self._emit('on_data', data)
            return data

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Erro ao receber: {e}")
            return None

    async def recv_async(self, size: int = None) -> Optional[bytes]:
        """
        Recepção assíncrona
        """
        if self._state != TCPClientState.CONNECTED or not self._reader:
            return None

        size = size or self._recv_buffer_size

        try:
            start_time = time.perf_counter()

            data = await self._reader.read(size)

            if not data:
                self._state = TCPClientState.DISCONNECTED
                self._emit('on_disconnect')
                return None

            latency = time.perf_counter() - start_time

            with self._lock:
                self._stats.bytes_received += len(data)
                self._stats.packets_received += 1
                self._stats.recv_latencies.append(latency)
                self._stats.last_activity = time.time()

            self._emit('on_data', data)
            return data

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Erro ao receber: {e}")
            return None

    def disconnect(self) -> None:
        """Desconecta (síncrono)"""
        try:
            if self._ssl_socket:
                self._ssl_socket.close()
            if self._socket:
                self._socket.close()

            self._state = TCPClientState.DISCONNECTED
            self._emit('on_disconnect')
            logger.info(f"TCP desconectado de {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")

    async def disconnect_async(self) -> None:
        """Desconecta (assíncrono)"""
        try:
            if self._writer:
                self._writer.close()
                await self._writer.wait_closed()

            self._state = TCPClientState.DISCONNECTED
            self._emit('on_disconnect')
            logger.info(f"TCP async desconectado de {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")

    @property
    def state(self) -> TCPClientState:
        return self._state

    @property
    def stats(self) -> TCPStats:
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._state == TCPClientState.CONNECTED

    @property
    def avg_send_latency_ms(self) -> float:
        if not self._stats.send_latencies:
            return 0.0
        return sum(self._stats.send_latencies) / len(self._stats.send_latencies) * 1000

    @property
    def avg_recv_latency_ms(self) -> float:
        if not self._stats.recv_latencies:
            return 0.0
        return sum(self._stats.recv_latencies) / len(self._stats.recv_latencies) * 1000
