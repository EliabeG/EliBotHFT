"""
Network Manager - Gerenciador de conexões de rede para HFT
Otimizado para Windows 11 com suporte a conexões assíncronas
"""

import asyncio
import socket
import ssl
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import threading

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class NetworkStats:
    """Estatísticas de rede para monitoramento de latência"""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_heartbeat: float = 0.0
    connection_time: float = 0.0
    reconnect_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies) * 1000

    @property
    def min_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return min(self.latencies) * 1000

    @property
    def max_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return max(self.latencies) * 1000


class NetworkManager:
    """
    Gerenciador central de conexões de rede
    Otimizado para trading de alta frequência
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections: Dict[str, Any] = {}
        self.stats: Dict[str, NetworkStats] = {}
        self.state = ConnectionState.DISCONNECTED
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_message': [],
            'on_error': [],
            'on_latency': []
        }
        self._running = False
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Configurações de socket otimizadas
        self.socket_options = {
            'tcp_nodelay': True,  # Desabilita Nagle's algorithm
            'so_keepalive': True,
            'so_rcvbuf': 65536,  # Buffer de recepção
            'so_sndbuf': 65536,  # Buffer de envio
        }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Registra callback para eventos de rede"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emite evento para todos os callbacks registrados"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Erro no callback {event}: {e}")

    def _configure_socket(self, sock: socket.socket) -> None:
        """Configura socket para baixa latência"""
        # TCP_NODELAY - desabilita buffering
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Keep-alive
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Buffers
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                       self.socket_options['so_rcvbuf'])
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                       self.socket_options['so_sndbuf'])

        # Windows specific optimizations
        try:
            # SIO_LOOPBACK_FAST_PATH para conexões locais
            sock.ioctl(socket.SIO_LOOPBACK_FAST_PATH, True)
        except (AttributeError, OSError):
            pass  # Não disponível em todas as versões

    async def connect(self, name: str, host: str, port: int,
                     use_ssl: bool = True) -> bool:
        """
        Estabelece conexão TCP otimizada
        """
        self.state = ConnectionState.CONNECTING
        self.stats[name] = NetworkStats()

        try:
            start_time = time.perf_counter()

            # Criar socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._configure_socket(sock)
            sock.setblocking(False)

            # SSL context se necessário
            ssl_context = None
            if use_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Conectar assincronamente
            if self._loop is None:
                self._loop = asyncio.get_event_loop()

            reader, writer = await asyncio.open_connection(
                host, port, ssl=ssl_context
            )

            connection_time = time.perf_counter() - start_time

            self.connections[name] = {
                'reader': reader,
                'writer': writer,
                'host': host,
                'port': port,
                'ssl': use_ssl
            }

            self.stats[name].connection_time = connection_time
            self.state = ConnectionState.CONNECTED

            logger.info(f"Conectado a {name} ({host}:{port}) em {connection_time*1000:.2f}ms")
            self._emit('on_connect', name)

            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            logger.error(f"Erro ao conectar {name}: {e}")
            self._emit('on_error', name, str(e))
            return False

    async def disconnect(self, name: str) -> None:
        """Desconecta conexão específica"""
        if name in self.connections:
            try:
                writer = self.connections[name]['writer']
                writer.close()
                await writer.wait_closed()
                del self.connections[name]
                logger.info(f"Desconectado de {name}")
                self._emit('on_disconnect', name)
            except Exception as e:
                logger.error(f"Erro ao desconectar {name}: {e}")

    async def send(self, name: str, data: bytes) -> bool:
        """
        Envia dados com medição de latência
        """
        if name not in self.connections:
            logger.error(f"Conexão {name} não encontrada")
            return False

        try:
            start_time = time.perf_counter()

            writer = self.connections[name]['writer']
            writer.write(data)
            await writer.drain()

            latency = time.perf_counter() - start_time

            with self._lock:
                self.stats[name].bytes_sent += len(data)
                self.stats[name].messages_sent += 1
                self.stats[name].latencies.append(latency)

            self._emit('on_latency', name, latency)

            return True

        except Exception as e:
            logger.error(f"Erro ao enviar para {name}: {e}")
            self._emit('on_error', name, str(e))
            return False

    async def receive(self, name: str, size: int = 65536) -> Optional[bytes]:
        """
        Recebe dados da conexão
        """
        if name not in self.connections:
            return None

        try:
            reader = self.connections[name]['reader']
            data = await reader.read(size)

            if data:
                with self._lock:
                    self.stats[name].bytes_received += len(data)
                    self.stats[name].messages_received += 1

                self._emit('on_message', name, data)

            return data

        except Exception as e:
            logger.error(f"Erro ao receber de {name}: {e}")
            return None

    def get_stats(self, name: str) -> Optional[NetworkStats]:
        """Retorna estatísticas da conexão"""
        return self.stats.get(name)

    def get_all_stats(self) -> Dict[str, NetworkStats]:
        """Retorna estatísticas de todas as conexões"""
        return dict(self.stats)

    async def health_check(self, name: str) -> bool:
        """Verifica saúde da conexão"""
        if name not in self.connections:
            return False

        try:
            conn = self.connections[name]
            return not conn['writer'].is_closing()
        except Exception:
            return False

    async def reconnect(self, name: str, max_retries: int = 5,
                       delay: float = 1.0) -> bool:
        """
        Reconecta com backoff exponencial
        """
        if name not in self.connections:
            return False

        conn = self.connections[name]
        self.state = ConnectionState.RECONNECTING

        for attempt in range(max_retries):
            logger.info(f"Tentativa de reconexão {attempt + 1}/{max_retries} para {name}")

            await self.disconnect(name)
            await asyncio.sleep(delay * (2 ** attempt))

            if await self.connect(name, conn['host'], conn['port'], conn['ssl']):
                self.stats[name].reconnect_count += 1
                return True

        self.state = ConnectionState.ERROR
        return False

    async def close_all(self) -> None:
        """Fecha todas as conexões"""
        for name in list(self.connections.keys()):
            await self.disconnect(name)
        self.state = ConnectionState.DISCONNECTED
