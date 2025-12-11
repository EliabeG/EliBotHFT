"""
WebSocket Client - Cliente WebSocket otimizado para HFT
Suporte a FXOpen TickTrader Web API
"""

import asyncio
import json
import time
import hashlib
import hmac
import base64
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
import logging
from datetime import datetime, timezone

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WebSocketStats:
    """Estatísticas do WebSocket"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    ping_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    last_message_time: float = 0.0
    connection_time: float = 0.0
    reconnects: int = 0


class WebSocketClient:
    """
    Cliente WebSocket de alta performance para FXOpen TickTrader
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa cliente WebSocket

        Args:
            config: Configuração contendo:
                - web_api_url: URL base da API
                - token_id: ID do token Web API
                - token_key: Chave do token
                - token_secret: Secret do token
                - auth_type: Tipo de autenticação (HMAC)
        """
        self.config = config
        self._ws: Optional[Any] = None
        self._session: Optional[Any] = None
        self._running = False
        self._connected = False
        self._stats = WebSocketStats()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_message': [],
            'on_tick': [],
            'on_trade': [],
            'on_error': [],
            'on_heartbeat': []
        }

        # Fila de mensagens para processamento
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Credenciais
        self.token_id = config.get('token_id', '')
        self.token_key = config.get('token_key', '')
        self.token_secret = config.get('token_secret', '')
        self.base_url = config.get('web_api_url', 'wss://ttdemomarginal.fxopen.net:8443')

        # Request tracking
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Registra callback para eventos"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emite evento para callbacks"""
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                logger.error(f"Erro no callback {event}: {e}")

    def _generate_signature(self, timestamp: str, method: str,
                           path: str, body: str = '') -> str:
        """
        Gera assinatura HMAC-SHA256 para autenticação
        """
        message = f"{timestamp}{self.token_id}{self.token_key}{method}{path}{body}"
        signature = hmac.new(
            self.token_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def _get_auth_headers(self, method: str = 'GET',
                         path: str = '/') -> Dict[str, str]:
        """
        Gera headers de autenticação para requisições
        """
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path)

        return {
            'X-API-ID': self.token_id,
            'X-API-KEY': self.token_key,
            'X-API-TIMESTAMP': timestamp,
            'X-API-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }

    async def connect(self) -> bool:
        """
        Estabelece conexão WebSocket com autenticação
        """
        if not WEBSOCKETS_AVAILABLE and not AIOHTTP_AVAILABLE:
            logger.error("Nenhuma biblioteca WebSocket disponível. Instale websockets ou aiohttp.")
            return False

        try:
            start_time = time.perf_counter()

            ws_url = f"{self.base_url}/api/v2/ws"

            if AIOHTTP_AVAILABLE:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    ws_url,
                    headers=self._get_auth_headers('GET', '/api/v2/ws'),
                    heartbeat=30.0,
                    receive_timeout=60.0
                )
            elif WEBSOCKETS_AVAILABLE:
                self._ws = await websockets.connect(
                    ws_url,
                    extra_headers=self._get_auth_headers('GET', '/api/v2/ws'),
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5
                )

            self._connected = True
            self._stats.connection_time = time.perf_counter() - start_time

            logger.info(f"WebSocket conectado em {self._stats.connection_time*1000:.2f}ms")
            self._emit('on_connect')

            # Iniciar receiver loop
            asyncio.create_task(self._receiver_loop())

            return True

        except Exception as e:
            logger.error(f"Erro ao conectar WebSocket: {e}")
            self._emit('on_error', str(e))
            return False

    async def disconnect(self) -> None:
        """Desconecta WebSocket"""
        self._running = False
        self._connected = False

        try:
            if self._ws:
                if AIOHTTP_AVAILABLE and hasattr(self._ws, 'close'):
                    await self._ws.close()
                elif WEBSOCKETS_AVAILABLE:
                    await self._ws.close()

            if self._session:
                await self._session.close()

            logger.info("WebSocket desconectado")
            self._emit('on_disconnect')

        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")

    async def _receiver_loop(self) -> None:
        """Loop de recebimento de mensagens"""
        self._running = True

        while self._running and self._connected:
            try:
                if AIOHTTP_AVAILABLE and hasattr(self._ws, 'receive'):
                    msg = await self._ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {self._ws.exception()}")
                        break
                elif WEBSOCKETS_AVAILABLE:
                    data = await self._ws.recv()
                    await self._handle_message(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no receiver loop: {e}")
                if not self._connected:
                    break
                await asyncio.sleep(0.1)

    async def _handle_message(self, data: str) -> None:
        """Processa mensagem recebida"""
        try:
            self._stats.messages_received += 1
            self._stats.bytes_received += len(data)
            self._stats.last_message_time = time.time()

            message = json.loads(data)
            msg_type = message.get('type', '')

            # Emitir evento genérico
            self._emit('on_message', message)

            # Processar por tipo
            if msg_type == 'tick':
                self._emit('on_tick', message.get('data', {}))
            elif msg_type == 'trade':
                self._emit('on_trade', message.get('data', {}))
            elif msg_type == 'heartbeat':
                self._emit('on_heartbeat')
            elif 'id' in message:
                # Resposta a request
                req_id = message['id']
                if req_id in self._pending_requests:
                    self._pending_requests[req_id].set_result(message)
                    del self._pending_requests[req_id]

        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON: {e}")
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")

    async def send(self, message: Dict[str, Any]) -> bool:
        """
        Envia mensagem pelo WebSocket
        """
        if not self._connected or not self._ws:
            logger.error("WebSocket não conectado")
            return False

        try:
            data = json.dumps(message)
            start_time = time.perf_counter()

            if AIOHTTP_AVAILABLE and hasattr(self._ws, 'send_str'):
                await self._ws.send_str(data)
            elif WEBSOCKETS_AVAILABLE:
                await self._ws.send(data)

            latency = time.perf_counter() - start_time

            self._stats.messages_sent += 1
            self._stats.bytes_sent += len(data)
            self._stats.latencies.append(latency)

            return True

        except Exception as e:
            logger.error(f"Erro ao enviar mensagem: {e}")
            return False

    async def request(self, method: str, params: Dict[str, Any] = None,
                     timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Envia request e aguarda resposta
        """
        self._request_id += 1
        req_id = self._request_id

        message = {
            'id': req_id,
            'method': method,
            'params': params or {}
        }

        future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future

        if not await self.send(message):
            del self._pending_requests[req_id]
            return None

        try:
            result = await asyncio.wait_for(future, timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Timeout na request {req_id}")
            if req_id in self._pending_requests:
                del self._pending_requests[req_id]
            return None

    async def subscribe_ticks(self, symbols: List[str]) -> bool:
        """
        Inscreve-se para receber ticks de símbolos
        """
        result = await self.request('subscribe', {
            'type': 'tick',
            'symbols': symbols
        })
        return result is not None and result.get('success', False)

    async def subscribe_trades(self) -> bool:
        """
        Inscreve-se para receber atualizações de trades
        """
        result = await self.request('subscribe', {
            'type': 'trade'
        })
        return result is not None and result.get('success', False)

    @property
    def stats(self) -> WebSocketStats:
        """Retorna estatísticas do WebSocket"""
        return self._stats

    @property
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self._connected

    @property
    def avg_latency_ms(self) -> float:
        """Latência média em ms"""
        if not self._stats.latencies:
            return 0.0
        return sum(self._stats.latencies) / len(self._stats.latencies) * 1000
