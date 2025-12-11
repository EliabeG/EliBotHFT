"""
FXOpen Client - Cliente para FXOpen TickTrader Web API
Integração completa com a conta de trading
"""

import asyncio
import aiohttp
import hashlib
import hmac
import base64
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FXOpenConfig:
    """Configuração de conexão FXOpen"""
    # Credenciais
    login: str = '28503781'
    password: str = 'rngGNGMW'
    server: str = 'ttdemomarginal.fxopen.net'

    # Web API
    token_id: str = '0473113a-f96d-4576-bd1b-507e71ec3d4f'
    token_key: str = 'EGqeZPpJQSW2BjCb'
    token_secret: str = 'YdafQEND2Fnrc5JGryX6ZPCJ5pf9rmyHnAk6wTDjWGddcRjWtxw369YhKzkBzPkM'
    auth_type: str = 'HMAC'

    # Conta
    account_type: str = 'Gross'
    platform: str = 'TickTrader'
    leverage: int = 500
    currency: str = 'USD'

    # URLs
    rest_url: str = field(default='')
    ws_url: str = field(default='')

    def __post_init__(self):
        if not self.rest_url:
            self.rest_url = f'https://{self.server}:8443/api/v2'
        if not self.ws_url:
            self.ws_url = f'wss://{self.server}:8443/api/v2/ws'


class OrderSide(Enum):
    BUY = 'Buy'
    SELL = 'Sell'


class OrderType(Enum):
    MARKET = 'Market'
    LIMIT = 'Limit'
    STOP = 'Stop'
    STOP_LIMIT = 'StopLimit'


class TimeInForce(Enum):
    GTC = 'GoodTillCancel'
    IOC = 'ImmediateOrCancel'
    FOK = 'FillOrKill'


@dataclass
class Tick:
    """Tick de mercado"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class AccountInfo:
    """Informações da conta"""
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int


@dataclass
class TradePosition:
    """Posição de trade"""
    position_id: str
    symbol: str
    side: str
    volume: float
    open_price: float
    open_time: datetime
    profit: float
    swap: float
    commission: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class FXOpenClient:
    """
    Cliente para FXOpen TickTrader Web API

    Funcionalidades:
    - Autenticação HMAC
    - Market Data (ticks, quotes)
    - Trading (ordens, posições)
    - Account info
    """

    def __init__(self, config: FXOpenConfig = None):
        """
        Inicializa cliente

        Args:
            config: Configuração de conexão
        """
        self.config = config or FXOpenConfig()

        # Estado
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

        # Callbacks
        self._on_tick: List[Callable[[Tick], None]] = []
        self._on_trade: List[Callable[[dict], None]] = []
        self._on_account: List[Callable[[AccountInfo], None]] = []

        # Cache
        self._account_info: Optional[AccountInfo] = None
        self._positions: Dict[str, TradePosition] = {}
        self._quotes: Dict[str, Tick] = {}

        # Subscriptions
        self._subscribed_symbols: set = set()

        logger.info(f"FXOpenClient inicializado para {self.config.server}")

    def _generate_signature(self, timestamp: str, method: str,
                           path: str, body: str = '') -> str:
        """Gera assinatura HMAC-SHA256"""
        message = f"{timestamp}{self.config.token_id}{self.config.token_key}{method}{path}{body}"

        signature = hmac.new(
            self.config.token_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode('utf-8')

    def _get_headers(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Gera headers de autenticação"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path, body)

        return {
            'Authorization': f'HMAC {self.config.token_id}:{self.config.token_key}:{timestamp}:{signature}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    async def connect(self) -> bool:
        """Conecta à API"""
        try:
            # Criar sessão HTTP
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Verificar conexão com chamada de account
            account = await self.get_account_info()
            if account:
                self._connected = True
                self._account_info = account
                logger.info(f"Conectado! Conta: {account.account_id}, "
                           f"Balance: {account.balance:.2f} {account.currency}")
                return True

            return False

        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            return False

    async def disconnect(self) -> None:
        """Desconecta da API"""
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Desconectado da FXOpen")

    async def _request(self, method: str, endpoint: str,
                      data: dict = None) -> Optional[dict]:
        """Faz requisição HTTP autenticada"""
        if not self._session:
            logger.error("Sessão não inicializada")
            return None

        url = f"{self.config.rest_url}{endpoint}"
        body = json.dumps(data) if data else ''

        headers = self._get_headers(method.upper(), endpoint, body)

        try:
            async with self._session.request(
                method, url, headers=headers, data=body if data else None
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    logger.error(f"Erro {response.status}: {error}")
                    return None

        except Exception as e:
            logger.error(f"Erro na requisição: {e}")
            return None

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Obtém informações da conta"""
        result = await self._request('GET', '/account')
        if not result:
            return None

        try:
            return AccountInfo(
                account_id=str(result.get('Id', '')),
                balance=float(result.get('Balance', 0)),
                equity=float(result.get('Equity', 0)),
                margin=float(result.get('Margin', 0)),
                free_margin=float(result.get('FreeMargin', 0)),
                margin_level=float(result.get('MarginLevel', 0)),
                currency=result.get('Currency', 'USD'),
                leverage=int(result.get('Leverage', 500))
            )
        except Exception as e:
            logger.error(f"Erro ao parsear account info: {e}")
            return None

    async def get_symbols(self) -> List[dict]:
        """Obtém lista de símbolos disponíveis"""
        result = await self._request('GET', '/symbol')
        return result if result else []

    async def get_tick(self, symbol: str) -> Optional[Tick]:
        """Obtém tick atual de um símbolo"""
        result = await self._request('GET', f'/tick/{symbol}')
        if not result:
            return None

        try:
            return Tick(
                symbol=symbol,
                bid=float(result.get('BestBid', {}).get('Price', 0)),
                ask=float(result.get('BestAsk', {}).get('Price', 0)),
                timestamp=datetime.now(),
                bid_volume=float(result.get('BestBid', {}).get('Volume', 0)),
                ask_volume=float(result.get('BestAsk', {}).get('Volume', 0))
            )
        except Exception as e:
            logger.error(f"Erro ao parsear tick: {e}")
            return None

    async def get_ticks(self, symbols: List[str]) -> Dict[str, Tick]:
        """Obtém ticks de múltiplos símbolos"""
        ticks = {}
        for symbol in symbols:
            tick = await self.get_tick(symbol)
            if tick:
                ticks[symbol] = tick
                self._quotes[symbol] = tick
        return ticks

    async def get_positions(self) -> List[TradePosition]:
        """Obtém posições abertas"""
        result = await self._request('GET', '/position')
        if not result:
            return []

        positions = []
        for pos in result:
            try:
                position = TradePosition(
                    position_id=str(pos.get('Id', '')),
                    symbol=pos.get('Symbol', ''),
                    side=pos.get('Side', ''),
                    volume=float(pos.get('Volume', 0)),
                    open_price=float(pos.get('Price', 0)),
                    open_time=datetime.fromisoformat(pos.get('Created', '').replace('Z', '+00:00')),
                    profit=float(pos.get('Profit', 0)),
                    swap=float(pos.get('Swap', 0)),
                    commission=float(pos.get('Commission', 0)),
                    stop_loss=float(pos['StopLoss']) if pos.get('StopLoss') else None,
                    take_profit=float(pos['TakeProfit']) if pos.get('TakeProfit') else None
                )
                positions.append(position)
                self._positions[position.position_id] = position
            except Exception as e:
                logger.error(f"Erro ao parsear position: {e}")

        return positions

    async def open_position(self, symbol: str, side: OrderSide, volume: float,
                           order_type: OrderType = OrderType.MARKET,
                           price: float = None, stop_loss: float = None,
                           take_profit: float = None,
                           comment: str = 'EliBotHFT') -> Optional[str]:
        """
        Abre nova posição

        Args:
            symbol: Símbolo
            side: Lado (BUY/SELL)
            volume: Volume em lots
            order_type: Tipo de ordem
            price: Preço (para limit/stop)
            stop_loss: Stop loss
            take_profit: Take profit
            comment: Comentário

        Returns:
            ID da posição ou None
        """
        data = {
            'Symbol': symbol,
            'Side': side.value,
            'Type': order_type.value,
            'Volume': volume,
            'Comment': comment
        }

        if price:
            data['Price'] = price
        if stop_loss:
            data['StopLoss'] = stop_loss
        if take_profit:
            data['TakeProfit'] = take_profit

        result = await self._request('POST', '/trade', data)

        if result and 'Id' in result:
            position_id = str(result['Id'])
            logger.info(f"Posição aberta: {position_id} {side.value} {volume} {symbol}")
            return position_id

        return None

    async def close_position(self, position_id: str, volume: float = None) -> bool:
        """
        Fecha posição

        Args:
            position_id: ID da posição
            volume: Volume a fechar (None = tudo)

        Returns:
            True se fechado com sucesso
        """
        data = {'PositionId': position_id}
        if volume:
            data['Volume'] = volume

        result = await self._request('DELETE', f'/position/{position_id}', data)

        if result:
            logger.info(f"Posição fechada: {position_id}")
            self._positions.pop(position_id, None)
            return True

        return False

    async def modify_position(self, position_id: str,
                             stop_loss: float = None,
                             take_profit: float = None) -> bool:
        """
        Modifica posição (SL/TP)

        Args:
            position_id: ID da posição
            stop_loss: Novo stop loss
            take_profit: Novo take profit

        Returns:
            True se modificado com sucesso
        """
        data = {'Id': position_id}
        if stop_loss is not None:
            data['StopLoss'] = stop_loss
        if take_profit is not None:
            data['TakeProfit'] = take_profit

        result = await self._request('PUT', f'/position/{position_id}', data)
        return result is not None

    async def place_order(self, symbol: str, side: OrderSide, volume: float,
                         order_type: OrderType, price: float,
                         stop_loss: float = None, take_profit: float = None,
                         time_in_force: TimeInForce = TimeInForce.GTC,
                         expiration: datetime = None) -> Optional[str]:
        """
        Coloca ordem pendente

        Args:
            symbol: Símbolo
            side: Lado
            volume: Volume
            order_type: Tipo
            price: Preço
            stop_loss: Stop loss
            take_profit: Take profit
            time_in_force: Validade
            expiration: Data de expiração

        Returns:
            ID da ordem ou None
        """
        data = {
            'Symbol': symbol,
            'Side': side.value,
            'Type': order_type.value,
            'Volume': volume,
            'Price': price,
            'TimeInForce': time_in_force.value
        }

        if stop_loss:
            data['StopLoss'] = stop_loss
        if take_profit:
            data['TakeProfit'] = take_profit
        if expiration:
            data['Expiration'] = expiration.isoformat()

        result = await self._request('POST', '/order', data)

        if result and 'Id' in result:
            order_id = str(result['Id'])
            logger.info(f"Ordem colocada: {order_id} {order_type.value} {side.value} {volume} {symbol} @ {price}")
            return order_id

        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancela ordem pendente"""
        result = await self._request('DELETE', f'/order/{order_id}')
        if result:
            logger.info(f"Ordem cancelada: {order_id}")
            return True
        return False

    async def get_orders(self) -> List[dict]:
        """Obtém ordens pendentes"""
        result = await self._request('GET', '/order')
        return result if result else []

    async def get_history(self, from_time: datetime = None,
                         to_time: datetime = None,
                         limit: int = 100) -> List[dict]:
        """Obtém histórico de trades"""
        params = {'limit': limit}
        if from_time:
            params['from'] = from_time.isoformat()
        if to_time:
            params['to'] = to_time.isoformat()

        # Construir query string
        query = '&'.join(f"{k}={v}" for k, v in params.items())
        endpoint = f'/tradehistory?{query}'

        result = await self._request('GET', endpoint)
        return result if result else []

    # WebSocket Methods

    async def connect_websocket(self) -> bool:
        """Conecta ao WebSocket para streaming"""
        if not self._session:
            await self.connect()

        try:
            # Headers de autenticação para WS
            headers = self._get_headers('GET', '/ws')

            self._ws = await self._session.ws_connect(
                self.config.ws_url,
                headers=headers,
                heartbeat=30.0
            )

            logger.info("WebSocket conectado")

            # Iniciar receiver
            asyncio.create_task(self._ws_receiver())

            return True

        except Exception as e:
            logger.error(f"Erro ao conectar WebSocket: {e}")
            return False

    async def _ws_receiver(self) -> None:
        """Loop de recepção de mensagens WebSocket"""
        while self._ws and not self._ws.closed:
            try:
                msg = await self._ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_message(data)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket fechado")
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

            except Exception as e:
                logger.error(f"Erro no WS receiver: {e}")
                await asyncio.sleep(1)

    async def _handle_ws_message(self, data: dict) -> None:
        """Processa mensagem do WebSocket"""
        msg_type = data.get('Type', '')

        if msg_type == 'Tick':
            tick = Tick(
                symbol=data.get('Symbol', ''),
                bid=float(data.get('Bid', 0)),
                ask=float(data.get('Ask', 0)),
                timestamp=datetime.now()
            )
            self._quotes[tick.symbol] = tick

            for cb in self._on_tick:
                try:
                    cb(tick)
                except Exception as e:
                    logger.error(f"Erro no callback de tick: {e}")

        elif msg_type in ('TradeExecuted', 'PositionOpened', 'PositionClosed'):
            for cb in self._on_trade:
                try:
                    cb(data)
                except Exception as e:
                    logger.error(f"Erro no callback de trade: {e}")

        elif msg_type == 'AccountInfo':
            # Atualizar account info
            pass

    async def subscribe_ticks(self, symbols: List[str]) -> bool:
        """Inscreve para receber ticks"""
        if not self._ws:
            await self.connect_websocket()

        for symbol in symbols:
            msg = {
                'Type': 'SubscribeTick',
                'Symbol': symbol
            }
            await self._ws.send_json(msg)
            self._subscribed_symbols.add(symbol)
            logger.info(f"Inscrito para ticks de {symbol}")

        return True

    async def unsubscribe_ticks(self, symbols: List[str]) -> bool:
        """Remove inscrição de ticks"""
        if not self._ws:
            return False

        for symbol in symbols:
            msg = {
                'Type': 'UnsubscribeTick',
                'Symbol': symbol
            }
            await self._ws.send_json(msg)
            self._subscribed_symbols.discard(symbol)

        return True

    def register_tick_callback(self, callback: Callable[[Tick], None]) -> None:
        """Registra callback para ticks"""
        self._on_tick.append(callback)

    def register_trade_callback(self, callback: Callable[[dict], None]) -> None:
        """Registra callback para trades"""
        self._on_trade.append(callback)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def account(self) -> Optional[AccountInfo]:
        return self._account_info

    @property
    def quotes(self) -> Dict[str, Tick]:
        return self._quotes

    @property
    def positions(self) -> Dict[str, TradePosition]:
        return self._positions
