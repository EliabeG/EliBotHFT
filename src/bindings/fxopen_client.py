"""
FXOpen Client - Cliente para FXOpen TickTrader Web API
Integração completa via WebSocket (Feed e Trade)

Baseado na documentação oficial:
- Feed: wss://marginalttdemowebapi.fxopen.net/feed (Market Data)
- Trade: wss://marginalttdemowebapi.fxopen.net/trade (Trading)

Referência: https://marginalttlivewebapi.fxopen.net/api/doc/WebSockets-Trade
"""

import asyncio
import aiohttp
import hashlib
import hmac
import base64
import time
import json
import uuid
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

    # WebSocket URLs (baseado na documentação oficial)
    # Para conta DEMO: ttdemomarginal.fxopen.net -> marginalttdemowebapi.fxopen.net
    # Para conta LIVE: ttlivemarginal.fxopen.net -> marginalttlivewebapi.fxopen.net
    webapi_host: str = 'marginalttdemowebapi.fxopen.net'  # Demo account

    @property
    def feed_url(self) -> str:
        return f'wss://{self.webapi_host}/feed'

    @property
    def trade_url(self) -> str:
        return f'wss://{self.webapi_host}/trade'


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
    Cliente para FXOpen TickTrader Web API via WebSocket

    Arquitetura:
    - Feed WebSocket (/feed): Market data, ticks, quotes
    - Trade WebSocket (/trade): Account info, trading, positions

    Autenticação:
    - HMAC-SHA256 via mensagem Login no WebSocket
    - Signature = Base64(HMAC-SHA256(timestamp + webApiId + webApiKey, secret))
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
        self._feed_connected = False
        self._trade_connected = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._feed_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._trade_ws: Optional[aiohttp.ClientWebSocketResponse] = None

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

        # Request tracking
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # Symbol specifications (contract sizes)
        self._symbol_specs: Dict[str, dict] = {}

        logger.info(f"FXOpenClient inicializado para {self.config.server}")
        logger.info(f"Feed URL: {self.config.feed_url}")
        logger.info(f"Trade URL: {self.config.trade_url}")

    def _generate_signature(self, timestamp: int) -> str:
        """
        Gera assinatura HMAC-SHA256 conforme documentação FXOpen

        Signature = Base64(HMAC-SHA256(timestamp + webApiId + webApiKey, secret))
        """
        message = f"{timestamp}{self.config.token_id}{self.config.token_key}"

        signature = hmac.new(
            self.config.token_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode('utf-8')

    def _create_login_request(self) -> dict:
        """Cria mensagem de login conforme documentação"""
        timestamp = int(time.time() * 1000)  # Milliseconds
        signature = self._generate_signature(timestamp)

        return {
            "Id": str(uuid.uuid4()),
            "Request": "Login",
            "Params": {
                "AuthType": self.config.auth_type,
                "WebApiId": self.config.token_id,
                "WebApiKey": self.config.token_key,
                "Timestamp": timestamp,
                "Signature": signature
            }
        }

    async def connect(self) -> bool:
        """Conecta aos WebSockets (Feed e Trade)"""
        try:
            # Criar sessão HTTP
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=True)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Conectar ao Trade WebSocket primeiro (para account info)
            logger.info(f"Conectando ao Trade WebSocket: {self.config.trade_url}")
            trade_ok = await self._connect_trade_ws()

            if trade_ok:
                logger.info("Trade WebSocket conectado!")
                self._trade_connected = True

                # Conectar ao Feed WebSocket
                logger.info(f"Conectando ao Feed WebSocket: {self.config.feed_url}")
                feed_ok = await self._connect_feed_ws()

                if feed_ok:
                    logger.info("Feed WebSocket conectado!")
                    self._feed_connected = True
                    self._connected = True
                    return True
                else:
                    logger.warning("Feed WebSocket falhou, mas Trade está ok")
                    self._connected = True
                    return True

            logger.error("Falha ao conectar Trade WebSocket")
            return False

        except Exception as e:
            logger.error(f"Erro ao conectar: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _connect_trade_ws(self) -> bool:
        """Conecta ao WebSocket de Trading (/trade)"""
        try:
            self._trade_ws = await self._session.ws_connect(
                self.config.trade_url,
                heartbeat=30.0,
                ssl=True
            )

            # Enviar login
            login_request = self._create_login_request()
            logger.debug(f"Enviando login request: {json.dumps(login_request, indent=2)}")
            await self._trade_ws.send_json(login_request)

            # Aguardar resposta
            response = await asyncio.wait_for(
                self._trade_ws.receive(),
                timeout=10.0
            )

            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                logger.debug(f"Login response: {json.dumps(data, indent=2)}")

                if data.get('Response') == 'Login' and data.get('Result') is not None:
                    result = data.get('Result', {})
                    if 'Error' in result:
                        logger.error(f"Login falhou: {result['Error']}")
                        return False

                    logger.info("Login Trade bem-sucedido!")

                    # Iniciar receiver
                    asyncio.create_task(self._trade_ws_receiver())

                    # Buscar account info
                    await self._request_account_info()

                    return True
                else:
                    logger.error(f"Resposta inesperada: {data}")
                    return False

            logger.error(f"Tipo de mensagem inesperado: {response.type}")
            return False

        except asyncio.TimeoutError:
            logger.error("Timeout aguardando resposta de login (Trade)")
            return False
        except Exception as e:
            logger.error(f"Erro ao conectar Trade WS: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _connect_feed_ws(self) -> bool:
        """Conecta ao WebSocket de Feed (/feed)"""
        try:
            self._feed_ws = await self._session.ws_connect(
                self.config.feed_url,
                heartbeat=30.0,
                ssl=True
            )

            # Enviar login
            login_request = self._create_login_request()
            await self._feed_ws.send_json(login_request)

            # Aguardar resposta
            response = await asyncio.wait_for(
                self._feed_ws.receive(),
                timeout=10.0
            )

            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                logger.debug(f"Feed Login response: {json.dumps(data, indent=2)}")

                if data.get('Response') == 'Login':
                    result = data.get('Result', {})
                    if 'Error' in result:
                        logger.error(f"Feed Login falhou: {result['Error']}")
                        return False

                    logger.info("Login Feed bem-sucedido!")

                    # Buscar especificações dos símbolos ANTES de iniciar o receiver
                    await self._load_symbol_specs()

                    # Só depois iniciar o receiver para evitar "Concurrent call to receive()"
                    self._feed_connected = True
                    asyncio.create_task(self._feed_ws_receiver())

                    return True

            return False

        except asyncio.TimeoutError:
            logger.error("Timeout aguardando resposta de login (Feed)")
            return False
        except Exception as e:
            logger.error(f"Erro ao conectar Feed WS: {e}")
            return False

    async def disconnect(self) -> None:
        """Desconecta dos WebSockets"""
        self._connected = False
        self._feed_connected = False
        self._trade_connected = False

        if self._feed_ws and not self._feed_ws.closed:
            await self._feed_ws.close()
            self._feed_ws = None

        if self._trade_ws and not self._trade_ws.closed:
            await self._trade_ws.close()
            self._trade_ws = None

        if self._session:
            await self._session.close()
            # Aguardar limpeza das conexões SSL
            await asyncio.sleep(0.25)
            self._session = None

        logger.info("Desconectado da FXOpen")

    async def _trade_ws_receiver(self) -> None:
        """Loop de recepção de mensagens do Trade WebSocket"""
        while self._trade_ws and not self._trade_ws.closed:
            try:
                msg = await self._trade_ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_trade_message(data)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("Trade WebSocket fechado")
                    self._trade_connected = False
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Trade WebSocket error: {self._trade_ws.exception()}")
                    break

            except Exception as e:
                logger.error(f"Erro no Trade WS receiver: {e}")
                await asyncio.sleep(1)

    async def _feed_ws_receiver(self) -> None:
        """Loop de recepção de mensagens do Feed WebSocket"""
        logger.info("Feed WS receiver iniciado")
        tick_count = 0
        while self._feed_ws and not self._feed_ws.closed:
            try:
                msg = await self._feed_ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # Log periódico de ticks
                    if data.get('Response') == 'FeedTick':
                        tick_count += 1
                        if tick_count % 100 == 1:  # Log a cada 100 ticks
                            symbol = data.get('Result', {}).get('Symbol', '?')
                            logger.info(f"Feed recebendo ticks ({tick_count} total) - último: {symbol}")
                    await self._handle_feed_message(data)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("Feed WebSocket fechado")
                    self._feed_connected = False
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Feed WebSocket error: {self._feed_ws.exception()}")
                    break

            except Exception as e:
                logger.error(f"Erro no Feed WS receiver: {e}")
                await asyncio.sleep(1)
        logger.warning("Feed WS receiver encerrado")

    async def _handle_trade_message(self, data: dict) -> None:
        """Processa mensagem do Trade WebSocket"""
        response_type = data.get('Response', '')
        request_id = data.get('Id', '')

        # Resolver pending request
        if request_id in self._pending_requests:
            future = self._pending_requests.pop(request_id)
            if not future.done():
                future.set_result(data)
            return

        # Notificações
        if 'Notify' in data:
            notify_type = data.get('Notify', '')

            if notify_type == 'PositionUpdate':
                for cb in self._on_trade:
                    try:
                        cb(data)
                    except Exception as e:
                        logger.error(f"Erro no callback de trade: {e}")

            elif notify_type == 'AccountUpdate':
                result = data.get('Result', {})
                if result:
                    self._update_account_from_result(result)

    async def _handle_feed_message(self, data: dict) -> None:
        """Processa mensagem do Feed WebSocket"""
        response_type = data.get('Response', '')

        if response_type == 'FeedTick' or response_type == 'Tick':
            # Tick update (subscription ou polling)
            result = data.get('Result', {})
            symbol = result.get('Symbol', '')

            if symbol:
                bid_best = result.get('BestBid', {})
                ask_best = result.get('BestAsk', {})

                tick = Tick(
                    symbol=symbol,
                    bid=float(bid_best.get('Price', 0)) if bid_best else 0,
                    ask=float(ask_best.get('Price', 0)) if ask_best else 0,
                    timestamp=datetime.now(),
                    bid_volume=float(bid_best.get('Volume', 0)) if bid_best else 0,
                    ask_volume=float(ask_best.get('Volume', 0)) if ask_best else 0
                )

                self._quotes[symbol] = tick

                # Log tick recebido (debug)
                logger.debug(f"Tick {symbol}: bid={tick.bid:.5f}, ask={tick.ask:.5f}")

                for cb in self._on_tick:
                    try:
                        cb(tick)
                    except Exception as e:
                        logger.error(f"Erro no callback de tick: {e}")
        elif response_type == 'FeedSubscribe':
            # Confirmação de inscrição
            result = data.get('Result', {})
            logger.info(f"FeedSubscribe confirmado: {result}")
        else:
            # Log outras mensagens do feed
            logger.info(f"Feed message não tratada: {response_type} - {data}")

    def _update_account_from_result(self, result: dict) -> None:
        """Atualiza account info do resultado"""
        try:
            balance = float(result.get('Balance', 0))
            equity = float(result.get('Equity', 0))
            margin = float(result.get('Margin', 0))

            # FreeMargin pode não ser fornecido diretamente
            # FreeMargin = Equity - Margin
            free_margin = float(result.get('FreeMargin', 0))
            if free_margin == 0 and equity > 0:
                free_margin = equity - margin

            # BalanceCurrency pode estar em campo diferente
            currency = result.get('Currency', result.get('BalanceCurrency', 'USD'))

            logger.info(f"Account info atualizado: Balance={balance}, Equity={equity}, Margin={margin}, FreeMargin={free_margin}")

            self._account_info = AccountInfo(
                account_id=str(result.get('Id', '')),
                balance=balance,
                equity=equity,
                margin=margin,
                free_margin=free_margin,
                margin_level=float(result.get('MarginLevel', 0)),
                currency=currency,
                leverage=int(result.get('Leverage', 500))
            )
        except Exception as e:
            logger.error(f"Erro ao parsear account info: {e}")
            import traceback
            traceback.print_exc()

    async def _send_trade_request(self, request: dict, timeout: float = 10.0) -> Optional[dict]:
        """Envia request ao Trade WebSocket e aguarda resposta"""
        if not self._trade_ws or self._trade_ws.closed:
            logger.error("Trade WebSocket não conectado")
            return None

        request_id = request.get('Id', str(uuid.uuid4()))
        request['Id'] = request_id

        # Criar future para aguardar resposta
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await self._trade_ws.send_json(request)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            logger.error(f"Timeout aguardando resposta para {request.get('Request', '')}")
            return None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            logger.error(f"Erro enviando request: {e}")
            return None

    async def _load_symbol_specs(self) -> None:
        """Carrega especificações dos símbolos (contract size, etc)"""
        if not self._feed_ws or self._feed_ws.closed:
            logger.warning("Feed WS não conectado para carregar specs")
            return

        request = {
            "Id": str(uuid.uuid4()),
            "Request": "Symbols"
        }

        try:
            await self._feed_ws.send_json(request)

            # Aguardar resposta
            response = await asyncio.wait_for(
                self._feed_ws.receive(),
                timeout=10.0
            )

            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                logger.debug(f"Symbols response: {json.dumps(data, indent=2)}")
                if data.get('Response') == 'Symbols' and 'Result' in data:
                    # Result.Symbols contém a lista de símbolos
                    symbols_list = data['Result'].get('Symbols', [])
                    for symbol_data in symbols_list:
                        symbol = symbol_data.get('Symbol', '')
                        if symbol:
                            self._symbol_specs[symbol] = {
                                'contract_size': float(symbol_data.get('ContractSize', 100000)),
                                'min_amount': float(symbol_data.get('MinTradeAmount', 1000)),
                                'max_amount': float(symbol_data.get('MaxTradeAmount', 10000000)),
                                'step': float(symbol_data.get('TradeAmountStep', 1000)),
                                'precision': int(symbol_data.get('Precision', 5))
                            }
                    logger.info(f"Carregadas specs de {len(self._symbol_specs)} símbolos")

        except Exception as e:
            logger.error(f"Erro ao carregar specs: {e}")
            # Usar defaults para pares de moedas
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']:
                self._symbol_specs[symbol] = {
                    'contract_size': 100000,
                    'min_amount': 0.01,
                    'max_amount': 1000,
                    'step': 0.01,
                    'precision': 5 if symbol != 'USDJPY' else 3
                }

    def _lots_to_amount(self, symbol: str, lots: float) -> float:
        """Converte lotes para unidades (Amount)"""
        # Obter contract size do símbolo
        specs = self._symbol_specs.get(symbol, {})
        contract_size = specs.get('contract_size', 100000)

        amount = lots * contract_size
        logger.info(f"Convertendo {lots} lotes -> {amount} unidades ({symbol}, contract_size={contract_size})")
        return amount

    async def _request_account_info(self) -> None:
        """Requisita informações da conta"""
        request = {
            "Id": str(uuid.uuid4()),
            "Request": "Account"
        }

        logger.debug(f"Requesting account info: {json.dumps(request)}")
        response = await self._send_trade_request(request)
        logger.debug(f"Account response: {json.dumps(response, indent=2) if response else 'None'}")
        if response and 'Result' in response:
            self._update_account_from_result(response['Result'])

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Obtém informações da conta"""
        await self._request_account_info()
        return self._account_info

    async def get_tick(self, symbol: str) -> Optional[Tick]:
        """Obtém tick atual de um símbolo"""
        if symbol in self._quotes:
            return self._quotes[symbol]

        # Requisitar via Feed WS
        if self._feed_ws and not self._feed_ws.closed:
            request = {
                "Id": str(uuid.uuid4()),
                "Request": "GetTick",
                "Params": {
                    "Symbol": symbol
                }
            }
            await self._feed_ws.send_json(request)

            # Aguardar um pouco e retornar do cache
            await asyncio.sleep(0.5)
            return self._quotes.get(symbol)

        return None

    async def get_positions(self) -> List[TradePosition]:
        """Obtém posições/trades abertas"""
        # Para conta Gross usamos "Trades", para Net usamos "Positions"
        request = {
            "Id": str(uuid.uuid4()),
            "Request": "Trades"
        }

        logger.debug(f"Requesting trades: {json.dumps(request)}")
        response = await self._send_trade_request(request)
        logger.debug(f"Trades response: {json.dumps(response, indent=2) if response else 'None'}")

        if not response or 'Result' not in response:
            return []

        # Result.Trades contém a lista de trades
        trades_list = response.get('Result', {}).get('Trades', [])

        positions = []
        for pos in trades_list:
            try:
                # Filtrar apenas trades abertos (Status: Calculated = posição aberta)
                status = pos.get('Status', '')
                if status not in ('Calculated', 'Opened', 'PartiallyFilled', 'Filled'):
                    continue

                # Filtrar apenas posições (Type: Position)
                trade_type = pos.get('Type', '')
                if trade_type != 'Position':
                    continue

                position = TradePosition(
                    position_id=str(pos.get('Id', '')),
                    symbol=pos.get('Symbol', ''),
                    side=pos.get('Side', ''),
                    volume=float(pos.get('Amount', pos.get('Volume', 0))),
                    open_price=float(pos.get('Price', 0)),
                    open_time=datetime.now(),  # Parse do timestamp real se disponível
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

    async def subscribe_ticks(self, symbols: List[str]) -> bool:
        """Inscreve para receber ticks via Feed WebSocket"""
        if not self._feed_ws or self._feed_ws.closed:
            logger.error("Feed WebSocket não conectado")
            return False

        for symbol in symbols:
            request = {
                "Id": str(uuid.uuid4()),
                "Request": "FeedSubscribe",
                "Params": {
                    "Subscribe": [{
                        "Symbol": symbol,
                        "BookDepth": 1  # Top of book
                    }]
                }
            }
            await self._feed_ws.send_json(request)
            self._subscribed_symbols.add(symbol)
            logger.info(f"Inscrito para ticks de {symbol}")

        return True

    async def unsubscribe_ticks(self, symbols: List[str]) -> bool:
        """Remove inscrição de ticks"""
        if not self._feed_ws or self._feed_ws.closed:
            return False

        for symbol in symbols:
            request = {
                "Id": str(uuid.uuid4()),
                "Request": "FeedUnsubscribe",
                "Params": {
                    "Unsubscribe": [symbol]
                }
            }
            await self._feed_ws.send_json(request)
            self._subscribed_symbols.discard(symbol)

        return True

    async def open_position(self, symbol: str, side: OrderSide, volume: float,
                           order_type: OrderType = OrderType.MARKET,
                           price: float = None, stop_loss: float = None,
                           take_profit: float = None,
                           comment: str = 'EliBotHFT') -> Optional[str]:
        """Abre nova posição (volume em lotes, convertido para unidades)"""
        # Converter lotes para unidades (Amount)
        amount = self._lots_to_amount(symbol, volume)

        # Obter precisão do símbolo
        specs = self._symbol_specs.get(symbol, {})
        precision = specs.get('precision', 5)

        params = {
            'Symbol': symbol,
            'Side': side.value,
            'Type': order_type.value,
            'Amount': amount,  # Unidades, não lotes
            'Comment': comment
        }

        if price:
            params['Price'] = round(price, precision)
        if stop_loss:
            params['StopLoss'] = round(stop_loss, precision)
        if take_profit:
            params['TakeProfit'] = round(take_profit, precision)

        request = {
            "Id": str(uuid.uuid4()),
            "Request": "TradeCreate",
            "Params": params
        }

        logger.info(f"Enviando ordem: {json.dumps(request, indent=2)}")

        response = await self._send_trade_request(request)

        logger.info(f"Resposta da ordem: {json.dumps(response, indent=2) if response else 'None'}")

        if not response:
            logger.error("Sem resposta do servidor")
            return None

        # Verificar erro no nível superior (pode vir com Response="Error" ou Response="TradeCreate" + Error)
        if 'Error' in response:
            logger.error(f"Erro ao abrir posição: {response.get('Error', 'Unknown error')}")
            return None

        # Verificar se há Result
        if 'Result' in response:
            result = response['Result']

            # Verificar erro dentro do Result
            if isinstance(result, dict) and 'Error' in result:
                logger.error(f"Erro ao abrir posição: {result['Error']}")
                return None

            # Sucesso - Result pode conter Trade com Id
            trade_data = result.get('Trade', result)  # Trade está em Result.Trade
            if isinstance(trade_data, dict) and 'Id' in trade_data:
                position_id = str(trade_data['Id'])
                logger.info(f"Posição aberta: {position_id} {side.value} {volume} {symbol}")
                return position_id

        logger.warning("Resposta não contém Result ou Trade.Id")
        return None

    async def close_position(self, position_id: str, volume: float = None) -> bool:
        """Fecha posição"""
        params = {
            'Type': 'Close',
            'Id': int(position_id)
        }
        if volume:
            params['Amount'] = volume

        request = {
            "Id": str(uuid.uuid4()),
            "Request": "TradeDelete",  # Correto: TradeDelete, não ClosePosition
            "Params": params
        }

        logger.info(f"Fechando posição: {json.dumps(request, indent=2)}")

        response = await self._send_trade_request(request)

        logger.info(f"Resposta de fechamento: {json.dumps(response, indent=2) if response else 'None'}")

        if response and 'Result' in response:
            logger.info(f"Posição fechada: {position_id}")
            self._positions.pop(position_id, None)
            return True

        return False

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
