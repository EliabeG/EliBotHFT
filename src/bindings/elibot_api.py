"""
EliBot API - API de alto nível para controle do robô HFT
"""

import asyncio
import os
import sys
import yaml
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import threading

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.network import WebSocketClient
from core.fix_engine import FIXEngine, SessionConfig
from core.logger import AsyncLogger, LatencyLogger
from trading.book import OrderBook, BookSide
from trading.strategy import StrategyEngine, Signal, SignalType
from trading.oms import OrderManagementSystem, Order, OrderStatus, Side, OrderType
from trading.risk import RiskManager, RiskLimits
from .fxopen_client import FXOpenClient, FXOpenConfig, OrderSide

# ML System imports
from ml import MLManager, MLConfig, ErrorType

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuração do bot"""
    config_dir: str = 'config'
    data_dir: str = 'data'
    log_dir: str = 'data/logs'

    # Trading
    symbols: List[str] = None
    enabled: bool = True
    mode: str = 'paper'  # paper, live

    # FXOpen
    fxopen: FXOpenConfig = None

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['EURUSD']
        if self.fxopen is None:
            self.fxopen = FXOpenConfig()


class BotState:
    """Estados do bot"""
    STOPPED = 'stopped'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    ERROR = 'error'


class EliBotAPI:
    """
    API Principal do EliBotHFT

    Interface de alto nível para controlar todas as funcionalidades do robô.
    """

    def __init__(self, config: BotConfig = None, config_path: str = None):
        """
        Inicializa EliBotAPI

        Args:
            config: Configuração do bot
            config_path: Caminho para arquivo de configuração
        """
        # Configuração
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = config or BotConfig()

        # Estado
        self._state = BotState.STOPPED
        self._running = False
        self._start_time = 0.0

        # Componentes
        self._fxopen: Optional[FXOpenClient] = None
        self._strategy_engine: Optional[StrategyEngine] = None
        self._oms: Optional[OrderManagementSystem] = None
        self._risk_manager: Optional[RiskManager] = None
        self._order_books: Dict[str, OrderBook] = {}
        self._ml_manager: Optional[MLManager] = None

        # Trade tracking for ML
        self._active_trades: Dict[str, dict] = {}  # position_id -> trade_info
        self._ml_save_interval = 300  # Salvar estado ML a cada 5 minutos
        self._last_ml_save = 0.0

        # Logging
        self._logger = AsyncLogger(
            'elibot',
            log_dir=self.config.log_dir,
            console_output=True
        )
        self._latency_logger = LatencyLogger(
            'latency',
            log_dir=self.config.log_dir
        )

        # Callbacks
        self._on_state_change: List[Callable[[str], None]] = []
        self._on_signal: List[Callable[[Signal], None]] = []
        self._on_trade: List[Callable[[dict], None]] = []

        # Loop de eventos
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None

        logger.info("EliBotAPI inicializado")

    def _load_config(self, config_path: str) -> BotConfig:
        """Carrega configuração de arquivo"""
        config = BotConfig()

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            # Aplicar configurações
            if 'instruments' in data:
                config.symbols = [
                    i['symbol'] for i in data['instruments']
                    if i.get('enabled', True)
                ]

            if 'strategy' in data:
                config.mode = data['strategy'].get('mode', 'paper')
                config.enabled = data['strategy'].get('enabled', True)

        return config

    def _change_state(self, new_state: str) -> None:
        """Muda estado do bot"""
        old_state = self._state
        self._state = new_state

        logger.info(f"Estado: {old_state} -> {new_state}")

        for cb in self._on_state_change:
            try:
                cb(new_state)
            except Exception as e:
                logger.error(f"Erro no callback de estado: {e}")

    async def start(self) -> bool:
        """
        Inicia o bot

        Returns:
            True se iniciado com sucesso
        """
        if self._state != BotState.STOPPED:
            logger.warning(f"Bot não está parado (estado: {self._state})")
            return False

        self._change_state(BotState.STARTING)
        self._start_time = time.time()

        try:
            # Inicializar componentes
            await self._initialize_components()

            # Conectar à FXOpen
            if not await self._connect():
                raise Exception("Falha ao conectar à FXOpen")

            # Iniciar loops
            self._running = True
            self._main_task = asyncio.create_task(self._main_loop())

            self._change_state(BotState.RUNNING)
            logger.info("Bot iniciado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            self._change_state(BotState.ERROR)
            return False

    async def stop(self) -> None:
        """Para o bot"""
        if self._state not in (BotState.RUNNING, BotState.ERROR):
            return

        self._change_state(BotState.STOPPING)
        self._running = False

        try:
            # Cancelar task principal
            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass

            # Fechar posições se configurado
            if self.config.mode == 'live':
                await self._close_all_positions()

            # Salvar estado ML antes de fechar
            if self._ml_manager:
                try:
                    ml_dir = os.path.join(self.config.data_dir, 'ml')
                    os.makedirs(ml_dir, exist_ok=True)
                    ml_state_path = os.path.join(ml_dir, 'ml_state.json')
                    self._ml_manager.save_state(ml_state_path)
                    logger.info(f"Estado ML salvo em {ml_state_path}")
                except Exception as e:
                    logger.error(f"Erro ao salvar estado ML: {e}")

            # Desconectar
            if self._fxopen:
                await self._fxopen.disconnect()

            # Fechar loggers
            self._logger.close()
            self._latency_logger.close()

        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")

        self._change_state(BotState.STOPPED)
        logger.info("Bot parado")

    async def _initialize_components(self) -> None:
        """Inicializa componentes internos"""
        # Risk Manager
        risk_config = os.path.join(self.config.config_dir, 'risk_limits.json')
        if os.path.exists(risk_config):
            self._risk_manager = RiskManager(config_path=risk_config)
        else:
            self._risk_manager = RiskManager()

        # OMS
        self._oms = OrderManagementSystem()

        # Strategy Engine
        strategy_config = os.path.join(self.config.config_dir, 'strategy_params.yaml')
        self._strategy_engine = StrategyEngine(
            config_path=strategy_config if os.path.exists(strategy_config) else None
        )

        # Registrar callback de sinais
        self._strategy_engine.register_signal_callback(self._on_strategy_signal)

        # Order Books
        for symbol in self.config.symbols:
            self._order_books[symbol] = OrderBook(symbol)

        # ML Manager
        ml_config = MLConfig(
            model_store_path=os.path.join(self.config.data_dir, 'ml', 'models'),
            state_file=os.path.join(self.config.data_dir, 'ml', 'ml_state.json'),
            training_interval=100,  # Treinar a cada 100 trades
            min_samples_to_train=50,
            enable_error_learning=True,
            enable_pattern_analysis=True,
            enable_adaptive_optimization=True,
            enable_auto_training=True
        )
        self._ml_manager = MLManager(
            config=ml_config,
            strategy_names=self._strategy_engine.get_strategy_names() if self._strategy_engine else []
        )

        # Carregar estado ML anterior se existir
        ml_state_path = os.path.join(self.config.data_dir, 'ml', 'ml_state.json')
        if os.path.exists(ml_state_path):
            try:
                self._ml_manager.load_state(ml_state_path)
                logger.info("Estado ML carregado com sucesso")
            except Exception as e:
                logger.warning(f"Falha ao carregar estado ML: {e}")

        # Conectar ML ao strategy engine para ajuste de pesos
        if self._strategy_engine and self._ml_manager:
            self._ml_manager.register_weight_callback(
                self._strategy_engine.update_strategy_weights
            )

        logger.info("Componentes inicializados (incluindo ML)")

    async def _connect(self) -> bool:
        """Conecta à FXOpen"""
        self._fxopen = FXOpenClient(self.config.fxopen)

        if not await self._fxopen.connect():
            return False

        # Registrar callbacks
        self._fxopen.register_tick_callback(self._on_tick)
        self._fxopen.register_trade_callback(self._on_trade_update)

        # Inscrever para ticks
        await self._fxopen.subscribe_ticks(self.config.symbols)

        # Atualizar account info no risk manager
        if self._fxopen.account:
            self._risk_manager.update_account(
                self._fxopen.account.balance,
                self._fxopen.account.free_margin
            )

        return True

    async def _main_loop(self) -> None:
        """Loop principal do bot"""
        logger.info("Loop principal iniciado")

        while self._running:
            try:
                # Atualizar ticks manualmente se não estiver usando WebSocket
                for symbol in self.config.symbols:
                    tick = await self._fxopen.get_tick(symbol)
                    if tick:
                        self._on_tick(tick)

                # Atualizar posições
                await self._update_positions()

                # Atualizar account
                account = await self._fxopen.get_account_info()
                if account:
                    self._risk_manager.update_account(account.balance, account.free_margin)

                # Auto-save do estado ML periodicamente
                await self._auto_save_ml_state()

                # Pequeno delay para não sobrecarregar API
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                await asyncio.sleep(1)

        logger.info("Loop principal encerrado")

    async def _auto_save_ml_state(self) -> None:
        """Salva estado ML periodicamente"""
        try:
            current_time = time.time()

            # Salvar a cada intervalo definido
            if current_time - self._last_ml_save >= self._ml_save_interval:
                if self._ml_manager:
                    ml_dir = os.path.join(self.config.data_dir, 'ml')
                    os.makedirs(ml_dir, exist_ok=True)

                    ml_state_path = os.path.join(ml_dir, 'ml_state.json')
                    self._ml_manager.save_state(ml_state_path)
                    self._last_ml_save = current_time
                    logger.debug(f"Estado ML salvo em {ml_state_path}")

        except Exception as e:
            logger.error(f"Erro ao salvar estado ML: {e}")

    def _on_tick(self, tick) -> None:
        """Callback para ticks"""
        start = self._latency_logger.start_timer()

        try:
            # Log periódico de ticks recebidos
            if not hasattr(self, '_tick_count'):
                self._tick_count = 0
            self._tick_count += 1
            if self._tick_count % 50 == 1:
                logger.info(f"Processando tick #{self._tick_count}: {tick.symbol} bid={tick.bid:.5f} ask={tick.ask:.5f}")

            # Atualizar order book
            if tick.symbol in self._order_books:
                book = self._order_books[tick.symbol]
                book.update(BookSide.BID, tick.bid, tick.bid_volume)
                book.update(BookSide.ASK, tick.ask, tick.ask_volume)

            # Processar estratégias
            if self._strategy_engine and self.config.enabled:
                from trading.book import Quote
                quote = Quote(
                    bid_price=tick.bid,
                    bid_size=tick.bid_volume,
                    ask_price=tick.ask,
                    ask_size=tick.ask_volume,
                    timestamp=time.time()
                )

                book = self._order_books.get(tick.symbol)
                if book:
                    self._strategy_engine.on_tick(tick.symbol, quote, book)

        except Exception as e:
            logger.error(f"Erro ao processar tick: {e}")

        self._latency_logger.stop_timer(start, 'tick_processing')

    def _on_strategy_signal(self, signal: Signal) -> None:
        """Callback para sinais de estratégia"""
        logger.info(f"Sinal recebido: {signal.signal_type.value} {signal.symbol} @ {signal.price}")
        logger.info(f"Modo atual: {self.config.mode}, Trading habilitado: {self.config.enabled}")

        # Emitir para callbacks externos
        for cb in self._on_signal:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Erro no callback de sinal: {e}")

        # Executar sinal se em modo live
        if self.config.mode == 'live' and self.config.enabled:
            logger.info("Executando sinal em modo LIVE...")
            asyncio.create_task(self._execute_signal(signal))
        else:
            logger.info(f"Sinal ignorado (modo={self.config.mode}, enabled={self.config.enabled})")

    async def _execute_signal(self, signal: Signal) -> None:
        """Executa sinal de trading"""
        logger.info(f"Iniciando execução de sinal: {signal.signal_type.value} {signal.symbol}")

        # ML: Verificar risco de erro baseado em padrões
        ml_risk = 0.0
        ml_recommendation = None
        if self._ml_manager:
            try:
                # Obter features atuais do mercado
                book = self._order_books.get(signal.symbol)
                market_features = {}
                if book:
                    quote = book.get_quote()
                    market_features = {
                        'spread_pips': quote.spread_bps / 10 if hasattr(quote, 'spread_bps') else 0,
                        'volatility': getattr(signal, 'volatility', 0.01),
                        'signal_confidence': signal.confidence if hasattr(signal, 'confidence') else 0.7,
                    }

                # Obter predição de risco do ML
                ml_prediction = self._ml_manager.predict_trade_risk(
                    symbol=signal.symbol,
                    strategy_name=signal.strategy_name,
                    signal_type=signal.signal_type.value,
                    features=market_features
                )

                if ml_prediction:
                    ml_risk = ml_prediction.get('risk_score', 0.0)
                    ml_recommendation = ml_prediction.get('recommendation', None)

                    logger.info(f"ML Risk Score: {ml_risk:.2f}")

                    # Se risco ML > 70%, reduzir tamanho da posição
                    if ml_risk > 0.7:
                        original_size = signal.size or 0.01
                        signal.size = original_size * 0.5
                        logger.warning(
                            f"ML: Alto risco detectado ({ml_risk:.2f}). "
                            f"Tamanho reduzido: {original_size} -> {signal.size}"
                        )

                    # Se risco ML > 90%, rejeitar sinal
                    if ml_risk > 0.9:
                        logger.warning(f"ML: Sinal rejeitado devido a risco muito alto ({ml_risk:.2f})")
                        return

            except Exception as e:
                logger.error(f"Erro ao obter predição ML: {e}")

        # Verificar risco tradicional
        order = Order(
            symbol=signal.symbol,
            side=Side.BUY if signal.signal_type == SignalType.BUY else Side.SELL,
            quantity=signal.size or 0.01,
            order_type=OrderType.MARKET,
            strategy_name=signal.strategy_name
        )

        risk_check = self._risk_manager.check_pre_trade(order)
        logger.info(f"Risk check: passed={risk_check.passed}, message={risk_check.message}")

        if not risk_check:
            logger.warning(f"Sinal rejeitado por risco: {risk_check.message}")
            return

        # Executar ordem
        try:
            position_id = None

            if signal.signal_type == SignalType.BUY:
                logger.info(f"Abrindo posição BUY {signal.symbol} volume={signal.size or 0.01}")
                position_id = await self._fxopen.open_position(
                    symbol=signal.symbol,
                    side=OrderSide.BUY,
                    volume=signal.size or 0.01,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            elif signal.signal_type == SignalType.SELL:
                logger.info(f"Abrindo posição SELL {signal.symbol} volume={signal.size or 0.01}")
                position_id = await self._fxopen.open_position(
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    volume=signal.size or 0.01,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
            elif signal.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
                # Encontrar e fechar posição
                logger.info(f"Fechando posições para {signal.symbol}")
                positions = await self._fxopen.get_positions()
                for pos in positions:
                    if pos.symbol == signal.symbol:
                        await self._fxopen.close_position(pos.position_id)
                        logger.info(f"Posição {pos.position_id} fechada")
                        break

            if position_id:
                logger.info(f"Ordem executada com sucesso! ID: {position_id}")
            else:
                logger.warning("Ordem não retornou ID de posição")

        except Exception as e:
            logger.error(f"Erro ao executar sinal: {e}")
            import traceback
            traceback.print_exc()

    def _on_trade_update(self, data: dict) -> None:
        """Callback para atualizações de trade"""
        # Processar para ML learning
        self._process_trade_for_ml(data)

        # Callbacks externos
        for cb in self._on_trade:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Erro no callback de trade: {e}")

    def _process_trade_for_ml(self, data: dict) -> None:
        """Processa trade para aprendizado ML"""
        try:
            if not self._ml_manager:
                return

            position_id = data.get('PositionId') or data.get('position_id')
            if not position_id:
                return

            # Verificar tipo de evento
            event_type = data.get('Type') or data.get('type', '')

            # Trade aberto - registrar para tracking
            if event_type in ['Opened', 'opened', 'Position']:
                symbol = data.get('Symbol') or data.get('symbol', '')
                side = data.get('Side') or data.get('side', '')
                open_price = data.get('OpenPrice') or data.get('open_price', 0)
                volume = data.get('Volume') or data.get('volume', 0)
                strategy_name = data.get('Comment') or data.get('strategy', 'unknown')

                # Obter features de mercado atuais
                book = self._order_books.get(symbol)
                market_features = {}
                if book:
                    quote = book.get_quote()
                    market_features = {
                        'spread_pips': quote.spread_bps / 10 if hasattr(quote, 'spread_bps') else 0,
                        'bid': quote.bid_price,
                        'ask': quote.ask_price,
                    }

                self._active_trades[str(position_id)] = {
                    'position_id': position_id,
                    'symbol': symbol,
                    'side': side,
                    'entry_price': open_price,
                    'volume': volume,
                    'strategy_name': strategy_name,
                    'open_time': time.time(),
                    'market_features': market_features,
                    'signal_confidence': data.get('signal_confidence', 0.7)
                }
                logger.debug(f"Trade registrado para ML: {position_id}")

            # Trade fechado - alimentar ML
            elif event_type in ['Closed', 'closed', 'PositionClosed']:
                trade_info = self._active_trades.pop(str(position_id), None)

                if trade_info:
                    close_price = data.get('ClosePrice') or data.get('close_price', 0)
                    profit = data.get('Profit') or data.get('profit', 0)
                    close_time = time.time()

                    # Calcular métricas
                    holding_time_ms = int((close_time - trade_info['open_time']) * 1000)
                    entry_price = trade_info['entry_price']
                    pip_size = 0.0001 if 'JPY' not in trade_info['symbol'] else 0.01
                    slippage_pips = abs(close_price - entry_price) / pip_size if entry_price else 0

                    # Obter histórico de preços recente
                    price_history = self._get_recent_prices(trade_info['symbol'])

                    # Determinar regime de mercado
                    market_regime = self._strategy_engine.get_market_regime(
                        trade_info['symbol']
                    ) if self._strategy_engine else 'UNKNOWN'

                    # Alimentar o ML Manager
                    self._ml_manager.learn_from_trade(
                        trade_id=str(position_id),
                        symbol=trade_info['symbol'],
                        strategy_name=trade_info['strategy_name'],
                        signal_type='BUY' if trade_info['side'] in ['Buy', 'buy'] else 'SELL',
                        entry_price=entry_price,
                        exit_price=close_price,
                        pnl=profit,
                        holding_time_ms=holding_time_ms,
                        market_regime=market_regime,
                        signal_confidence=trade_info.get('signal_confidence', 0.7),
                        features=trade_info.get('market_features', {}),
                        price_history=price_history
                    )

                    logger.info(
                        f"Trade {position_id} processado pelo ML: "
                        f"PnL={profit:.2f}, regime={market_regime}"
                    )

        except Exception as e:
            logger.error(f"Erro ao processar trade para ML: {e}")

    def _get_recent_prices(self, symbol: str, lookback: int = 50) -> List[float]:
        """Obtém histórico recente de preços"""
        try:
            book = self._order_books.get(symbol)
            if book and hasattr(book, 'price_history'):
                return list(book.price_history)[-lookback:]

            # Fallback: usar mid price atual
            if book:
                quote = book.get_quote()
                mid = (quote.bid_price + quote.ask_price) / 2
                return [mid]  # Pelo menos um preço

            return []
        except Exception:
            return []

    async def _update_positions(self) -> None:
        """Atualiza posições"""
        try:
            positions = await self._fxopen.get_positions()

            for pos in positions:
                from trading.strategy.base_strategy import Position
                position = Position(
                    symbol=pos.symbol,
                    side='long' if pos.side == 'Buy' else 'short',
                    size=pos.volume,
                    entry_price=pos.open_price,
                    entry_time=pos.open_time.timestamp(),
                    unrealized_pnl=pos.profit
                )

                self._risk_manager.update_position(position)
                self._strategy_engine.update_position(position)

        except Exception as e:
            logger.error(f"Erro ao atualizar posições: {e}")

    async def _close_all_positions(self) -> None:
        """Fecha todas as posições"""
        try:
            positions = await self._fxopen.get_positions()
            for pos in positions:
                await self._fxopen.close_position(pos.position_id)
                logger.info(f"Posição fechada: {pos.position_id}")
        except Exception as e:
            logger.error(f"Erro ao fechar posições: {e}")

    # ==================== Public API ====================

    def register_state_callback(self, callback: Callable[[str], None]) -> None:
        """Registra callback para mudanças de estado"""
        self._on_state_change.append(callback)

    def register_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Registra callback para sinais"""
        self._on_signal.append(callback)

    def register_trade_callback(self, callback: Callable[[dict], None]) -> None:
        """Registra callback para trades"""
        self._on_trade.append(callback)

    def enable_trading(self) -> None:
        """Habilita trading"""
        self.config.enabled = True
        logger.info("Trading habilitado")

    def disable_trading(self) -> None:
        """Desabilita trading"""
        self.config.enabled = False
        logger.info("Trading desabilitado")

    def set_mode(self, mode: str) -> None:
        """Define modo (paper/live)"""
        if mode in ('paper', 'live'):
            self.config.mode = mode
            logger.info(f"Modo definido: {mode}")

    async def get_account(self) -> Optional[Any]:
        """Obtém informações da conta"""
        if self._fxopen:
            return await self._fxopen.get_account_info()
        return None

    async def get_positions(self) -> List[Any]:
        """Obtém posições abertas"""
        if self._fxopen:
            return await self._fxopen.get_positions()
        return []

    async def get_tick(self, symbol: str) -> Optional[Any]:
        """Obtém tick atual"""
        if self._fxopen:
            return await self._fxopen.get_tick(symbol)
        return None

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Obtém order book"""
        return self._order_books.get(symbol)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == BotState.RUNNING

    @property
    def uptime(self) -> float:
        """Tempo de execução em segundos"""
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0

    @property
    def stats(self) -> dict:
        """Estatísticas do bot"""
        return {
            'state': self._state,
            'uptime': self.uptime,
            'mode': self.config.mode,
            'trading_enabled': self.config.enabled,
            'symbols': self.config.symbols,
            'risk': self._risk_manager.stats if self._risk_manager else {},
            'oms': self._oms.stats if self._oms else {},
            'strategies': self._strategy_engine.get_aggregate_stats() if self._strategy_engine else {},
            'ml': self._ml_manager.get_statistics() if self._ml_manager else {}
        }


# Função de conveniência para criar e iniciar o bot
async def create_bot(config_path: str = None) -> EliBotAPI:
    """Cria e configura o bot"""
    config = BotConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            # Aplicar configurações...

    return EliBotAPI(config=config)


def run_bot(config_path: str = None):
    """Executa o bot (função síncrona)"""
    async def _run():
        bot = await create_bot(config_path)
        await bot.start()

        try:
            # Manter rodando
            while bot.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await bot.stop()

    asyncio.run(_run())
