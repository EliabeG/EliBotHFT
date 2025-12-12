"""
Strategy Engine - Motor central de estratégias
Coordena múltiplas estratégias e gerencia sinais
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import threading
import time
import logging
import yaml
import os

from .base_strategy import BaseStrategy, Signal, SignalType, MarketState, Position
from .scalping_strategy import ScalpingStrategy
from .momentum_strategy import MomentumStrategy
from ..book import OrderBook, Quote

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeight:
    """Peso de uma estratégia na composição de sinais"""
    strategy_name: str
    weight: float = 1.0
    enabled: bool = True


class StrategyEngine:
    """
    Motor de Estratégias

    Funcionalidades:
    - Gerencia múltiplas estratégias simultâneas
    - Combina sinais de diferentes estratégias
    - Recarrega parâmetros em tempo real
    - Roteia sinais para o OMS
    """

    def __init__(self, config_path: str = None):
        """
        Inicializa motor de estratégias

        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        # Estratégias registradas
        self._strategies: Dict[str, BaseStrategy] = {}
        self._weights: Dict[str, StrategyWeight] = {}

        # Estado
        self._running = False
        self._lock = threading.Lock()

        # Order books
        self._books: Dict[str, OrderBook] = {}

        # Callbacks
        self._signal_callbacks: List[Callable[[Signal], None]] = []

        # Estatísticas
        self._signals_generated = 0
        self._ticks_processed = 0
        self._last_signal_time = 0.0

        # Carregar configuração
        if config_path and os.path.exists(config_path):
            self._load_config()

        # Registrar estratégias padrão
        self._register_default_strategies()

        logger.info("StrategyEngine inicializado")

    def _load_config(self) -> None:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuração carregada de {self.config_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            self.config = {}

    def reload_config(self) -> None:
        """Recarrega configuração em tempo real"""
        self._load_config()

        # Atualizar estratégias
        for name, strategy in self._strategies.items():
            if name in self.config.get('strategies', {}):
                strategy.update_config(self.config['strategies'][name])

        logger.info("Configuração recarregada")

    def _register_default_strategies(self) -> None:
        """Registra estratégias padrão"""
        # Scalping
        scalping_config = self.config.get('market_making', {})
        self.register_strategy(ScalpingStrategy(scalping_config))

        # Momentum
        momentum_config = self.config.get('momentum', {})
        self.register_strategy(MomentumStrategy(momentum_config))

    def register_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """
        Registra uma estratégia

        Args:
            strategy: Instância da estratégia
            weight: Peso na composição de sinais
        """
        with self._lock:
            self._strategies[strategy.name] = strategy
            self._weights[strategy.name] = StrategyWeight(
                strategy_name=strategy.name,
                weight=weight,
                enabled=strategy.enabled
            )

        logger.info(f"Estratégia registrada: {strategy.name} (weight={weight})")

    def unregister_strategy(self, name: str) -> None:
        """Remove uma estratégia"""
        with self._lock:
            if name in self._strategies:
                del self._strategies[name]
                del self._weights[name]
                logger.info(f"Estratégia removida: {name}")

    def register_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Registra callback para sinais"""
        self._signal_callbacks.append(callback)

    def _emit_signal(self, signal: Signal) -> None:
        """Emite sinal para callbacks"""
        self._signals_generated += 1
        self._last_signal_time = time.time()

        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Erro no callback de sinal: {e}")

    def register_book(self, symbol: str, book: OrderBook) -> None:
        """Registra order book para um símbolo"""
        self._books[symbol] = book

    def on_tick(self, symbol: str, quote: Quote, book: OrderBook = None) -> List[Signal]:
        """
        Processa tick de mercado

        Args:
            symbol: Símbolo
            quote: Cotação
            book: Order book (opcional)

        Returns:
            Lista de sinais gerados
        """
        self._ticks_processed += 1

        if book is None:
            book = self._books.get(symbol)

        if book is None:
            return []

        # Criar estado de mercado
        state = MarketState(
            symbol=symbol,
            quote=quote,
            book=book,
            timestamp=time.time()
        )

        signals = []

        # Processar cada estratégia
        with self._lock:
            for name, strategy in self._strategies.items():
                if not strategy.enabled:
                    continue

                weight = self._weights.get(name)
                if not weight or not weight.enabled:
                    continue

                try:
                    signal = strategy.on_tick(state)
                    if signal:
                        # Ajustar confiança pelo peso
                        signal.confidence *= weight.weight
                        signals.append(signal)
                        self._emit_signal(signal)

                except Exception as e:
                    logger.error(f"Erro na estratégia {name}: {e}")

        return signals

    def on_quote_update(self, symbol: str, quote: Quote) -> List[Signal]:
        """
        Processa atualização de cotação

        Args:
            symbol: Símbolo
            quote: Nova cotação

        Returns:
            Lista de sinais
        """
        signals = []

        with self._lock:
            for name, strategy in self._strategies.items():
                if not strategy.enabled:
                    continue

                try:
                    signal = strategy.on_quote(quote)
                    if signal:
                        signals.append(signal)
                        self._emit_signal(signal)
                except Exception as e:
                    logger.error(f"Erro na estratégia {name}: {e}")

        return signals

    def update_position(self, position: Position) -> None:
        """Atualiza posição em todas as estratégias"""
        with self._lock:
            for strategy in self._strategies.values():
                strategy.on_position_update(position)

    def notify_fill(self, strategy_name: str, order_id: str,
                   fill_price: float, fill_size: float) -> None:
        """Notifica fill para estratégia específica"""
        strategy = self._strategies.get(strategy_name)
        if strategy:
            strategy.on_fill(order_id, fill_price, fill_size)

    def notify_trade_closed(self, strategy_name: str, symbol: str, pnl: float) -> None:
        """Notifica fechamento de trade"""
        strategy = self._strategies.get(strategy_name)
        if strategy:
            strategy.on_trade_closed(symbol, pnl)

    def enable_strategy(self, name: str) -> bool:
        """Habilita estratégia"""
        with self._lock:
            if name in self._strategies:
                self._strategies[name].enable()
                self._weights[name].enabled = True
                return True
        return False

    def disable_strategy(self, name: str) -> bool:
        """Desabilita estratégia"""
        with self._lock:
            if name in self._strategies:
                self._strategies[name].disable()
                self._weights[name].enabled = False
                return True
        return False

    def set_strategy_weight(self, name: str, weight: float) -> bool:
        """Define peso de uma estratégia"""
        with self._lock:
            if name in self._weights:
                self._weights[name].weight = weight
                logger.info(f"Peso de {name} alterado para {weight}")
                return True
        return False

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Obtém estratégia por nome"""
        return self._strategies.get(name)

    def list_strategies(self) -> List[str]:
        """Lista nomes das estratégias"""
        return list(self._strategies.keys())

    def get_aggregate_stats(self) -> dict:
        """Obtém estatísticas agregadas"""
        stats = {
            'engine': {
                'signals_generated': self._signals_generated,
                'ticks_processed': self._ticks_processed,
                'active_strategies': sum(1 for s in self._strategies.values() if s.enabled)
            },
            'strategies': {}
        }

        for name, strategy in self._strategies.items():
            stats['strategies'][name] = strategy.stats

        return stats

    def get_strategy_names(self) -> List[str]:
        """
        Retorna nomes das estratégias registradas
        (Alias para list_strategies, usado pelo MLManager)
        """
        return self.list_strategies()

    def update_strategy_weights(self, weights: Dict[str, float]) -> None:
        """
        Atualiza pesos de múltiplas estratégias de uma vez
        (Usado pelo MLManager para ajustar pesos baseado em performance)

        Args:
            weights: Dict com nome da estratégia -> novo peso
        """
        with self._lock:
            for name, weight in weights.items():
                if name in self._weights:
                    old_weight = self._weights[name].weight
                    self._weights[name].weight = weight
                    logger.info(f"ML: Peso de {name} ajustado: {old_weight:.3f} -> {weight:.3f}")
                else:
                    logger.warning(f"ML: Estratégia '{name}' não encontrada para ajuste de peso")

    def get_market_regime(self, symbol: str) -> str:
        """
        Determina regime de mercado atual para um símbolo

        Regimes:
        - TRENDING_UP: Tendência de alta
        - TRENDING_DOWN: Tendência de baixa
        - RANGING: Mercado lateral
        - VOLATILE: Alta volatilidade
        - UNKNOWN: Não determinado

        Args:
            symbol: Símbolo do instrumento

        Returns:
            String indicando o regime de mercado
        """
        try:
            book = self._books.get(symbol)
            if not book:
                return 'UNKNOWN'

            # Obter histórico de preços se disponível
            if hasattr(book, 'price_history') and len(book.price_history) >= 10:
                prices = list(book.price_history)[-20:]
            else:
                # Usar quote atual como fallback
                quote = book.get_quote()
                if quote:
                    return 'UNKNOWN'  # Sem histórico suficiente
                return 'UNKNOWN'

            if len(prices) < 10:
                return 'UNKNOWN'

            # Calcular métricas simples
            import numpy as np

            prices_arr = np.array(prices)
            returns = np.diff(prices_arr) / prices_arr[:-1]

            # Volatilidade
            volatility = np.std(returns) * 100  # Em percentual

            # Tendência (média dos retornos)
            trend = np.mean(returns)

            # Direção (quantos retornos positivos vs negativos)
            positive_returns = np.sum(returns > 0)
            negative_returns = np.sum(returns < 0)

            # Classificar regime
            if volatility > 0.5:  # Alta volatilidade
                return 'VOLATILE'
            elif abs(trend) < 0.0001 and abs(positive_returns - negative_returns) < 3:
                return 'RANGING'
            elif trend > 0.0001 and positive_returns > negative_returns:
                return 'TRENDING_UP'
            elif trend < -0.0001 and negative_returns > positive_returns:
                return 'TRENDING_DOWN'
            else:
                return 'RANGING'

        except Exception as e:
            logger.debug(f"Erro ao determinar regime de mercado: {e}")
            return 'UNKNOWN'

    def start(self) -> None:
        """Inicia o motor"""
        self._running = True
        logger.info("StrategyEngine iniciado")

    def stop(self) -> None:
        """Para o motor"""
        self._running = False
        logger.info("StrategyEngine parado")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def strategy_count(self) -> int:
        return len(self._strategies)
