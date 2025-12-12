"""
Strategy Orchestrator - Orquestrador de Múltiplas Estratégias para HFT
Coordena, pondera e otimiza sinais de múltiplas estratégias
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import time
import logging
import numpy as np

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from .market_making_strategy import MarketMakingStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .order_flow_strategy import OrderFlowStrategy
from .momentum_strategy import MomentumStrategy
from .scalping_strategy import ScalpingStrategy
from ..book import OrderBook, Quote
from ..risk.advanced_risk_manager import AdvancedRiskManager, RiskLevel

logger = logging.getLogger(__name__)


class SignalAggregation(Enum):
    """Método de agregação de sinais"""
    WEIGHTED_VOTE = 'weighted_vote'  # Voto ponderado
    HIGHEST_CONFIDENCE = 'highest_confidence'  # Maior confiança
    UNANIMOUS = 'unanimous'  # Todos concordam
    MAJORITY = 'majority'  # Maioria


class MarketRegime(Enum):
    """Regime de mercado detectado"""
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    QUIET = 'quiet'


@dataclass
class StrategyPerformance:
    """Performance de uma estratégia"""
    name: str
    signals_generated: int = 0
    trades_executed: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.5
    profit_factor: float = 1.0
    sharpe_ratio: float = 0.0
    weight: float = 1.0
    enabled: bool = True


@dataclass
class AggregatedSignal:
    """Sinal agregado de múltiplas estratégias"""
    signal_type: SignalType
    symbol: str
    price: float
    strength: SignalStrength
    confidence: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    contributing_strategies: List[str]
    aggregation_method: SignalAggregation
    regime: MarketRegime
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyOrchestrator:
    """
    Orquestrador de Estratégias para HFT

    Funcionalidades:
    - Coordena múltiplas estratégias simultaneamente
    - Detecta regime de mercado
    - Agrega sinais com ponderação dinâmica
    - Ajusta pesos baseado em performance
    - Integração com risk manager avançado
    - Otimizado para EURUSD
    """

    def __init__(self, risk_manager: AdvancedRiskManager = None,
                 config: Dict[str, Any] = None):
        """
        Inicializa orquestrador

        Args:
            risk_manager: Gerenciador de risco avançado
            config: Configuração
        """
        self.config = config or {}
        self.risk_manager = risk_manager

        # Estratégias
        self._strategies: Dict[str, BaseStrategy] = {}
        self._performance: Dict[str, StrategyPerformance] = {}

        # Configuração de agregação
        self.aggregation_method = SignalAggregation(
            self.config.get('aggregation_method', 'weighted_vote')
        )
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.min_strategies_agree = self.config.get('min_strategies_agree', 2)

        # Regime detection
        self._regime = MarketRegime.RANGING
        self._regime_history: deque = deque(maxlen=100)
        self._price_history: deque = deque(maxlen=200)
        self._volatility_history: deque = deque(maxlen=50)

        # Regime-based strategy weights
        self._regime_weights: Dict[MarketRegime, Dict[str, float]] = {
            MarketRegime.TRENDING_UP: {
                'Momentum': 1.5,
                'Breakout': 1.3,
                'MeanReversion': 0.3,
                'MarketMaking': 0.5,
                'OrderFlow': 1.0,
                'Scalping': 0.7
            },
            MarketRegime.TRENDING_DOWN: {
                'Momentum': 1.5,
                'Breakout': 1.3,
                'MeanReversion': 0.3,
                'MarketMaking': 0.5,
                'OrderFlow': 1.0,
                'Scalping': 0.7
            },
            MarketRegime.RANGING: {
                'Momentum': 0.5,
                'Breakout': 0.3,
                'MeanReversion': 1.5,
                'MarketMaking': 1.3,
                'OrderFlow': 1.0,
                'Scalping': 1.2
            },
            MarketRegime.VOLATILE: {
                'Momentum': 0.7,
                'Breakout': 1.5,
                'MeanReversion': 0.5,
                'MarketMaking': 0.3,
                'OrderFlow': 1.2,
                'Scalping': 0.5
            },
            MarketRegime.QUIET: {
                'Momentum': 0.3,
                'Breakout': 0.2,
                'MeanReversion': 1.2,
                'MarketMaking': 1.5,
                'OrderFlow': 0.8,
                'Scalping': 1.0
            }
        }

        # Signal history
        self._signal_history: deque = deque(maxlen=100)
        self._last_signal_time = 0.0
        self.signal_cooldown_ms = self.config.get('signal_cooldown_ms', 1000)

        # Order books
        self._books: Dict[str, OrderBook] = {}

        # Threading
        self._lock = threading.RLock()

        # Callbacks
        self._signal_callbacks: List[Callable[[AggregatedSignal], None]] = []

        # Initialize default strategies
        self._initialize_strategies()

        logger.info("StrategyOrchestrator initialized")

    def _initialize_strategies(self) -> None:
        """Inicializa estratégias padrão"""
        # Configurações específicas para EURUSD
        eurusd_config = {
            'pip_size': 0.0001,
            'signal_cooldown_ms': 2000
        }

        # Market Making
        mm_config = {
            **eurusd_config,
            'min_spread_pips': 0.3,
            'target_spread_pips': 0.5,
            'max_inventory_lots': 0.5,
            'stop_loss_pips': 10,
            'take_profit_pips': 3
        }
        self.register_strategy(MarketMakingStrategy(mm_config))

        # Mean Reversion
        mr_config = {
            **eurusd_config,
            'lookback_period': 50,
            'bb_period': 20,
            'entry_z_score': 2.0,
            'stop_loss_pips': 15,
            'take_profit_pips': 8
        }
        self.register_strategy(MeanReversionStrategy(mr_config))

        # Breakout
        bo_config = {
            **eurusd_config,
            'atr_fast_period': 10,
            'atr_slow_period': 50,
            'stop_loss_atr_multiplier': 1.5,
            'take_profit_atr_multiplier': 2.0
        }
        self.register_strategy(BreakoutStrategy(bo_config))

        # Order Flow
        of_config = {
            **eurusd_config,
            'imbalance_threshold': 0.4,
            'stop_loss_pips': 8,
            'take_profit_pips': 5
        }
        self.register_strategy(OrderFlowStrategy(of_config))

        # Momentum (já existente)
        mom_config = {
            **eurusd_config,
            'short_window': 10,
            'medium_window': 30,
            'long_window': 100,
            'entry_threshold': 0.0001,
            'stop_loss_pips': 30,
            'take_profit_pips': 5
        }
        self.register_strategy(MomentumStrategy(mom_config))

        # Scalping (já existente)
        scalp_config = {
            **eurusd_config,
            'min_spread_pips': 0.5,
            'max_spread_pips': 3.0,
            'stop_loss_pips': 5,
            'take_profit_pips': 3
        }
        self.register_strategy(ScalpingStrategy(scalp_config))

    def register_strategy(self, strategy: BaseStrategy, initial_weight: float = 1.0) -> None:
        """Registra uma estratégia"""
        with self._lock:
            self._strategies[strategy.name] = strategy
            self._performance[strategy.name] = StrategyPerformance(
                name=strategy.name,
                weight=initial_weight
            )

        logger.info(f"Strategy registered: {strategy.name}")

    def unregister_strategy(self, name: str) -> None:
        """Remove uma estratégia"""
        with self._lock:
            self._strategies.pop(name, None)
            self._performance.pop(name, None)

        logger.info(f"Strategy unregistered: {name}")

    def register_book(self, symbol: str, book: OrderBook) -> None:
        """Registra order book"""
        self._books[symbol] = book

    def register_signal_callback(self, callback: Callable[[AggregatedSignal], None]) -> None:
        """Registra callback para sinais agregados"""
        self._signal_callbacks.append(callback)

    # ==================== TICK PROCESSING ====================

    def on_tick(self, symbol: str, quote: Quote, book: OrderBook = None) -> Optional[AggregatedSignal]:
        """
        Processa tick através de todas as estratégias

        Returns:
            Sinal agregado ou None
        """
        if book is None:
            book = self._books.get(symbol)

        if book is None:
            return None

        # Atualizar dados de mercado
        self._update_market_data(quote)

        # Detectar regime
        self._detect_regime()

        # Verificar cooldown
        if not self._can_generate_signal():
            return None

        # Verificar risk manager
        if self.risk_manager:
            if not self.risk_manager.is_trading_allowed:
                return None

        # Criar estado de mercado
        state = MarketState(
            symbol=symbol,
            quote=quote,
            book=book,
            timestamp=time.time()
        )

        # Coletar sinais de todas as estratégias
        signals = []

        with self._lock:
            for name, strategy in self._strategies.items():
                if not strategy.enabled:
                    continue

                perf = self._performance.get(name)
                if not perf or not perf.enabled:
                    continue

                try:
                    signal = strategy.on_tick(state)
                    if signal:
                        # Ajustar peso pelo regime
                        regime_weight = self._get_regime_weight(name)
                        perf_weight = perf.weight

                        adjusted_confidence = signal.confidence * regime_weight * perf_weight
                        signal.confidence = min(1.0, adjusted_confidence)

                        signals.append((name, signal))
                        perf.signals_generated += 1

                except Exception as e:
                    logger.error(f"Error in strategy {name}: {e}")

        # Agregar sinais
        if signals:
            return self._aggregate_signals(signals, symbol, quote)

        return None

    def _update_market_data(self, quote: Quote) -> None:
        """Atualiza dados de mercado para regime detection"""
        mid = quote.mid_price
        if mid > 0:
            self._price_history.append(mid)

            # Calcular volatilidade
            if len(self._price_history) >= 20:
                prices = list(self._price_history)[-20:]
                returns = np.diff(prices) / np.array(prices[:-1])
                volatility = np.std(returns) * 100
                self._volatility_history.append(volatility)

    def _detect_regime(self) -> None:
        """Detecta regime de mercado atual"""
        if len(self._price_history) < 50:
            return

        prices = np.array(list(self._price_history))

        # Calcular indicadores
        # Trend: regressão linear
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        trend_strength = slope * len(prices) / np.mean(prices)

        # Volatilidade média
        avg_volatility = np.mean(list(self._volatility_history)) if self._volatility_history else 0

        # Range: amplitude relativa
        price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)

        # Classificar regime
        if avg_volatility > 0.05:  # Alta volatilidade
            self._regime = MarketRegime.VOLATILE
        elif avg_volatility < 0.01:  # Baixa volatilidade
            self._regime = MarketRegime.QUIET
        elif trend_strength > 0.002:  # Tendência de alta
            self._regime = MarketRegime.TRENDING_UP
        elif trend_strength < -0.002:  # Tendência de baixa
            self._regime = MarketRegime.TRENDING_DOWN
        else:
            self._regime = MarketRegime.RANGING

        self._regime_history.append(self._regime)

    def _get_regime_weight(self, strategy_name: str) -> float:
        """Obtém peso da estratégia para o regime atual"""
        weights = self._regime_weights.get(self._regime, {})
        return weights.get(strategy_name, 1.0)

    def _can_generate_signal(self) -> bool:
        """Verifica cooldown de sinais"""
        elapsed = (time.time() - self._last_signal_time) * 1000
        return elapsed >= self.signal_cooldown_ms

    # ==================== SIGNAL AGGREGATION ====================

    def _aggregate_signals(self, signals: List[Tuple[str, Signal]],
                          symbol: str, quote: Quote) -> Optional[AggregatedSignal]:
        """Agrega sinais de múltiplas estratégias"""
        if not signals:
            return None

        if self.aggregation_method == SignalAggregation.WEIGHTED_VOTE:
            return self._aggregate_weighted_vote(signals, symbol, quote)
        elif self.aggregation_method == SignalAggregation.HIGHEST_CONFIDENCE:
            return self._aggregate_highest_confidence(signals, symbol, quote)
        elif self.aggregation_method == SignalAggregation.UNANIMOUS:
            return self._aggregate_unanimous(signals, symbol, quote)
        elif self.aggregation_method == SignalAggregation.MAJORITY:
            return self._aggregate_majority(signals, symbol, quote)

        return None

    def _aggregate_weighted_vote(self, signals: List[Tuple[str, Signal]],
                                  symbol: str, quote: Quote) -> Optional[AggregatedSignal]:
        """Agregação por voto ponderado"""
        buy_score = 0.0
        sell_score = 0.0
        buy_signals = []
        sell_signals = []

        for name, signal in signals:
            weight = self._performance[name].weight if name in self._performance else 1.0

            if signal.signal_type == SignalType.BUY:
                buy_score += signal.confidence * weight
                buy_signals.append((name, signal))
            elif signal.signal_type == SignalType.SELL:
                sell_score += signal.confidence * weight
                sell_signals.append((name, signal))

        # Determinar direção vencedora
        if buy_score > sell_score and buy_score >= self.min_confidence:
            if len(buy_signals) >= self.min_strategies_agree:
                return self._create_aggregated_signal(buy_signals, SignalType.BUY, symbol, quote)

        elif sell_score > buy_score and sell_score >= self.min_confidence:
            if len(sell_signals) >= self.min_strategies_agree:
                return self._create_aggregated_signal(sell_signals, SignalType.SELL, symbol, quote)

        return None

    def _aggregate_highest_confidence(self, signals: List[Tuple[str, Signal]],
                                       symbol: str, quote: Quote) -> Optional[AggregatedSignal]:
        """Agregação pela maior confiança"""
        if not signals:
            return None

        # Encontrar sinal com maior confiança
        best_signal = max(signals, key=lambda x: x[1].confidence)
        name, signal = best_signal

        if signal.confidence >= self.min_confidence:
            return self._create_aggregated_signal([best_signal], signal.signal_type, symbol, quote)

        return None

    def _aggregate_unanimous(self, signals: List[Tuple[str, Signal]],
                             symbol: str, quote: Quote) -> Optional[AggregatedSignal]:
        """Agregação unânime - todos devem concordar"""
        if len(signals) < 2:
            return None

        # Verificar se todos concordam
        directions = set()
        for name, signal in signals:
            if signal.signal_type in (SignalType.BUY, SignalType.SELL):
                directions.add(signal.signal_type)

        if len(directions) == 1:
            signal_type = directions.pop()
            return self._create_aggregated_signal(signals, signal_type, symbol, quote)

        return None

    def _aggregate_majority(self, signals: List[Tuple[str, Signal]],
                            symbol: str, quote: Quote) -> Optional[AggregatedSignal]:
        """Agregação por maioria"""
        buy_count = sum(1 for _, s in signals if s.signal_type == SignalType.BUY)
        sell_count = sum(1 for _, s in signals if s.signal_type == SignalType.SELL)

        total = buy_count + sell_count
        if total == 0:
            return None

        if buy_count > sell_count and buy_count / total > 0.5:
            buy_signals = [(n, s) for n, s in signals if s.signal_type == SignalType.BUY]
            return self._create_aggregated_signal(buy_signals, SignalType.BUY, symbol, quote)

        elif sell_count > buy_count and sell_count / total > 0.5:
            sell_signals = [(n, s) for n, s in signals if s.signal_type == SignalType.SELL]
            return self._create_aggregated_signal(sell_signals, SignalType.SELL, symbol, quote)

        return None

    def _create_aggregated_signal(self, signals: List[Tuple[str, Signal]],
                                   signal_type: SignalType, symbol: str,
                                   quote: Quote) -> AggregatedSignal:
        """Cria sinal agregado"""
        # Calcular confiança média ponderada
        total_weight = 0.0
        weighted_confidence = 0.0
        stop_losses = []
        take_profits = []
        contributing = []

        for name, signal in signals:
            weight = self._performance[name].weight if name in self._performance else 1.0
            weighted_confidence += signal.confidence * weight
            total_weight += weight
            contributing.append(name)

            if signal.stop_loss:
                stop_losses.append(signal.stop_loss)
            if signal.take_profit:
                take_profits.append(signal.take_profit)

        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5

        # Usar SL/TP mais conservador
        stop_loss = None
        take_profit = None

        if stop_losses:
            if signal_type == SignalType.BUY:
                stop_loss = max(stop_losses)  # SL mais alto (mais conservador para long)
            else:
                stop_loss = min(stop_losses)  # SL mais baixo (mais conservador para short)

        if take_profits:
            if signal_type == SignalType.BUY:
                take_profit = min(take_profits)  # TP mais baixo (mais conservador)
            else:
                take_profit = max(take_profits)

        # Determinar força
        if avg_confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif avg_confidence >= 0.6:
            strength = SignalStrength.STRONG
        elif avg_confidence >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Preço de entrada
        price = quote.ask_price if signal_type == SignalType.BUY else quote.bid_price

        self._last_signal_time = time.time()

        aggregated = AggregatedSignal(
            signal_type=signal_type,
            symbol=symbol,
            price=price,
            strength=strength,
            confidence=avg_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contributing_strategies=contributing,
            aggregation_method=self.aggregation_method,
            regime=self._regime,
            metadata={
                'strategy_count': len(signals),
                'total_weight': total_weight,
                'regime': self._regime.value
            }
        )

        # Registrar
        self._signal_history.append(aggregated)

        # Emitir para callbacks
        for cb in self._signal_callbacks:
            try:
                cb(aggregated)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

        return aggregated

    # ==================== PERFORMANCE TRACKING ====================

    def record_trade_result(self, strategy_name: str, pnl: float, is_win: bool) -> None:
        """Registra resultado de trade para uma estratégia"""
        with self._lock:
            perf = self._performance.get(strategy_name)
            if not perf:
                return

            perf.trades_executed += 1
            perf.total_pnl += pnl

            if is_win:
                perf.wins += 1
            else:
                perf.losses += 1

            # Atualizar métricas
            if perf.trades_executed > 0:
                perf.avg_pnl = perf.total_pnl / perf.trades_executed
                perf.win_rate = perf.wins / perf.trades_executed

            # Profit factor
            wins_pnl = sum(1 for t in range(perf.wins))  # Simplificado
            losses_pnl = abs(perf.total_pnl - perf.avg_pnl * perf.wins) if perf.losses > 0 else 1
            perf.profit_factor = wins_pnl / losses_pnl if losses_pnl > 0 else 1.0

            # Atualizar peso baseado em performance
            self._update_strategy_weight(strategy_name)

    def _update_strategy_weight(self, strategy_name: str) -> None:
        """Atualiza peso da estratégia baseado em performance"""
        perf = self._performance.get(strategy_name)
        if not perf or perf.trades_executed < 10:
            return

        # Calcular score de performance
        score = 0.0

        # Win rate contribui
        score += perf.win_rate * 0.3

        # Profit factor contribui
        pf_score = min(2.0, perf.profit_factor) / 2.0
        score += pf_score * 0.4

        # P&L médio contribui
        if perf.avg_pnl > 0:
            score += 0.3
        elif perf.avg_pnl < 0:
            score -= 0.2

        # Ajustar peso (entre 0.2 e 2.0)
        perf.weight = max(0.2, min(2.0, 0.5 + score))

        logger.debug(f"Strategy {strategy_name} weight updated to {perf.weight:.2f}")

    # ==================== STRATEGY CONTROL ====================

    def enable_strategy(self, name: str) -> bool:
        """Habilita estratégia"""
        with self._lock:
            if name in self._strategies:
                self._strategies[name].enable()
                if name in self._performance:
                    self._performance[name].enabled = True
                return True
        return False

    def disable_strategy(self, name: str) -> bool:
        """Desabilita estratégia"""
        with self._lock:
            if name in self._strategies:
                self._strategies[name].disable()
                if name in self._performance:
                    self._performance[name].enabled = False
                return True
        return False

    def set_strategy_weight(self, name: str, weight: float) -> bool:
        """Define peso manualmente"""
        with self._lock:
            if name in self._performance:
                self._performance[name].weight = max(0.1, min(3.0, weight))
                return True
        return False

    def update_position(self, position: Position) -> None:
        """Atualiza posição em todas as estratégias"""
        with self._lock:
            for strategy in self._strategies.values():
                strategy.on_position_update(position)

    # ==================== PROPERTIES ====================

    @property
    def regime(self) -> MarketRegime:
        """Regime de mercado atual"""
        return self._regime

    @property
    def active_strategies(self) -> List[str]:
        """Lista de estratégias ativas"""
        return [
            name for name, perf in self._performance.items()
            if perf.enabled
        ]

    @property
    def stats(self) -> dict:
        """Estatísticas do orquestrador"""
        return {
            'regime': self._regime.value,
            'aggregation_method': self.aggregation_method.value,
            'active_strategies': len(self.active_strategies),
            'total_strategies': len(self._strategies),
            'signals_generated': len(self._signal_history),
            'strategy_performance': {
                name: {
                    'weight': perf.weight,
                    'signals': perf.signals_generated,
                    'trades': perf.trades_executed,
                    'win_rate': perf.win_rate,
                    'pnl': perf.total_pnl,
                    'enabled': perf.enabled
                }
                for name, perf in self._performance.items()
            }
        }

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Obtém estratégia por nome"""
        return self._strategies.get(name)

    def list_strategies(self) -> List[str]:
        """Lista nomes das estratégias"""
        return list(self._strategies.keys())
