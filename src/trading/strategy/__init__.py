"""
Strategy Module - Motor de estratégias de trading HFT
Inclui múltiplas estratégias otimizadas para EURUSD
"""

from .strategy_engine import StrategyEngine
from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from .scalping_strategy import ScalpingStrategy
from .momentum_strategy import MomentumStrategy
from .market_making_strategy import MarketMakingStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .order_flow_strategy import OrderFlowStrategy
from .strategy_orchestrator import StrategyOrchestrator, MarketRegime, SignalAggregation

__all__ = [
    # Engine
    'StrategyEngine',
    'StrategyOrchestrator',

    # Base
    'BaseStrategy',
    'Signal',
    'SignalType',
    'SignalStrength',
    'MarketState',
    'Position',

    # Strategies
    'ScalpingStrategy',
    'MomentumStrategy',
    'MarketMakingStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'OrderFlowStrategy',

    # Enums
    'MarketRegime',
    'SignalAggregation'
]
