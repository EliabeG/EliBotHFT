"""
Strategy Module - Motor de estratégias de trading
"""

from .strategy_engine import StrategyEngine
from .base_strategy import BaseStrategy, Signal, SignalType
from .scalping_strategy import ScalpingStrategy
from .momentum_strategy import MomentumStrategy

__all__ = [
    'StrategyEngine',
    'BaseStrategy',
    'Signal',
    'SignalType',
    'ScalpingStrategy',
    'MomentumStrategy'
]
