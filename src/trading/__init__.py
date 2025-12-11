"""
EliBotHFT Trading Module
Lógica de mercado, estratégias e gestão de ordens
"""

from .book import OrderBook, OrderBookLevel, OrderBookUpdate
from .strategy import StrategyEngine, BaseStrategy, Signal, SignalType
from .oms import OrderManagementSystem, Order, OrderStatus
from .risk import RiskManager, RiskLimits, RiskCheck

__all__ = [
    'OrderBook',
    'OrderBookLevel',
    'OrderBookUpdate',
    'StrategyEngine',
    'BaseStrategy',
    'Signal',
    'SignalType',
    'OrderManagementSystem',
    'Order',
    'OrderStatus',
    'RiskManager',
    'RiskLimits',
    'RiskCheck'
]

__version__ = '1.0.0'
