"""
EliBotHFT - High Frequency Trading Robot for Forex
Desenvolvido para FXOpen TickTrader Platform

Este pacote contém:
- core: Motor de baixa latência
- trading: Lógica de mercado e estratégias
- bindings: API Python e cliente FXOpen
"""

__version__ = '1.0.0'
__author__ = 'EliBotHFT'

from .bindings import EliBotAPI, FXOpenClient, FXOpenConfig
from .trading import (
    OrderBook, OrderManagementSystem, StrategyEngine,
    RiskManager, Signal, SignalType
)

__all__ = [
    'EliBotAPI',
    'FXOpenClient',
    'FXOpenConfig',
    'OrderBook',
    'OrderManagementSystem',
    'StrategyEngine',
    'RiskManager',
    'Signal',
    'SignalType'
]
