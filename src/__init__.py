"""
EliBotHFT - High Frequency Trading Robot for Forex
Desenvolvido para FXOpen TickTrader Platform

Este pacote contém:
- core: Motor de baixa latência
- trading: Lógica de mercado e estratégias
- bindings: API Python e cliente FXOpen
- ml: Sistema de Machine Learning para aprendizado com erros
"""

__version__ = '2.0.0'
__author__ = 'EliBotHFT'

from .bindings import EliBotAPI, FXOpenClient, FXOpenConfig
from .trading import (
    OrderBook, OrderManagementSystem, StrategyEngine,
    RiskManager, Signal, SignalType
)
from .ml import (
    MLManager, MLConfig,
    ErrorClassifier, TradeError, ErrorType, ErrorSeverity,
    FeatureExtractor, TradeFeatures, MarketFeatures,
    PatternAnalyzer, ErrorPattern, PatternMatch,
    LearningEngine, LearningConfig, ModelType,
    ModelStore, ModelMetadata,
    AdaptiveOptimizer, OptimizationResult
)

__all__ = [
    # Bindings
    'EliBotAPI',
    'FXOpenClient',
    'FXOpenConfig',
    # Trading
    'OrderBook',
    'OrderManagementSystem',
    'StrategyEngine',
    'RiskManager',
    'Signal',
    'SignalType',
    # ML Manager
    'MLManager',
    'MLConfig',
    # Error Classification
    'ErrorClassifier',
    'TradeError',
    'ErrorType',
    'ErrorSeverity',
    # Feature Extraction
    'FeatureExtractor',
    'TradeFeatures',
    'MarketFeatures',
    # Pattern Analysis
    'PatternAnalyzer',
    'ErrorPattern',
    'PatternMatch',
    # Learning Engine
    'LearningEngine',
    'LearningConfig',
    'ModelType',
    # Model Store
    'ModelStore',
    'ModelMetadata',
    # Adaptive Optimizer
    'AdaptiveOptimizer',
    'OptimizationResult',
]
