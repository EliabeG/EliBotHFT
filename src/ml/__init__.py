"""
EliBotHFT Machine Learning Module

A comprehensive ML system for learning from trading errors and continuously
improving trading performance through:

1. Error Classification - Categorize and analyze trading errors
2. Feature Extraction - Extract relevant features from market data and trades
3. Pattern Analysis - Identify patterns in error occurrences
4. Learning Engine - Train models to predict and prevent errors
5. Adaptive Optimization - Dynamically adjust strategy parameters

This module implements a feedback loop that:
- Monitors all trades and their outcomes
- Classifies errors by type (slippage, timing, prediction, etc.)
- Learns patterns that precede errors
- Adjusts strategy weights and parameters to minimize future errors
"""

from .error_classifier import (
    ErrorType,
    ErrorSeverity,
    TradeError,
    ErrorClassifier,
)
from .feature_extractor import (
    FeatureExtractor,
    TradeFeatures,
    MarketFeatures,
)
from .pattern_analyzer import (
    PatternAnalyzer,
    ErrorPattern,
    PatternMatch,
)
from .learning_engine import (
    LearningEngine,
    LearningConfig,
    ModelType,
)
from .model_store import (
    ModelStore,
    ModelMetadata,
)
from .adaptive_optimizer import (
    AdaptiveOptimizer,
    OptimizationResult,
)
from .ml_manager import (
    MLManager,
    MLConfig,
)

__all__ = [
    # Error Classification
    'ErrorType',
    'ErrorSeverity',
    'TradeError',
    'ErrorClassifier',
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
    # Manager
    'MLManager',
    'MLConfig',
]

__version__ = '1.0.0'
