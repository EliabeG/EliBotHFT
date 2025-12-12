"""
EliBotHFT Machine Learning Module - Enhanced Edition

A comprehensive ML system for learning from trading errors and continuously
improving trading performance through:

Core Components:
1. Error Classification - Categorize and analyze trading errors
2. Feature Extraction - Extract relevant features from market data and trades
3. Pattern Analysis - Identify patterns in error occurrences
4. Learning Engine - Train models to predict and prevent errors
5. Adaptive Optimization - Dynamically adjust strategy parameters

Advanced Components:
6. Advanced Neural Networks - Deep, Recurrent, Attention-based architectures
7. Ensemble Methods - Bagging, Boosting, Stacking, Weighted ensembles
8. Online Learning - Incremental learning with concept drift detection
9. Advanced Features - Technical indicators, microstructure, volatility metrics
10. Performance Analytics - Comprehensive trading performance analysis
11. Self-Optimization - Hyperparameter tuning, neural architecture search
12. Real-time Predictions - Low-latency prediction pipeline

This module implements a feedback loop that:
- Monitors all trades and their outcomes
- Classifies errors by type (slippage, timing, prediction, etc.)
- Learns patterns that precede errors
- Adapts to changing market conditions using drift detection
- Auto-optimizes hyperparameters and model architecture
- Provides real-time risk predictions
- Adjusts strategy weights and parameters to minimize future errors
"""

# ==================== Core Components ====================

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
    OptimizationResult as BaseOptimizationResult,
)
from .ml_manager import (
    MLManager,
    MLConfig,
    TradeAnalysis,
)

# ==================== Advanced Neural Networks ====================

from .advanced_networks import (
    DeepNetwork,
    NetworkConfig,
    RecurrentNetwork,
    TransformerBlock,
    EnsembleNetwork,
    AdaptiveNetwork,
    AttentionLayer,
    LSTMCell,
    GRUCell,
    ResidualBlock,
    BatchNormalization,
    DropoutLayer,
    ActivationFunction,
    apply_activation,
)

# ==================== Ensemble Methods ====================

from .ensemble_methods import (
    BaggingEnsemble,
    AdaBoostEnsemble,
    GradientBoostingEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
    DiversityEnsemble,
    EnsembleSelector,
    SimpleNeuralNet,
    AggregationMethod,
)

# ==================== Online Learning & Drift Detection ====================

from .online_learning import (
    ADWIN,
    DDM,
    EDDM,
    PageHinkley,
    MultiDriftDetector,
    OnlineLearner,
    IncrementalSGD,
    PassiveAggressiveLearner,
    AdaptiveWindowManager,
    StreamProcessor,
    ForgetfulLearner,
    DriftType,
    DriftDetectionResult,
)

# ==================== Advanced Feature Engineering ====================

from .advanced_features import (
    AdvancedFeatureExtractor,
    FeatureSet,
    FeatureCategory,
    TechnicalIndicators,
    OrderBookFeatures,
    MicrostructureFeatures,
    VolatilityFeatures,
    MomentumFeatures,
    TemporalFeatures,
    StatisticalFeatures,
)

# ==================== Performance Analytics ====================

from .performance_analytics import (
    PerformanceAnalyzer,
    PerformanceReport,
    TradeRecord,
    ReturnMetrics,
    RiskMetrics,
    RiskAdjustedMetrics,
    TradingMetrics,
    ModelMetrics,
    RealTimeMonitor,
    MetricType,
)

# ==================== Self-Optimization ====================

from .self_optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
    ParticleSwarmOptimizer,
    FeatureSelector,
    NeuralArchitectureSearch,
    OnlineHyperparameterTuner,
    SelfOptimizingSystem,
    HyperParameter,
    OptimizationMethod,
    OptimizationResult,
)

# ==================== Real-time Prediction Pipeline ====================

from .realtime_predictor import (
    RealtimePredictor,
    FeaturePipeline,
    PredictionCache,
    ConfidenceCalibrator,
    ModelServer,
    LatencyMonitor,
    PredictionQueue,
    PredictionType,
    PredictionRequest,
    PredictionResponse,
    MultiModelRouter,
    EnsemblePredictor,
)

# ==================== Enhanced ML Manager ====================

from .enhanced_ml_manager import (
    EnhancedMLManager,
    EnhancedMLConfig,
)

# ==================== Exports ====================

__all__ = [
    # Core - Error Classification
    'ErrorType',
    'ErrorSeverity',
    'TradeError',
    'ErrorClassifier',

    # Core - Feature Extraction
    'FeatureExtractor',
    'TradeFeatures',
    'MarketFeatures',

    # Core - Pattern Analysis
    'PatternAnalyzer',
    'ErrorPattern',
    'PatternMatch',

    # Core - Learning Engine
    'LearningEngine',
    'LearningConfig',
    'ModelType',

    # Core - Model Store
    'ModelStore',
    'ModelMetadata',

    # Core - Adaptive Optimizer
    'AdaptiveOptimizer',
    'BaseOptimizationResult',

    # Core - ML Manager
    'MLManager',
    'MLConfig',
    'TradeAnalysis',

    # Advanced Networks
    'DeepNetwork',
    'NetworkConfig',
    'RecurrentNetwork',
    'TransformerBlock',
    'EnsembleNetwork',
    'AdaptiveNetwork',
    'AttentionLayer',
    'LSTMCell',
    'GRUCell',
    'ResidualBlock',
    'BatchNormalization',
    'DropoutLayer',
    'ActivationFunction',
    'apply_activation',

    # Ensemble Methods
    'BaggingEnsemble',
    'AdaBoostEnsemble',
    'GradientBoostingEnsemble',
    'StackingEnsemble',
    'WeightedEnsemble',
    'DiversityEnsemble',
    'EnsembleSelector',
    'SimpleNeuralNet',
    'AggregationMethod',

    # Online Learning
    'ADWIN',
    'DDM',
    'EDDM',
    'PageHinkley',
    'MultiDriftDetector',
    'OnlineLearner',
    'IncrementalSGD',
    'PassiveAggressiveLearner',
    'AdaptiveWindowManager',
    'StreamProcessor',
    'ForgetfulLearner',
    'DriftType',
    'DriftDetectionResult',

    # Advanced Features
    'AdvancedFeatureExtractor',
    'FeatureSet',
    'FeatureCategory',
    'TechnicalIndicators',
    'OrderBookFeatures',
    'MicrostructureFeatures',
    'VolatilityFeatures',
    'MomentumFeatures',
    'TemporalFeatures',
    'StatisticalFeatures',

    # Performance Analytics
    'PerformanceAnalyzer',
    'PerformanceReport',
    'TradeRecord',
    'ReturnMetrics',
    'RiskMetrics',
    'RiskAdjustedMetrics',
    'TradingMetrics',
    'ModelMetrics',
    'RealTimeMonitor',
    'MetricType',

    # Self-Optimization
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'ParticleSwarmOptimizer',
    'FeatureSelector',
    'NeuralArchitectureSearch',
    'OnlineHyperparameterTuner',
    'SelfOptimizingSystem',
    'HyperParameter',
    'OptimizationMethod',
    'OptimizationResult',

    # Real-time Predictions
    'RealtimePredictor',
    'FeaturePipeline',
    'PredictionCache',
    'ConfidenceCalibrator',
    'ModelServer',
    'LatencyMonitor',
    'PredictionQueue',
    'PredictionType',
    'PredictionRequest',
    'PredictionResponse',
    'MultiModelRouter',
    'EnsemblePredictor',

    # Enhanced ML Manager
    'EnhancedMLManager',
    'EnhancedMLConfig',
]

__version__ = '2.0.0'
