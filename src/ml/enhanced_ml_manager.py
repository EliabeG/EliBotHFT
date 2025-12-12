"""
Enhanced ML Manager - Integration of All Advanced ML Components

This module provides a unified interface for all ML capabilities:
1. Advanced Neural Networks (LSTM, Attention, Transformers)
2. Ensemble Methods (Bagging, Boosting, Stacking)
3. Online Learning with Drift Detection
4. Advanced Feature Engineering
5. Performance Analytics
6. Self-Optimization (Hyperparameter tuning, NAS)
7. Real-time Predictions

This is the main entry point for the enhanced ML system.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from pathlib import Path
import asyncio
import json
import time

# Core ML components
from .error_classifier import ErrorClassifier, TradeError, ErrorType, ErrorSeverity
from .feature_extractor import FeatureExtractor
from .pattern_analyzer import PatternAnalyzer, PatternMatch
from .learning_engine import LearningEngine, LearningConfig, ModelType
from .model_store import ModelStore
from .adaptive_optimizer import AdaptiveOptimizer
from .ml_manager import MLManager, MLConfig, TradeAnalysis

# Advanced components
from .advanced_networks import (
    DeepNetwork, NetworkConfig, RecurrentNetwork,
    TransformerBlock, EnsembleNetwork, AdaptiveNetwork,
    AttentionLayer, LSTMCell, GRUCell, ActivationFunction
)
from .ensemble_methods import (
    BaggingEnsemble, AdaBoostEnsemble, GradientBoostingEnsemble,
    StackingEnsemble, WeightedEnsemble, DiversityEnsemble,
    EnsembleSelector, SimpleNeuralNet
)
from .online_learning import (
    ADWIN, DDM, EDDM, PageHinkley, MultiDriftDetector,
    OnlineLearner, IncrementalSGD, PassiveAggressiveLearner,
    AdaptiveWindowManager, StreamProcessor, ForgetfulLearner,
    DriftType, DriftDetectionResult
)
from .advanced_features import (
    AdvancedFeatureExtractor, TechnicalIndicators,
    OrderBookFeatures, MicrostructureFeatures,
    VolatilityFeatures, MomentumFeatures,
    TemporalFeatures, StatisticalFeatures, FeatureSet
)
from .performance_analytics import (
    PerformanceAnalyzer, PerformanceReport, TradeRecord,
    ReturnMetrics, RiskMetrics, RiskAdjustedMetrics,
    TradingMetrics, ModelMetrics, RealTimeMonitor
)
from .self_optimization import (
    BayesianOptimizer, GeneticOptimizer, ParticleSwarmOptimizer,
    FeatureSelector, NeuralArchitectureSearch,
    OnlineHyperparameterTuner, SelfOptimizingSystem,
    HyperParameter, OptimizationMethod, OptimizationResult
)
from .realtime_predictor import (
    RealtimePredictor, FeaturePipeline, PredictionCache,
    ModelServer, LatencyMonitor, ConfidenceCalibrator,
    PredictionType, PredictionRequest, PredictionResponse,
    MultiModelRouter, EnsemblePredictor
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMLConfig:
    """Configuration for Enhanced ML System"""

    # Base ML Config
    pip_size: float = 0.0001
    history_length: int = 1000

    # Neural Network
    network_type: str = 'deep'  # 'deep', 'recurrent', 'attention'
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    learning_rate: float = 0.001

    # Ensemble
    use_ensemble: bool = True
    ensemble_type: str = 'weighted'  # 'bagging', 'boosting', 'stacking', 'weighted'
    n_estimators: int = 5

    # Online Learning
    enable_drift_detection: bool = True
    drift_detector_type: str = 'multi'  # 'adwin', 'ddm', 'eddm', 'multi'
    window_size: int = 1000

    # Feature Engineering
    use_advanced_features: bool = True
    feature_selection: bool = True
    n_features_to_select: Optional[int] = None

    # Self-Optimization
    enable_auto_optimization: bool = True
    optimization_method: str = 'bayesian'  # 'grid', 'random', 'bayesian', 'genetic', 'pso'
    optimization_budget: int = 50

    # Real-time Predictions
    enable_caching: bool = True
    cache_ttl_ms: float = 100.0
    enable_async: bool = False

    # Persistence
    model_path: str = 'data/ml_models'
    state_file: str = 'data/enhanced_ml_state.json'
    auto_save_interval: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class EnhancedMLManager:
    """
    Enhanced ML Manager integrating all advanced components.
    """

    def __init__(
        self,
        config: Optional[EnhancedMLConfig] = None,
        strategy_names: Optional[List[str]] = None
    ):
        self.config = config or EnhancedMLConfig()
        self.strategy_names = strategy_names or [
            'scalping', 'momentum', 'mean_reversion',
            'market_making', 'breakout', 'order_flow'
        ]

        # Initialize all components
        self._init_components()

        # State tracking
        self.trade_count = 0
        self.error_count = 0
        self.prediction_count = 0
        self.optimization_count = 0
        self.drift_count = 0

        # Callbacks
        self.on_weight_update: Optional[Callable[[Dict[str, float]], None]] = None
        self.on_risk_alert: Optional[Callable[[str, float], None]] = None
        self.on_drift_detected: Optional[Callable[[DriftDetectionResult], None]] = None
        self.on_optimization_complete: Optional[Callable[[OptimizationResult], None]] = None

        # Running state
        self._running = False
        self._background_task: Optional[asyncio.Task] = None

        logger.info("EnhancedMLManager initialized with full feature set")

    def _init_components(self):
        """Initialize all ML components"""

        # ========== Core Components ==========

        # Base ML Manager
        base_config = MLConfig(
            pip_size=self.config.pip_size,
            history_length=self.config.history_length,
            learning_rate=self.config.learning_rate,
            model_store_path=self.config.model_path
        )
        self.base_ml = MLManager(config=base_config, strategy_names=self.strategy_names)

        # ========== Advanced Feature Engineering ==========

        self.advanced_features = AdvancedFeatureExtractor(
            price_history_length=self.config.history_length,
            pip_size=self.config.pip_size
        )

        # Feature pipeline for real-time preprocessing
        self.feature_pipeline = FeaturePipeline(
            feature_names=[],  # Will be populated on first extraction
            normalization='standard'
        )

        # Feature selector
        self.feature_selector = FeatureSelector(
            method='importance',
            n_features=self.config.n_features_to_select
        )

        # ========== Neural Networks ==========

        # Determine input/output sizes (will be updated after first data)
        input_size = 50  # Default, will be updated
        output_size = len(ErrorType)  # One for each error type

        network_config = NetworkConfig(
            input_size=input_size,
            hidden_sizes=self.config.hidden_sizes,
            output_size=output_size,
            dropout_rate=self.config.dropout_rate,
            use_batch_norm=self.config.use_batch_norm,
            learning_rate=self.config.learning_rate
        )

        if self.config.network_type == 'deep':
            self.primary_network = DeepNetwork(network_config)
        elif self.config.network_type == 'recurrent':
            self.primary_network = RecurrentNetwork(
                input_size, self.config.hidden_sizes[0],
                output_size, num_layers=2
            )
        else:  # attention
            self.primary_network = DeepNetwork(network_config)

        # Adaptive network for drift handling
        self.adaptive_network = AdaptiveNetwork(network_config)

        # ========== Ensemble Methods ==========

        if self.config.use_ensemble:
            # Create base models for ensemble
            def create_base_model():
                return SimpleNeuralNet(
                    input_size, self.config.hidden_sizes, output_size,
                    learning_rate=self.config.learning_rate
                )

            if self.config.ensemble_type == 'bagging':
                self.ensemble = BaggingEnsemble(
                    create_base_model,
                    n_estimators=self.config.n_estimators
                )
            elif self.config.ensemble_type == 'boosting':
                self.ensemble = GradientBoostingEnsemble(
                    n_estimators=self.config.n_estimators
                )
            elif self.config.ensemble_type == 'stacking':
                base_models = [create_base_model() for _ in range(self.config.n_estimators)]
                meta_model = SimpleNeuralNet(
                    self.config.n_estimators * output_size,
                    [32], output_size
                )
                self.ensemble = StackingEnsemble(base_models, meta_model)
            else:  # weighted
                models = [create_base_model() for _ in range(self.config.n_estimators)]
                self.ensemble = WeightedEnsemble(models)
        else:
            self.ensemble = None

        # ========== Online Learning & Drift Detection ==========

        if self.config.enable_drift_detection:
            if self.config.drift_detector_type == 'adwin':
                self.drift_detector = ADWIN()
            elif self.config.drift_detector_type == 'ddm':
                self.drift_detector = DDM()
            elif self.config.drift_detector_type == 'eddm':
                self.drift_detector = EDDM()
            else:  # multi
                self.drift_detector = MultiDriftDetector()
        else:
            self.drift_detector = None

        # Incremental learner
        self.incremental_learner = IncrementalSGD(
            input_size, output_size,
            learning_rate=self.config.learning_rate
        )

        # Online learner wrapper
        self.online_learner = OnlineLearner(
            self.incremental_learner,
            self.drift_detector,
            window_size=self.config.window_size
        )

        # Adaptive window manager
        self.window_manager = AdaptiveWindowManager(
            initial_size=self.config.window_size
        )

        # ========== Self-Optimization ==========

        if self.config.enable_auto_optimization:
            # Define hyperparameter space
            self.param_space = [
                HyperParameter('learning_rate', 'float', 0.0001, 0.1, log_scale=True),
                HyperParameter('dropout_rate', 'float', 0.0, 0.5),
                HyperParameter('hidden_size_1', 'int', 32, 256),
                HyperParameter('hidden_size_2', 'int', 16, 128),
            ]

            # Select optimizer
            if self.config.optimization_method == 'bayesian':
                self.optimizer = BayesianOptimizer()
            elif self.config.optimization_method == 'genetic':
                self.optimizer = GeneticOptimizer()
            elif self.config.optimization_method == 'pso':
                self.optimizer = ParticleSwarmOptimizer()
            else:
                from .self_optimization import RandomSearchOptimizer
                self.optimizer = RandomSearchOptimizer()

            # Online hyperparameter tuner
            self.online_tuner = OnlineHyperparameterTuner(
                self.param_space,
                window_size=self.config.window_size
            )

            # Neural architecture search
            self.nas = NeuralArchitectureSearch(
                input_size, output_size,
                min_layers=1, max_layers=4
            )
        else:
            self.optimizer = None
            self.online_tuner = None
            self.nas = None

        # ========== Performance Analytics ==========

        self.performance_analyzer = PerformanceAnalyzer()
        self.realtime_monitor = RealTimeMonitor(self.performance_analyzer)

        # ========== Real-time Predictions ==========

        self.realtime_predictor = RealtimePredictor(
            feature_pipeline=self.feature_pipeline,
            cache_enabled=self.config.enable_caching,
            cache_ttl_ms=self.config.cache_ttl_ms,
            async_enabled=self.config.enable_async
        )

        # Register models with predictor
        self.realtime_predictor.register_model(
            self.primary_network, 'primary', '1.0',
            [PredictionType.ERROR_PROBABILITY, PredictionType.ERROR_TYPE]
        )

        # Model server for versioning
        self.model_server = ModelServer()

        # Latency monitor
        self.latency_monitor = LatencyMonitor()

    def update_market_data(
        self,
        price: float,
        volume: float = 0.0,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        order_book: Optional[Dict] = None
    ):
        """Update with new market data"""
        # Update base ML
        self.base_ml.update_market_data(
            price, volume, bid_price, ask_price, order_book
        )

        # Update advanced features
        self.advanced_features.update(
            price, volume, high, low, open_price=price
        )

    def analyze_trade(
        self,
        trade_id: str,
        strategy_name: str,
        signal_type: str,
        entry_price: float,
        signal_confidence: float,
        signal_strength: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        bid_price: float,
        ask_price: float,
        market_regime: str,
        risk_level: str,
        symbol: str = "EURUSD"
    ) -> Dict[str, Any]:
        """
        Enhanced trade analysis combining all ML capabilities.

        Returns comprehensive analysis including:
        - Error risk prediction
        - Pattern matches
        - Strategy recommendations
        - Real-time ensemble predictions
        """
        start_time = time.time()

        # Get base analysis
        base_analysis = self.base_ml.analyze_trade(
            trade_id, strategy_name, signal_type,
            entry_price, entry_price, signal_confidence,
            signal_strength, stop_loss, take_profit,
            position_size, bid_price, ask_price,
            market_regime, risk_level, symbol
        )

        # Extract advanced features
        order_book = {'bids': [(bid_price, 1.0)], 'asks': [(ask_price, 1.0)]}
        feature_set = self.advanced_features.extract_all_features(order_book)
        feature_vector = feature_set.to_vector()

        # Ensure feature dimensions match
        if len(feature_vector) > 0 and not self.feature_pipeline.is_fitted:
            # Initialize feature pipeline with current features
            self.feature_pipeline.feature_names = self.advanced_features.feature_names
            self.feature_pipeline.fit(feature_vector.reshape(1, -1))

        # Real-time prediction
        error_prediction = None
        if len(feature_vector) > 0:
            try:
                error_prediction = self.realtime_predictor.predict(
                    feature_vector,
                    PredictionType.ERROR_PROBABILITY
                )
            except Exception as e:
                logger.debug(f"Real-time prediction failed: {e}")

        # Ensemble prediction (if enabled)
        ensemble_prediction = None
        if self.ensemble is not None and self.online_learner.is_trained:
            try:
                processed = self.feature_pipeline.transform(feature_vector.reshape(1, -1))
                ensemble_prediction = self.ensemble.predict_proba(processed)[0]
            except Exception as e:
                logger.debug(f"Ensemble prediction failed: {e}")

        # Combine predictions
        combined_error_risk = base_analysis.error_risk
        if error_prediction is not None:
            combined_error_risk = 0.5 * combined_error_risk + 0.5 * error_prediction.confidence
        if ensemble_prediction is not None:
            combined_error_risk = 0.7 * combined_error_risk + 0.3 * np.max(ensemble_prediction)

        # Enhanced recommendations
        recommendations = list(base_analysis.recommendations)

        # Add drift-based recommendations
        if self.drift_detector is not None and self.drift_count > 0:
            recommendations.append("Market regime may be changing - reduce position size")

        # Add optimization-based recommendations
        if self.online_tuner is not None:
            params = self.online_tuner.get_params()
            if params.get('learning_rate', 0.01) > 0.05:
                recommendations.append("Model adapting rapidly - market uncertainty high")

        analysis_time = (time.time() - start_time) * 1000
        self.latency_monitor.record(analysis_time, "trade_analysis")

        return {
            'base_analysis': base_analysis.to_dict(),
            'combined_error_risk': combined_error_risk,
            'should_trade': combined_error_risk < 0.7,
            'advanced_features': {
                'hurst_exponent': feature_set.features.get('hurst', 0.5),
                'trend_strength': feature_set.features.get('trend_strength', 0),
                'volatility_regime': feature_set.features.get('realized_vol', 0),
            },
            'ensemble_prediction': ensemble_prediction.tolist() if ensemble_prediction is not None else None,
            'recommendations': recommendations[:5],
            'analysis_time_ms': analysis_time,
            'drift_detected': self.drift_count > 0,
            'model_version': self.realtime_predictor.model_server.active_versions.get('primary', 'unknown')
        }

    def record_trade_result(
        self,
        trade_id: str,
        strategy_name: str,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        holding_time_ms: int,
        market_regime: str,
        signal_confidence: float = 0.7,
        symbol: str = "EURUSD"
    ) -> Dict[str, Any]:
        """
        Record trade result and update all ML components.
        """
        self.trade_count += 1

        # Record in base ML
        error = self.base_ml.learn_from_trade(
            trade_id, symbol, strategy_name, signal_type,
            entry_price, exit_price, pnl, holding_time_ms,
            market_regime, signal_confidence
        )

        if error:
            self.error_count += 1

        # Record in performance analyzer
        trade_record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy_name,
            side='long' if signal_type in ['BUY', 'CLOSE_SHORT'] else 'short',
            entry_price=entry_price,
            exit_price=exit_price,
            size=1.0,
            pnl=pnl,
            pnl_pct=pnl / entry_price,
            entry_time=time.time() - holding_time_ms / 1000,
            exit_time=time.time(),
            holding_time_ms=holding_time_ms,
            market_regime=market_regime,
            signal_confidence=signal_confidence,
            was_error=error is not None,
            error_type=error.error_type.name if error else None
        )
        self.performance_analyzer.add_trade(trade_record)

        # Update online learning
        feature_set = self.advanced_features.extract_all_features()
        feature_vector = feature_set.to_vector()

        if len(feature_vector) > 0:
            # Create label
            n_classes = len(ErrorType)
            label = np.zeros(n_classes)
            if error:
                label[error.error_type.value - 1] = 1.0
            else:
                label[0] = 1.0  # No error class

            # Update online learner
            try:
                update_result = self.online_learner.partial_fit(
                    feature_vector.reshape(1, -1),
                    label.reshape(1, -1)
                )

                # Check for drift
                if update_result.get('drift_detected', False):
                    self.drift_count += 1
                    if self.on_drift_detected:
                        drift_result = DriftDetectionResult(
                            drift_detected=True,
                            drift_type=DriftType.SUDDEN,
                            confidence=0.8,
                            warning_level=True,
                            statistics={'trade_count': self.trade_count}
                        )
                        self.on_drift_detected(drift_result)
            except Exception as e:
                logger.debug(f"Online learning update failed: {e}")

            # Update window manager
            prediction_error = 1.0 if error else 0.0
            self.window_manager.add(feature_vector, label, prediction_error)

        # Update online hyperparameter tuner
        if self.online_tuner is not None:
            performance = 1.0 if pnl > 0 else 0.0
            self.online_tuner.update(performance)

        # Auto-save periodically
        if self.trade_count % self.config.auto_save_interval == 0:
            self._auto_save()

        return {
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.trade_count,
            'error_detected': error is not None,
            'error_type': error.error_type.name if error else None,
            'drift_count': self.drift_count,
            'window_size': len(self.window_manager.data)
        }

    def predict_risk(
        self,
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk prediction.
        """
        self.prediction_count += 1

        # Get feature vector
        if features is not None:
            # Use provided features
            feature_vector = np.array(list(features.values()))
        else:
            # Extract current features
            feature_set = self.advanced_features.extract_all_features()
            feature_vector = feature_set.to_vector()

        if len(feature_vector) == 0:
            return {'error': 'No features available', 'risk_score': 0.5}

        predictions = {}

        # Real-time prediction
        try:
            response = self.realtime_predictor.predict(
                feature_vector,
                PredictionType.ERROR_PROBABILITY
            )
            predictions['realtime'] = {
                'prediction': response.prediction.tolist() if hasattr(response.prediction, 'tolist') else response.prediction,
                'confidence': response.confidence,
                'latency_ms': response.latency_ms
            }
        except Exception as e:
            logger.debug(f"Realtime prediction failed: {e}")

        # Ensemble prediction
        if self.ensemble is not None:
            try:
                processed = self.feature_pipeline.transform(feature_vector.reshape(1, -1))
                pred = self.ensemble.predict_proba(processed)[0]
                predictions['ensemble'] = {
                    'prediction': pred.tolist(),
                    'confidence': float(np.max(pred))
                }
            except Exception as e:
                logger.debug(f"Ensemble prediction failed: {e}")

        # Adaptive network prediction
        try:
            pred = self.adaptive_network.predict(feature_vector.reshape(1, -1))[0]
            predictions['adaptive'] = {
                'prediction': pred.tolist(),
                'confidence': float(np.max(pred))
            }
        except Exception as e:
            logger.debug(f"Adaptive prediction failed: {e}")

        # Combine predictions
        all_confidences = []
        for pred_name, pred_data in predictions.items():
            if 'confidence' in pred_data:
                all_confidences.append(pred_data['confidence'])

        combined_risk = np.mean(all_confidences) if all_confidences else 0.5

        return {
            'risk_score': combined_risk,
            'individual_predictions': predictions,
            'recommendation': 'AVOID' if combined_risk > 0.7 else ('CAUTION' if combined_risk > 0.5 else 'PROCEED'),
            'prediction_count': self.prediction_count
        }

    def optimize_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        """
        if self.optimizer is None:
            raise RuntimeError("Auto-optimization not enabled")

        self.optimization_count += 1

        def objective(params: Dict) -> float:
            # Create model with params
            config = NetworkConfig(
                input_size=X_train.shape[1],
                hidden_sizes=[
                    params.get('hidden_size_1', 64),
                    params.get('hidden_size_2', 32)
                ],
                output_size=y_train.shape[1],
                dropout_rate=params.get('dropout_rate', 0.2),
                learning_rate=params.get('learning_rate', 0.01)
            )
            model = DeepNetwork(config)

            # Train
            for epoch in range(50):
                # Mini-batch training
                indices = np.random.permutation(len(X_train))
                for i in range(0, len(indices), 32):
                    batch_idx = indices[i:i+32]
                    model.train_step(X_train[batch_idx], y_train[batch_idx])

            # Evaluate
            pred = model.predict(X_val)
            accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(y_val, axis=1))

            return accuracy

        result = self.optimizer.optimize(
            objective, self.param_space, self.config.optimization_budget
        )

        if self.on_optimization_complete:
            self.on_optimization_complete(result)

        logger.info(f"Optimization complete: best_score={result.best_score:.4f}")

        return result

    def get_performance_report(self) -> PerformanceReport:
        """Get comprehensive performance report"""
        return self.performance_analyzer.generate_report()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        base_stats = self.base_ml.get_statistics()

        return {
            'base_ml': base_stats,
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.trade_count if self.trade_count > 0 else 0,
            'prediction_count': self.prediction_count,
            'optimization_count': self.optimization_count,
            'drift_count': self.drift_count,
            'model_metrics': self.performance_analyzer.get_model_metrics(),
            'regime_performance': self.performance_analyzer.get_regime_performance(),
            'latency': self.latency_monitor.get_stats(),
            'realtime_predictor': self.realtime_predictor.get_statistics(),
            'window_size': len(self.window_manager.data),
            'online_learner': {
                'n_updates': self.online_learner.n_updates,
                'n_drifts': self.online_learner.n_drifts,
                'is_trained': self.online_learner.is_trained
            }
        }

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current optimized strategy weights"""
        return self.base_ml.get_strategy_weights()

    def _auto_save(self):
        """Auto-save state"""
        try:
            self.save_state()
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    def save_state(self, filepath: Optional[str] = None):
        """Save complete state"""
        filepath = filepath or self.config.state_file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config.to_dict(),
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'prediction_count': self.prediction_count,
            'optimization_count': self.optimization_count,
            'drift_count': self.drift_count,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Save base ML state
        self.base_ml.save_state()

        # Save performance report
        base_path = Path(filepath).parent
        self.performance_analyzer.save_report(str(base_path / 'performance_report.json'))

        logger.info(f"Enhanced ML state saved to {filepath}")

    def load_state(self, filepath: Optional[str] = None):
        """Load state from file"""
        filepath = filepath or self.config.state_file

        if not Path(filepath).exists():
            logger.warning(f"No state file found at {filepath}")
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.trade_count = state.get('trade_count', 0)
        self.error_count = state.get('error_count', 0)
        self.prediction_count = state.get('prediction_count', 0)
        self.optimization_count = state.get('optimization_count', 0)
        self.drift_count = state.get('drift_count', 0)

        # Load base ML state
        self.base_ml.load_state()

        logger.info(f"Enhanced ML state loaded from {filepath}")

    async def start(self):
        """Start background tasks"""
        self._running = True

        # Start async prediction worker if enabled
        if self.config.enable_async:
            self.realtime_predictor.start_async_worker()

        self._background_task = asyncio.create_task(self._background_loop())
        logger.info("EnhancedMLManager started")

    async def stop(self):
        """Stop and cleanup"""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        if self.config.enable_async:
            self.realtime_predictor.stop_async_worker()

        self.save_state()
        logger.info("EnhancedMLManager stopped")

    async def _background_loop(self):
        """Background loop for periodic tasks"""
        while self._running:
            try:
                # Check alerts
                alerts = self.realtime_monitor.check_alerts()
                for alert in alerts:
                    logger.warning(f"Performance alert: {alert['message']}")

                # Periodic reporting
                if self.trade_count > 0 and self.trade_count % 100 == 0:
                    stats = self.get_statistics()
                    logger.info(f"Performance: trades={stats['trade_count']}, error_rate={stats['error_rate']:.2%}")

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background loop error: {e}")
                await asyncio.sleep(60)

    def reset(self):
        """Reset all state"""
        self.trade_count = 0
        self.error_count = 0
        self.prediction_count = 0
        self.optimization_count = 0
        self.drift_count = 0

        self.base_ml.reset()
        self.performance_analyzer.reset()
        self.realtime_predictor.cache.clear() if self.realtime_predictor.cache else None

        logger.info("EnhancedMLManager reset complete")
