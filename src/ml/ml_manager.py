"""
ML Manager - Central Integration Point for Machine Learning System

This module integrates all ML components and provides a unified interface
for the trading system to interact with the ML capabilities.

The MLManager:
1. Coordinates error classification and pattern analysis
2. Manages model training and updates
3. Provides predictions for trade decisions
4. Handles persistence and state management
5. Exposes a simple API for trading components
"""

import logging
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json

from .error_classifier import ErrorClassifier, TradeError, ErrorType, ErrorSeverity
from .feature_extractor import FeatureExtractor, TradeFeatures, MarketFeatures
from .pattern_analyzer import PatternAnalyzer, PatternMatch
from .learning_engine import LearningEngine, LearningConfig, ModelType
from .model_store import ModelStore, ModelMetadata
from .adaptive_optimizer import AdaptiveOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuration for the ML system"""

    # Feature extraction
    pip_size: float = 0.0001
    history_length: int = 1000

    # Learning engine
    learning_rate: float = 0.01
    replay_buffer_size: int = 10000
    min_samples_to_train: int = 100
    training_interval: int = 50

    # Pattern analysis
    min_pattern_occurrences: int = 5
    similarity_threshold: float = 0.7

    # Optimization
    optimization_method: str = "thompson_sampling"
    risk_aversion: float = 0.5
    exploration_bonus: float = 1.0

    # Model store
    model_store_path: str = "data/models"
    max_model_versions: int = 10

    # Persistence
    auto_save_interval: int = 100  # Save every N trades
    state_file: str = "data/ml_state.json"

    # Thresholds
    error_risk_threshold: float = 0.7  # Block trade if risk > this
    min_confidence_threshold: float = 0.5

    # Enabled features
    enable_error_learning: bool = True
    enable_pattern_analysis: bool = True
    enable_adaptive_optimization: bool = True
    enable_auto_training: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pip_size': self.pip_size,
            'history_length': self.history_length,
            'learning_rate': self.learning_rate,
            'replay_buffer_size': self.replay_buffer_size,
            'min_samples_to_train': self.min_samples_to_train,
            'training_interval': self.training_interval,
            'min_pattern_occurrences': self.min_pattern_occurrences,
            'similarity_threshold': self.similarity_threshold,
            'optimization_method': self.optimization_method,
            'risk_aversion': self.risk_aversion,
            'exploration_bonus': self.exploration_bonus,
            'model_store_path': self.model_store_path,
            'max_model_versions': self.max_model_versions,
            'auto_save_interval': self.auto_save_interval,
            'state_file': self.state_file,
            'error_risk_threshold': self.error_risk_threshold,
            'min_confidence_threshold': self.min_confidence_threshold,
            'enable_error_learning': self.enable_error_learning,
            'enable_pattern_analysis': self.enable_pattern_analysis,
            'enable_adaptive_optimization': self.enable_adaptive_optimization,
            'enable_auto_training': self.enable_auto_training,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TradeAnalysis:
    """Complete ML analysis for a trade"""

    # Risk assessment
    error_risk: float
    predicted_success: bool
    prediction_confidence: float

    # Error predictions
    predicted_errors: List[Tuple[ErrorType, float]]
    pattern_matches: List[PatternMatch]

    # Recommendations
    should_trade: bool
    risk_factors: List[str]
    recommendations: List[str]

    # Position sizing
    suggested_size_multiplier: float

    # Strategy selection
    recommended_strategy: str
    strategy_confidence: float
    strategy_weights: Dict[str, float]

    # Metadata
    analysis_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_risk': self.error_risk,
            'predicted_success': self.predicted_success,
            'prediction_confidence': self.prediction_confidence,
            'predicted_errors': [(e.name, p) for e, p in self.predicted_errors],
            'should_trade': self.should_trade,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
            'suggested_size_multiplier': self.suggested_size_multiplier,
            'recommended_strategy': self.recommended_strategy,
            'strategy_confidence': self.strategy_confidence,
            'strategy_weights': self.strategy_weights,
            'analysis_time_ms': self.analysis_time_ms,
        }


class MLManager:
    """
    Central manager for all ML operations.

    This class provides a unified interface for:
    - Analyzing trades before execution
    - Recording trade outcomes for learning
    - Predicting errors and risks
    - Optimizing strategy weights
    - Managing model persistence
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        strategy_names: Optional[List[str]] = None
    ):
        """
        Initialize the ML Manager.

        Args:
            config: ML configuration
            strategy_names: List of strategy names to manage
        """
        self.config = config or MLConfig()

        # Default strategy names if not provided
        self.strategy_names = strategy_names or [
            'scalping', 'momentum', 'mean_reversion',
            'market_making', 'breakout', 'order_flow'
        ]

        # Initialize components
        self._init_components()

        # State tracking
        self.trade_count = 0
        self.error_count = 0
        self.last_save_time = datetime.now()

        # Callbacks
        self.on_weight_update: Optional[Callable[[Dict[str, float]], None]] = None
        self.on_risk_alert: Optional[Callable[[str, float], None]] = None
        self.on_training_complete: Optional[Callable[[Any], None]] = None

        # Running state
        self._running = False
        self._background_task: Optional[asyncio.Task] = None

        logger.info("MLManager initialized")

    def _init_components(self):
        """Initialize all ML components"""

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            pip_size=self.config.pip_size,
            history_length=self.config.history_length
        )

        # Error classifier
        self.error_classifier = ErrorClassifier(
            slippage_threshold_pips=1.0,
            timing_threshold_pct=0.3
        )

        # Pattern analyzer
        self.pattern_analyzer = PatternAnalyzer(
            min_pattern_occurrences=self.config.min_pattern_occurrences,
            similarity_threshold=self.config.similarity_threshold
        )

        # Learning engine
        learning_config = LearningConfig(
            base_learning_rate=self.config.learning_rate,
            replay_buffer_size=self.config.replay_buffer_size,
            min_samples_to_train=self.config.min_samples_to_train,
            training_interval=self.config.training_interval
        )
        self.learning_engine = LearningEngine(config=learning_config)

        # Model store
        self.model_store = ModelStore(
            base_path=self.config.model_store_path,
            max_versions_per_model=self.config.max_model_versions
        )

        # Adaptive optimizer
        self.optimizer = AdaptiveOptimizer(
            strategy_names=self.strategy_names,
            risk_aversion=self.config.risk_aversion,
            exploration_bonus=self.config.exploration_bonus
        )

        # Wire up callbacks
        self.learning_engine.on_training_complete = self._on_training_complete
        self.optimizer.on_weight_update = self._on_optimizer_weight_update

    def _on_training_complete(self, result):
        """Handle training completion"""
        # Save model if significant improvement
        if result.validation_accuracy > 0.6:
            self._save_model(result.model_type)

        if self.on_training_complete:
            self.on_training_complete(result)

    def _on_optimizer_weight_update(self, new_weights: Dict[str, float]):
        """Handle optimizer weight update"""
        if self.on_weight_update:
            self.on_weight_update(new_weights)

    def update_market_data(
        self,
        price: float,
        volume: float = 0.0,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        order_book: Optional[Dict] = None
    ):
        """
        Update with new market data.

        Args:
            price: Current price
            volume: Tick volume
            bid_price: Best bid price
            ask_price: Best ask price
            order_book: Order book snapshot (bids, asks lists)
        """
        self.feature_extractor.update_price(price, volume)

        if order_book:
            self.feature_extractor.update_order_book(
                order_book.get('bids', []),
                order_book.get('asks', [])
            )

    def analyze_trade(
        self,
        trade_id: str,
        strategy_name: str,
        signal_type: str,
        entry_price: float,
        expected_entry_price: float,
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
    ) -> TradeAnalysis:
        """
        Analyze a potential trade before execution.

        This is the main pre-trade analysis method that combines all ML
        predictions to provide a comprehensive risk assessment.

        Args:
            trade_id: Unique trade identifier
            strategy_name: Name of strategy generating the signal
            signal_type: Type of signal (BUY, SELL, etc.)
            entry_price: Expected entry price
            expected_entry_price: Originally expected price from signal
            signal_confidence: Confidence from strategy
            signal_strength: Signal strength value
            stop_loss: Planned stop loss price
            take_profit: Planned take profit price
            position_size: Planned position size in lots
            bid_price: Current best bid
            ask_price: Current best ask
            market_regime: Current market regime
            risk_level: Current risk level
            symbol: Trading symbol

        Returns:
            TradeAnalysis with complete ML assessment
        """
        start_time = datetime.now()

        # Extract market features
        market_features = self.feature_extractor.extract_market_features(
            bid_price, ask_price, symbol
        )

        # Extract trade features
        trade_features = self.feature_extractor.extract_trade_features(
            trade_id=trade_id,
            entry_price=entry_price,
            expected_entry_price=expected_entry_price,
            entry_latency_ms=0,  # Will be updated after execution
            signal_type=signal_type,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            strategy_name=strategy_name,
            num_confirming=1,
            num_conflicting=0,
            position_size_lots=position_size,
            position_risk_pct=1.0,  # TODO: Calculate actual risk
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_win_rate=self.optimizer.performance.get(
                strategy_name, type('', (), {'win_rate': 0.5})()
            ).win_rate,
            strategy_recent_pnl=self.optimizer.performance.get(
                strategy_name, type('', (), {'recent_pnl': 0.0})()
            ).recent_pnl,
            consecutive_wins=0,
            consecutive_losses=0,
            regime_win_rate=0.5,
            daily_pnl=0,
            drawdown_pct=0,
            var_utilization=0,
            risk_level=risk_level,
            time_since_last_trade_ms=0,
            trades_in_last_hour=0,
            bid_price=bid_price,
            ask_price=ask_price,
            symbol=symbol
        )

        # Get feature vector
        feature_vector = trade_features.to_vector()

        # Predict error risk
        error_risk = 0.5
        predicted_errors = []

        if self.config.enable_error_learning:
            error_risk = self.learning_engine.predict_error_risk(feature_vector)
            predicted_errors = self.learning_engine.predict_error_type(feature_vector)

        # Predict success
        predicted_success, prediction_confidence = True, 0.5
        if self.config.enable_error_learning:
            predicted_success, prediction_confidence = self.learning_engine.predict_outcome(feature_vector)

        # Pattern matching
        pattern_matches = []
        if self.config.enable_pattern_analysis:
            features_dict = trade_features.to_dict()
            features_dict.update(market_features.to_dict())
            pattern_matches = self.pattern_analyzer.analyze_current_conditions(
                features_dict, strategy_name, market_regime
            )

        # Collect risk factors and recommendations
        risk_factors = []
        recommendations = []

        # From error predictions
        for error_type, prob in predicted_errors[:3]:
            if prob > 0.3:
                risk_factors.append(f"High probability of {error_type.name}: {prob:.1%}")

        # From pattern matches
        for match in pattern_matches[:3]:
            risk_factors.extend(match.risk_factors[:2])
            recommendations.extend(match.recommendations[:2])

        # Add market-based risk factors
        if market_features.volatility_ratio > 1.5:
            risk_factors.append("Elevated volatility ratio")
        if market_features.spread_pips > 2.0:
            risk_factors.append("Wide spread")

        # Determine if should trade
        should_trade = (
            error_risk < self.config.error_risk_threshold and
            prediction_confidence >= self.config.min_confidence_threshold and
            not any(m.overall_risk > 0.8 for m in pattern_matches)
        )

        if not should_trade:
            if error_risk >= self.config.error_risk_threshold:
                recommendations.insert(0, f"Consider waiting - error risk high ({error_risk:.1%})")
            if any(m.overall_risk > 0.8 for m in pattern_matches):
                recommendations.insert(0, "Known error pattern detected - proceed with caution")

        # Get position size multiplier
        size_multiplier = 1.0
        if self.config.enable_adaptive_optimization:
            size_multiplier = self.optimizer.get_position_size_multiplier(
                strategy_name, error_risk
            )

        # Get strategy recommendation
        recommended_strategy, strategy_confidence = strategy_name, signal_confidence
        if self.config.enable_adaptive_optimization:
            recommended_strategy, strategy_confidence = self.optimizer.get_strategy_recommendation(
                market_regime, risk_level, pattern_matches
            )

        # Get current strategy weights
        strategy_weights = self.optimizer.weights.copy()

        # Calculate analysis time
        analysis_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Trigger risk alert if needed
        if error_risk > 0.6 and self.on_risk_alert:
            self.on_risk_alert(strategy_name, error_risk)

        return TradeAnalysis(
            error_risk=error_risk,
            predicted_success=predicted_success,
            prediction_confidence=prediction_confidence,
            predicted_errors=predicted_errors,
            pattern_matches=pattern_matches,
            should_trade=should_trade,
            risk_factors=risk_factors,
            recommendations=list(set(recommendations))[:5],  # Deduplicate
            suggested_size_multiplier=size_multiplier,
            recommended_strategy=recommended_strategy,
            strategy_confidence=strategy_confidence,
            strategy_weights=strategy_weights,
            analysis_time_ms=analysis_time_ms
        )

    def record_trade_result(
        self,
        trade_id: str,
        strategy_name: str,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        expected_entry_price: float,
        planned_stop: float,
        planned_target: float,
        signal_confidence: float,
        signal_strength: str,
        pnl: float,
        holding_time_ms: int,
        market_regime: str,
        volatility: float,
        spread_pips: float,
        symbol: str = "EURUSD"
    ) -> Optional[TradeError]:
        """
        Record a completed trade for learning.

        This is called after trade execution to update all ML components
        with the trade outcome.

        Args:
            trade_id: Trade identifier
            strategy_name: Strategy that generated the trade
            signal_type: Type of signal
            entry_price: Actual entry price
            exit_price: Actual exit price
            expected_entry_price: Originally expected entry
            planned_stop: Planned stop loss
            planned_target: Planned take profit
            signal_confidence: Original signal confidence
            signal_strength: Original signal strength
            pnl: Profit/loss in currency
            holding_time_ms: How long position was held
            market_regime: Market regime during trade
            volatility: Volatility during trade
            spread_pips: Spread during trade
            symbol: Trading symbol

        Returns:
            TradeError if an error was classified, None otherwise
        """
        self.trade_count += 1

        # Get price history for analysis
        price_history = list(self.feature_extractor.price_history)[-100:]

        # Classify error if trade was losing or suboptimal
        error = None
        if pnl < 0 or self._was_suboptimal_trade(pnl, planned_target, entry_price, exit_price, signal_type):
            error = self.error_classifier.classify_trade_error(
                trade_id=trade_id,
                symbol=symbol,
                strategy_name=strategy_name,
                signal_type=signal_type,
                entry_price=entry_price,
                exit_price=exit_price,
                planned_stop=planned_stop,
                planned_target=planned_target,
                signal_confidence=signal_confidence,
                signal_strength=signal_strength,
                expected_entry_price=expected_entry_price,
                actual_pnl=pnl,
                holding_time_ms=holding_time_ms,
                market_regime=market_regime,
                volatility=volatility,
                spread_pips=spread_pips,
                price_history=price_history
            )
            self.error_count += 1

            # Add error to pattern analyzer
            if self.config.enable_pattern_analysis:
                self.pattern_analyzer.add_error(error)

        # Update learning engine
        if self.config.enable_error_learning:
            feature_vector = self._get_feature_vector_from_trade(
                entry_price, exit_price, signal_confidence, market_regime
            )
            strategy_idx = self.strategy_names.index(strategy_name) if strategy_name in self.strategy_names else 0
            self.learning_engine.add_trade_experience(
                features=feature_vector,
                error=error,
                pnl=pnl,
                strategy_index=strategy_idx
            )

        # Update optimizer
        if self.config.enable_adaptive_optimization:
            self.optimizer.record_trade_result(
                strategy_name=strategy_name,
                pnl=pnl,
                error=error,
                market_regime=market_regime
            )

        # Auto-save periodically
        if self.trade_count % self.config.auto_save_interval == 0:
            self._auto_save()

        logger.info(
            f"Trade {trade_id} recorded: PnL={pnl:.2f}, "
            f"Error={error.error_type.name if error else 'None'}"
        )

        return error

    def _was_suboptimal_trade(
        self,
        pnl: float,
        planned_target: float,
        entry_price: float,
        exit_price: float,
        signal_type: str
    ) -> bool:
        """Check if trade was suboptimal (could have been better)"""
        is_long = signal_type in ['BUY', 'CLOSE_SHORT']

        if is_long:
            max_possible = (planned_target - entry_price) * 10000  # Rough pip calc
        else:
            max_possible = (entry_price - planned_target) * 10000

        # If we got less than 50% of potential profit, it's suboptimal
        if max_possible > 0 and pnl < max_possible * 0.5:
            return True

        return False

    def _get_feature_vector_from_trade(
        self,
        entry_price: float,
        exit_price: float,
        signal_confidence: float,
        market_regime: str
    ) -> np.ndarray:
        """Get feature vector for a trade (simplified version)"""
        import numpy as np

        # Use current market features
        bid = entry_price * 0.9999
        ask = entry_price * 1.0001

        market_features = self.feature_extractor.extract_market_features(bid, ask)

        return market_features.to_vector()

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current optimized strategy weights"""
        return self.optimizer.weights.copy()

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get strategy weights for a specific regime"""
        return self.optimizer.get_regime_weights(regime)

    def optimize_strategies(self) -> OptimizationResult:
        """Manually trigger strategy optimization"""
        return self.optimizer.optimize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ML statistics"""
        return {
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.trade_count if self.trade_count > 0 else 0,
            'error_classifier': self.error_classifier.get_error_statistics(),
            'pattern_analyzer': self.pattern_analyzer.get_pattern_summary(),
            'learning_engine': self.learning_engine.get_training_statistics(),
            'optimizer': self.optimizer.get_statistics(),
            'model_store': self.model_store.get_statistics(),
        }

    def get_strategy_error_profile(self, strategy_name: str) -> Dict[str, Any]:
        """Get error profile for a specific strategy"""
        return self.error_classifier.get_strategy_error_profile(strategy_name)

    def get_regime_error_profile(self, regime: str) -> Dict[str, Any]:
        """Get error profile for a specific regime"""
        return self.error_classifier.get_regime_error_profile(regime)

    def _save_model(self, model_type: ModelType):
        """Save a model to the store"""
        if model_type == ModelType.ERROR_CLASSIFIER:
            # Get weights from learning engine
            if self.learning_engine.error_classifier:
                weights_data = {
                    'weights': [w.tolist() for w in self.learning_engine.error_classifier.weights],
                    'biases': [b.tolist() for b in self.learning_engine.error_classifier.biases]
                }

                # Get training stats
                stats = self.learning_engine.get_training_statistics()

                self.model_store.save_model(
                    model_type="error_classifier",
                    weights_data=weights_data,
                    accuracy=stats.get('recent_training', {}).get('avg_accuracy', 0),
                    loss=stats.get('recent_training', {}).get('avg_loss', 0),
                    validation_accuracy=stats.get('recent_training', {}).get('avg_val_accuracy', 0),
                    validation_loss=0,
                    samples_trained=stats.get('error_samples', 0),
                    epochs_trained=self.learning_engine.training_count,
                    training_time_ms=0,
                    config=self.config.to_dict(),
                    feature_size=self.learning_engine.feature_size or 0,
                    output_size=self.learning_engine.num_error_types
                )

    def _auto_save(self):
        """Auto-save state"""
        try:
            self.save_state()
            logger.debug("Auto-save completed")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    def save_state(self, filepath: Optional[str] = None):
        """
        Save complete ML state to file.

        Args:
            filepath: Path to save state (uses config default if None)
        """
        filepath = filepath or self.config.state_file
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config.to_dict(),
            'trade_count': self.trade_count,
            'error_count': self.error_count,
            'timestamp': datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        # Save component states
        base_path = Path(filepath).parent

        self.error_classifier.save_error_history(str(base_path / "error_history.json"))
        self.pattern_analyzer.save_patterns(str(base_path / "patterns.json"))
        self.learning_engine.save_models(str(base_path / "learning_engine.json"))
        self.optimizer.save_state(str(base_path / "optimizer.json"))

        logger.info(f"ML state saved to {filepath}")

    def load_state(self, filepath: Optional[str] = None):
        """
        Load complete ML state from file.

        Args:
            filepath: Path to load state (uses config default if None)
        """
        filepath = filepath or self.config.state_file

        if not Path(filepath).exists():
            logger.warning(f"No state file found at {filepath}")
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.trade_count = state.get('trade_count', 0)
        self.error_count = state.get('error_count', 0)

        # Load component states
        base_path = Path(filepath).parent

        error_history_path = base_path / "error_history.json"
        if error_history_path.exists():
            self.error_classifier.load_error_history(str(error_history_path))

        patterns_path = base_path / "patterns.json"
        if patterns_path.exists():
            self.pattern_analyzer.load_patterns(str(patterns_path))

        learning_path = base_path / "learning_engine.json"
        if learning_path.exists():
            self.learning_engine.load_models(str(learning_path))

        optimizer_path = base_path / "optimizer.json"
        if optimizer_path.exists():
            self.optimizer.load_state(str(optimizer_path))

        logger.info(f"ML state loaded from {filepath}")

    async def start(self):
        """Start ML manager background tasks"""
        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info("MLManager started")

    async def stop(self):
        """Stop ML manager"""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # Final save
        self.save_state()
        logger.info("MLManager stopped")

    async def _background_loop(self):
        """Background loop for periodic tasks"""
        while self._running:
            try:
                # Periodic optimization
                if self.config.enable_adaptive_optimization:
                    if self.trade_count > 0 and self.trade_count % 100 == 0:
                        self.optimizer.optimize()

                # Periodic model training check
                if self.config.enable_auto_training:
                    # Training is handled by learning engine internally
                    pass

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background loop: {e}")
                await asyncio.sleep(60)

    def reset(self):
        """Reset all ML state"""
        self.trade_count = 0
        self.error_count = 0

        self.error_classifier.clear_history()
        self.feature_extractor.reset()
        self.learning_engine.reset()
        self.optimizer.reset()

        logger.info("MLManager reset complete")
