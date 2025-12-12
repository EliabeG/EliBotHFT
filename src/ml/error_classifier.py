"""
Trade Error Classification System

This module provides comprehensive error classification for trading operations,
enabling the ML system to learn from different types of trading errors.

Error Types:
- PREDICTION_ERROR: Signal predicted wrong direction
- TIMING_ERROR: Right direction, wrong entry/exit time
- SLIPPAGE_ERROR: Execution price significantly worse than expected
- RISK_ERROR: Position violated risk limits
- REGIME_ERROR: Strategy not suited for current market regime
- SIGNAL_CONFLICT: Conflicting signals from strategies
- EXECUTION_ERROR: Order execution failures
- DATA_ERROR: Market data issues affecting decisions
"""

import logging
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of trading error types"""

    # Direction Errors
    PREDICTION_ERROR = auto()       # Signal predicted wrong direction
    FALSE_BREAKOUT = auto()         # Breakout signal was false
    TREND_REVERSAL_MISS = auto()    # Missed trend reversal

    # Timing Errors
    EARLY_ENTRY = auto()            # Entered too early
    LATE_ENTRY = auto()             # Entered too late, missed optimal price
    EARLY_EXIT = auto()             # Exited too early, missed profits
    LATE_EXIT = auto()              # Exited too late, reduced profits
    PREMATURE_STOP = auto()         # Stop loss hit but price reversed

    # Execution Errors
    SLIPPAGE_ERROR = auto()         # Significant slippage on execution
    PARTIAL_FILL = auto()           # Order only partially filled
    REJECTION = auto()              # Order rejected by broker
    TIMEOUT = auto()                # Order execution timeout

    # Risk Errors
    POSITION_SIZE_ERROR = auto()    # Position too large/small
    STOP_TOO_TIGHT = auto()         # Stop loss too close
    STOP_TOO_WIDE = auto()          # Stop loss too far
    DRAWDOWN_BREACH = auto()        # Exceeded drawdown limits

    # Regime Errors
    REGIME_MISMATCH = auto()        # Strategy not suited for regime
    VOLATILITY_SPIKE = auto()       # Unexpected volatility spike
    LIQUIDITY_GAP = auto()          # Insufficient liquidity
    SPREAD_WIDENING = auto()        # Spread widened unexpectedly

    # Strategy Errors
    SIGNAL_CONFLICT = auto()        # Conflicting signals
    CONFIDENCE_OVERESTIMATE = auto() # Signal confidence was too high
    CORRELATION_MISS = auto()       # Missed correlation with other assets

    # Data Errors
    STALE_DATA = auto()             # Decisions made on stale data
    DATA_GAP = auto()               # Missing data points
    PRICE_SPIKE = auto()            # Abnormal price spike (data issue)

    # Unknown
    UNKNOWN = auto()                # Unclassified error


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = 1         # Minor impact, cosmetic
    MEDIUM = 2      # Moderate impact, reduced profits
    HIGH = 3        # Significant impact, caused losses
    CRITICAL = 4    # Severe impact, large losses
    FATAL = 5       # System-level failure


@dataclass
class TradeError:
    """Represents a classified trading error"""

    # Identification
    error_id: str
    trade_id: str
    timestamp: datetime

    # Classification
    error_type: ErrorType
    severity: ErrorSeverity
    confidence: float  # How confident are we in this classification

    # Context
    symbol: str
    strategy_name: str
    signal_type: str

    # Financial Impact
    expected_pnl: float
    actual_pnl: float
    pnl_impact: float  # expected - actual
    slippage_pips: float

    # Market Context at Error Time
    market_regime: str
    volatility: float
    spread_pips: float

    # Trade Details
    entry_price: float
    exit_price: float
    planned_stop: float
    planned_target: float
    actual_stop: Optional[float] = None

    # Signal Details
    signal_confidence: float = 0.0
    signal_strength: str = ""

    # Analysis
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Features at Error Time (for ML)
    features: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_id': self.error_id,
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type.name,
            'severity': self.severity.name,
            'confidence': self.confidence,
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'signal_type': self.signal_type,
            'expected_pnl': self.expected_pnl,
            'actual_pnl': self.actual_pnl,
            'pnl_impact': self.pnl_impact,
            'slippage_pips': self.slippage_pips,
            'market_regime': self.market_regime,
            'volatility': self.volatility,
            'spread_pips': self.spread_pips,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'planned_stop': self.planned_stop,
            'planned_target': self.planned_target,
            'actual_stop': self.actual_stop,
            'signal_confidence': self.signal_confidence,
            'signal_strength': self.signal_strength,
            'root_cause': self.root_cause,
            'contributing_factors': self.contributing_factors,
            'recommendations': self.recommendations,
            'features': self.features,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeError':
        """Create from dictionary"""
        return cls(
            error_id=data['error_id'],
            trade_id=data['trade_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            error_type=ErrorType[data['error_type']],
            severity=ErrorSeverity[data['severity']],
            confidence=data['confidence'],
            symbol=data['symbol'],
            strategy_name=data['strategy_name'],
            signal_type=data['signal_type'],
            expected_pnl=data['expected_pnl'],
            actual_pnl=data['actual_pnl'],
            pnl_impact=data['pnl_impact'],
            slippage_pips=data['slippage_pips'],
            market_regime=data['market_regime'],
            volatility=data['volatility'],
            spread_pips=data['spread_pips'],
            entry_price=data['entry_price'],
            exit_price=data['exit_price'],
            planned_stop=data['planned_stop'],
            planned_target=data['planned_target'],
            actual_stop=data.get('actual_stop'),
            signal_confidence=data.get('signal_confidence', 0.0),
            signal_strength=data.get('signal_strength', ''),
            root_cause=data.get('root_cause', ''),
            contributing_factors=data.get('contributing_factors', []),
            recommendations=data.get('recommendations', []),
            features=data.get('features', {}),
            metadata=data.get('metadata', {}),
        )


class ErrorClassifier:
    """
    Intelligent error classification system that analyzes trades
    and classifies errors using rule-based and ML approaches.
    """

    def __init__(
        self,
        slippage_threshold_pips: float = 1.0,
        timing_threshold_pct: float = 0.3,
        volatility_spike_threshold: float = 2.0,
        min_confidence_threshold: float = 0.6
    ):
        """
        Initialize the error classifier.

        Args:
            slippage_threshold_pips: Threshold for significant slippage
            timing_threshold_pct: Threshold for timing errors (% of move)
            volatility_spike_threshold: Multiple of normal vol for spike
            min_confidence_threshold: Minimum confidence for classification
        """
        self.slippage_threshold = slippage_threshold_pips
        self.timing_threshold = timing_threshold_pct
        self.vol_spike_threshold = volatility_spike_threshold
        self.min_confidence = min_confidence_threshold

        # Error history for pattern analysis
        self.error_history: List[TradeError] = []
        self.error_counts: Dict[str, Dict[ErrorType, int]] = defaultdict(lambda: defaultdict(int))

        # Statistics
        self.total_errors = 0
        self.errors_by_type: Dict[ErrorType, int] = defaultdict(int)
        self.errors_by_strategy: Dict[str, int] = defaultdict(int)
        self.errors_by_regime: Dict[str, int] = defaultdict(int)

        # ML model for classification (will be trained over time)
        self._ml_model = None
        self._model_trained = False

        logger.info("ErrorClassifier initialized")

    def classify_trade_error(
        self,
        trade_id: str,
        symbol: str,
        strategy_name: str,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        planned_stop: float,
        planned_target: float,
        signal_confidence: float,
        signal_strength: str,
        expected_entry_price: float,
        actual_pnl: float,
        holding_time_ms: int,
        market_regime: str,
        volatility: float,
        spread_pips: float,
        price_history: List[float],
        volume_history: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradeError:
        """
        Classify the error type for a losing or suboptimal trade.

        This method analyzes multiple factors to determine:
        1. Primary error type
        2. Severity of the error
        3. Contributing factors
        4. Recommendations for improvement

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            strategy_name: Name of strategy that generated signal
            signal_type: Type of signal (BUY, SELL, etc.)
            entry_price: Actual entry price
            exit_price: Exit price
            planned_stop: Originally planned stop loss
            planned_target: Originally planned take profit
            signal_confidence: Confidence level of the original signal
            signal_strength: Strength of the signal
            expected_entry_price: Price expected at signal generation
            actual_pnl: Actual profit/loss in currency
            holding_time_ms: How long position was held
            market_regime: Market regime at trade time
            volatility: Volatility at trade time
            spread_pips: Spread at trade time
            price_history: Recent price history for analysis
            volume_history: Recent volume history (optional)
            metadata: Additional metadata

        Returns:
            TradeError with complete classification and analysis
        """
        timestamp = datetime.now()
        error_id = f"ERR_{trade_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"

        # Calculate key metrics
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01
        slippage_pips = abs(entry_price - expected_entry_price) / pip_size

        # Calculate expected PnL (what we hoped to make)
        if signal_type in ['BUY', 'CLOSE_SHORT']:
            expected_pnl = (planned_target - entry_price) / pip_size * 10  # Rough estimate
        else:
            expected_pnl = (entry_price - planned_target) / pip_size * 10

        pnl_impact = expected_pnl - actual_pnl

        # Analyze the trade
        error_type, severity, confidence = self._analyze_trade(
            signal_type=signal_type,
            entry_price=entry_price,
            exit_price=exit_price,
            planned_stop=planned_stop,
            planned_target=planned_target,
            expected_entry_price=expected_entry_price,
            actual_pnl=actual_pnl,
            holding_time_ms=holding_time_ms,
            slippage_pips=slippage_pips,
            volatility=volatility,
            spread_pips=spread_pips,
            price_history=price_history,
            signal_confidence=signal_confidence
        )

        # Determine root cause and recommendations
        root_cause, factors, recommendations = self._determine_root_cause(
            error_type=error_type,
            strategy_name=strategy_name,
            market_regime=market_regime,
            volatility=volatility,
            slippage_pips=slippage_pips,
            signal_confidence=signal_confidence,
            actual_pnl=actual_pnl
        )

        # Extract features for ML
        features = self._extract_error_features(
            price_history=price_history,
            volume_history=volume_history,
            volatility=volatility,
            spread_pips=spread_pips,
            signal_confidence=signal_confidence,
            holding_time_ms=holding_time_ms,
            slippage_pips=slippage_pips
        )

        # Create error object
        error = TradeError(
            error_id=error_id,
            trade_id=trade_id,
            timestamp=timestamp,
            error_type=error_type,
            severity=severity,
            confidence=confidence,
            symbol=symbol,
            strategy_name=strategy_name,
            signal_type=signal_type,
            expected_pnl=expected_pnl,
            actual_pnl=actual_pnl,
            pnl_impact=pnl_impact,
            slippage_pips=slippage_pips,
            market_regime=market_regime,
            volatility=volatility,
            spread_pips=spread_pips,
            entry_price=entry_price,
            exit_price=exit_price,
            planned_stop=planned_stop,
            planned_target=planned_target,
            signal_confidence=signal_confidence,
            signal_strength=signal_strength,
            root_cause=root_cause,
            contributing_factors=factors,
            recommendations=recommendations,
            features=features,
            metadata=metadata or {}
        )

        # Update statistics
        self._update_statistics(error)

        # Store in history
        self.error_history.append(error)

        logger.info(
            f"Classified error: {error_type.name} "
            f"(severity={severity.name}, confidence={confidence:.2f}) "
            f"for trade {trade_id}"
        )

        return error

    def _analyze_trade(
        self,
        signal_type: str,
        entry_price: float,
        exit_price: float,
        planned_stop: float,
        planned_target: float,
        expected_entry_price: float,
        actual_pnl: float,
        holding_time_ms: int,
        slippage_pips: float,
        volatility: float,
        spread_pips: float,
        price_history: List[float],
        signal_confidence: float
    ) -> Tuple[ErrorType, ErrorSeverity, float]:
        """
        Analyze a trade to determine error type, severity, and confidence.

        Returns:
            Tuple of (ErrorType, ErrorSeverity, confidence)
        """
        candidates: List[Tuple[ErrorType, float, ErrorSeverity]] = []

        is_long = signal_type in ['BUY', 'CLOSE_SHORT']
        pip_size = 0.0001  # Simplified

        # Calculate price movements
        if len(price_history) > 0:
            max_favorable = max(price_history) if is_long else min(price_history)
            max_adverse = min(price_history) if is_long else max(price_history)

            mfe_pips = abs(max_favorable - entry_price) / pip_size  # Max favorable excursion
            mae_pips = abs(entry_price - max_adverse) / pip_size    # Max adverse excursion
        else:
            mfe_pips = 0
            mae_pips = 0

        # Calculate what actually happened
        if is_long:
            trade_direction_correct = exit_price > entry_price
            hit_stop = exit_price <= planned_stop
            reached_target = exit_price >= planned_target
        else:
            trade_direction_correct = exit_price < entry_price
            hit_stop = exit_price >= planned_stop
            reached_target = exit_price <= planned_target

        # 1. Check for SLIPPAGE_ERROR
        if slippage_pips > self.slippage_threshold:
            severity = self._calculate_severity_by_pips(slippage_pips)
            confidence = min(0.95, 0.5 + slippage_pips / 10)
            candidates.append((ErrorType.SLIPPAGE_ERROR, confidence, severity))

        # 2. Check for PREDICTION_ERROR (wrong direction)
        if actual_pnl < 0 and not trade_direction_correct:
            severity = self._calculate_severity_by_pnl(actual_pnl)
            # Higher confidence if signal confidence was also high
            confidence = 0.7 + (signal_confidence * 0.2)
            candidates.append((ErrorType.PREDICTION_ERROR, confidence, severity))

        # 3. Check for TIMING_ERROR
        if mfe_pips > mae_pips * 1.5 and actual_pnl < 0:
            # Price moved favorably but we lost money - timing issue
            if hit_stop and mfe_pips > 10:
                # We hit stop but price went our way significantly
                candidates.append((
                    ErrorType.PREMATURE_STOP,
                    0.8,
                    ErrorSeverity.HIGH
                ))
            elif holding_time_ms < 5000 and mfe_pips > 5:
                # Exited too early
                candidates.append((
                    ErrorType.EARLY_EXIT,
                    0.75,
                    ErrorSeverity.MEDIUM
                ))

        # 4. Check for LATE_EXIT
        if mfe_pips > 20 and actual_pnl > 0 and actual_pnl < mfe_pips * 5:
            # We were up significantly more than we captured
            captured_ratio = actual_pnl / (mfe_pips * 10) if mfe_pips > 0 else 1
            if captured_ratio < 0.3:
                candidates.append((
                    ErrorType.LATE_EXIT,
                    0.7,
                    ErrorSeverity.MEDIUM
                ))

        # 5. Check for EARLY_ENTRY
        if mae_pips > mfe_pips and mae_pips > 10 and actual_pnl < 0:
            # Could have gotten better entry
            candidates.append((
                ErrorType.EARLY_ENTRY,
                0.65,
                ErrorSeverity.MEDIUM
            ))

        # 6. Check for STOP_TOO_TIGHT
        if hit_stop and mfe_pips > mae_pips * 2:
            # Stop was hit but price eventually went our way
            candidates.append((
                ErrorType.STOP_TOO_TIGHT,
                0.75,
                ErrorSeverity.HIGH
            ))

        # 7. Check for VOLATILITY_SPIKE
        if volatility > self.vol_spike_threshold * 0.005:  # Assuming 0.5% is normal
            candidates.append((
                ErrorType.VOLATILITY_SPIKE,
                0.6 + (volatility / 0.02),
                ErrorSeverity.MEDIUM
            ))

        # 8. Check for SPREAD_WIDENING
        if spread_pips > 2.0:  # Normal is ~0.5-1.0 for major pairs
            candidates.append((
                ErrorType.SPREAD_WIDENING,
                min(0.9, 0.5 + spread_pips / 5),
                ErrorSeverity.MEDIUM
            ))

        # 9. Check for CONFIDENCE_OVERESTIMATE
        if actual_pnl < 0 and signal_confidence > 0.8:
            candidates.append((
                ErrorType.CONFIDENCE_OVERESTIMATE,
                0.6,
                ErrorSeverity.MEDIUM
            ))

        # 10. Check for FALSE_BREAKOUT (if applicable)
        if len(price_history) > 10:
            # Check if price reversed quickly after breakout
            initial_move = price_history[5] - entry_price if is_long else entry_price - price_history[5]
            final_move = exit_price - entry_price if is_long else entry_price - exit_price

            if initial_move > 0 and final_move < 0:
                candidates.append((
                    ErrorType.FALSE_BREAKOUT,
                    0.7,
                    ErrorSeverity.HIGH
                ))

        # Select the highest confidence error type
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0], candidates[0][2], candidates[0][1]

        # Default to UNKNOWN if no specific error identified
        return ErrorType.UNKNOWN, ErrorSeverity.LOW, 0.5

    def _calculate_severity_by_pips(self, pips: float) -> ErrorSeverity:
        """Calculate severity based on pip value"""
        if pips < 2:
            return ErrorSeverity.LOW
        elif pips < 5:
            return ErrorSeverity.MEDIUM
        elif pips < 10:
            return ErrorSeverity.HIGH
        elif pips < 20:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.FATAL

    def _calculate_severity_by_pnl(self, pnl: float) -> ErrorSeverity:
        """Calculate severity based on PnL"""
        abs_pnl = abs(pnl)
        if abs_pnl < 10:
            return ErrorSeverity.LOW
        elif abs_pnl < 50:
            return ErrorSeverity.MEDIUM
        elif abs_pnl < 100:
            return ErrorSeverity.HIGH
        elif abs_pnl < 200:
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.FATAL

    def _determine_root_cause(
        self,
        error_type: ErrorType,
        strategy_name: str,
        market_regime: str,
        volatility: float,
        slippage_pips: float,
        signal_confidence: float,
        actual_pnl: float
    ) -> Tuple[str, List[str], List[str]]:
        """
        Determine root cause and generate recommendations.

        Returns:
            Tuple of (root_cause, contributing_factors, recommendations)
        """
        root_cause = ""
        factors = []
        recommendations = []

        # Root cause analysis based on error type
        root_cause_map = {
            ErrorType.PREDICTION_ERROR: "Signal model predicted incorrect market direction",
            ErrorType.FALSE_BREAKOUT: "Breakout signal was triggered by noise, not true breakout",
            ErrorType.TREND_REVERSAL_MISS: "Failed to detect trend reversal in time",
            ErrorType.EARLY_ENTRY: "Entry signal triggered before optimal price level",
            ErrorType.LATE_ENTRY: "Entry delayed, missing optimal price",
            ErrorType.EARLY_EXIT: "Position closed before reaching profit potential",
            ErrorType.LATE_EXIT: "Exit delayed, giving back profits",
            ErrorType.PREMATURE_STOP: "Stop loss placed too aggressively for market conditions",
            ErrorType.SLIPPAGE_ERROR: "Execution price significantly deviated from expected",
            ErrorType.POSITION_SIZE_ERROR: "Position sizing inappropriate for volatility",
            ErrorType.STOP_TOO_TIGHT: "Stop loss too close to entry for market volatility",
            ErrorType.STOP_TOO_WIDE: "Stop loss too far, increasing risk exposure",
            ErrorType.REGIME_MISMATCH: f"Strategy '{strategy_name}' not optimal for {market_regime} regime",
            ErrorType.VOLATILITY_SPIKE: "Unexpected volatility expansion affected trade",
            ErrorType.SPREAD_WIDENING: "Spread widened during trade execution",
            ErrorType.SIGNAL_CONFLICT: "Multiple strategies provided conflicting signals",
            ErrorType.CONFIDENCE_OVERESTIMATE: "Signal confidence did not reflect true probability",
        }

        root_cause = root_cause_map.get(error_type, "Unknown root cause")

        # Determine contributing factors
        if volatility > 0.01:
            factors.append("High market volatility")
        if slippage_pips > 1.0:
            factors.append(f"Significant slippage ({slippage_pips:.1f} pips)")
        if signal_confidence > 0.9 and actual_pnl < 0:
            factors.append("Overconfident signal")

        # Strategy-regime mismatch factors
        regime_strategy_fit = {
            'TRENDING_UP': ['momentum', 'breakout'],
            'TRENDING_DOWN': ['momentum', 'breakout'],
            'RANGING': ['mean_reversion', 'market_making'],
            'VOLATILE': ['scalping'],
            'QUIET': ['market_making', 'mean_reversion']
        }

        optimal_strategies = regime_strategy_fit.get(market_regime, [])
        if strategy_name.lower() not in [s.lower() for s in optimal_strategies]:
            factors.append(f"Strategy may not be optimal for {market_regime} regime")

        # Generate recommendations
        recommendations_map = {
            ErrorType.PREDICTION_ERROR: [
                "Review signal generation algorithm",
                "Add additional confirmation indicators",
                "Reduce position size for low-confidence signals",
                "Consider regime detection before signal generation"
            ],
            ErrorType.FALSE_BREAKOUT: [
                "Add volume confirmation to breakout signals",
                "Increase breakout threshold",
                "Wait for retest before entry",
                "Check for support/resistance clusters"
            ],
            ErrorType.EARLY_ENTRY: [
                "Add pullback confirmation",
                "Use limit orders instead of market orders",
                "Wait for price to settle after signal"
            ],
            ErrorType.LATE_ENTRY: [
                "Reduce latency in signal processing",
                "Use aggressive IOC orders during momentum",
                "Pre-position with smaller size"
            ],
            ErrorType.EARLY_EXIT: [
                "Implement trailing stops",
                "Use partial position exits",
                "Extend take profit targets"
            ],
            ErrorType.LATE_EXIT: [
                "Implement time-based exit rules",
                "Add momentum exhaustion detection",
                "Tighten trailing stops during reversal signals"
            ],
            ErrorType.PREMATURE_STOP: [
                "Widen stops based on ATR",
                "Use volatility-adjusted stop distances",
                "Consider using options for protection instead"
            ],
            ErrorType.SLIPPAGE_ERROR: [
                "Use limit orders during normal conditions",
                "Reduce position size in low liquidity",
                "Avoid trading during news events"
            ],
            ErrorType.STOP_TOO_TIGHT: [
                "Base stop on ATR multiple",
                "Use wider stops in volatile conditions",
                "Consider time-based stops as alternative"
            ],
            ErrorType.REGIME_MISMATCH: [
                f"Reduce weight for {strategy_name} in {market_regime} regime",
                "Add regime filter to strategy",
                "Use ensemble of regime-appropriate strategies"
            ],
            ErrorType.VOLATILITY_SPIKE: [
                "Reduce position size during high volatility",
                "Widen stops to accommodate volatility",
                "Avoid trading during volatility spikes"
            ],
            ErrorType.CONFIDENCE_OVERESTIMATE: [
                "Calibrate confidence model",
                "Add uncertainty estimation",
                "Use ensemble predictions for confidence"
            ]
        }

        recommendations = recommendations_map.get(error_type, [
            "Review trade details manually",
            "Add this pattern to error detection"
        ])

        return root_cause, factors, recommendations

    def _extract_error_features(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]],
        volatility: float,
        spread_pips: float,
        signal_confidence: float,
        holding_time_ms: int,
        slippage_pips: float
    ) -> Dict[str, float]:
        """
        Extract features for ML model training.

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Basic features
        features['volatility'] = volatility
        features['spread_pips'] = spread_pips
        features['signal_confidence'] = signal_confidence
        features['holding_time_ms'] = holding_time_ms
        features['slippage_pips'] = slippage_pips

        # Price history features
        if len(price_history) > 0:
            prices = np.array(price_history)
            features['price_mean'] = float(np.mean(prices))
            features['price_std'] = float(np.std(prices))
            features['price_range'] = float(np.max(prices) - np.min(prices))

            # Returns
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                features['return_mean'] = float(np.mean(returns))
                features['return_std'] = float(np.std(returns))
                features['return_skew'] = float(self._calculate_skew(returns))
                features['return_kurt'] = float(self._calculate_kurtosis(returns))

            # Trend
            features['price_trend'] = float((prices[-1] - prices[0]) / prices[0]) if prices[0] != 0 else 0

            # Momentum
            if len(prices) >= 5:
                features['momentum_5'] = float((prices[-1] - prices[-5]) / prices[-5]) if prices[-5] != 0 else 0
            if len(prices) >= 10:
                features['momentum_10'] = float((prices[-1] - prices[-10]) / prices[-10]) if prices[-10] != 0 else 0

        # Volume features
        if volume_history and len(volume_history) > 0:
            volumes = np.array(volume_history)
            features['volume_mean'] = float(np.mean(volumes))
            features['volume_std'] = float(np.std(volumes))
            features['volume_trend'] = float((volumes[-1] - volumes[0]) / volumes[0]) if volumes[0] != 0 else 0

        return features

    def _calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.sum(((data - mean) / std) ** 3) / n)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.sum(((data - mean) / std) ** 4) / n - 3)

    def _update_statistics(self, error: TradeError):
        """Update error statistics"""
        self.total_errors += 1
        self.errors_by_type[error.error_type] += 1
        self.errors_by_strategy[error.strategy_name] += 1
        self.errors_by_regime[error.market_regime] += 1
        self.error_counts[error.strategy_name][error.error_type] += 1

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if self.total_errors == 0:
            return {
                'total_errors': 0,
                'message': 'No errors recorded yet'
            }

        # Most common errors
        top_errors = sorted(
            self.errors_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Worst performing strategies
        worst_strategies = sorted(
            self.errors_by_strategy.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Error-prone regimes
        error_regimes = sorted(
            self.errors_by_regime.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Average severity
        if self.error_history:
            avg_severity = np.mean([e.severity.value for e in self.error_history])
            avg_pnl_impact = np.mean([e.pnl_impact for e in self.error_history])
        else:
            avg_severity = 0
            avg_pnl_impact = 0

        return {
            'total_errors': self.total_errors,
            'top_error_types': [(e.name, c) for e, c in top_errors],
            'worst_strategies': worst_strategies,
            'error_prone_regimes': error_regimes,
            'average_severity': avg_severity,
            'average_pnl_impact': avg_pnl_impact,
            'error_rate_by_strategy': {
                s: {et.name: count for et, count in self.error_counts[s].items()}
                for s in self.error_counts
            }
        }

    def get_strategy_error_profile(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get detailed error profile for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with error profile
        """
        strategy_errors = [
            e for e in self.error_history
            if e.strategy_name == strategy_name
        ]

        if not strategy_errors:
            return {
                'strategy': strategy_name,
                'total_errors': 0,
                'message': 'No errors recorded for this strategy'
            }

        # Error type distribution
        error_dist = defaultdict(int)
        for e in strategy_errors:
            error_dist[e.error_type.name] += 1

        # Regime-specific errors
        regime_errors = defaultdict(int)
        for e in strategy_errors:
            regime_errors[e.market_regime] += 1

        # Average metrics
        avg_slippage = np.mean([e.slippage_pips for e in strategy_errors])
        avg_pnl_impact = np.mean([e.pnl_impact for e in strategy_errors])

        # Most common recommendations
        all_recommendations = []
        for e in strategy_errors:
            all_recommendations.extend(e.recommendations)
        rec_counts = defaultdict(int)
        for r in all_recommendations:
            rec_counts[r] += 1
        top_recommendations = sorted(
            rec_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'strategy': strategy_name,
            'total_errors': len(strategy_errors),
            'error_distribution': dict(error_dist),
            'errors_by_regime': dict(regime_errors),
            'average_slippage_pips': avg_slippage,
            'average_pnl_impact': avg_pnl_impact,
            'top_recommendations': [r for r, _ in top_recommendations]
        }

    def get_regime_error_profile(self, regime: str) -> Dict[str, Any]:
        """
        Get error profile for a specific market regime.

        Args:
            regime: Market regime name

        Returns:
            Dictionary with error profile for the regime
        """
        regime_errors = [
            e for e in self.error_history
            if e.market_regime == regime
        ]

        if not regime_errors:
            return {
                'regime': regime,
                'total_errors': 0,
                'message': f'No errors recorded for {regime} regime'
            }

        # Strategy performance in this regime
        strategy_errors = defaultdict(list)
        for e in regime_errors:
            strategy_errors[e.strategy_name].append(e)

        strategy_stats = {}
        for s, errors in strategy_errors.items():
            strategy_stats[s] = {
                'error_count': len(errors),
                'avg_pnl_impact': np.mean([e.pnl_impact for e in errors]),
                'common_error': max(
                    set(e.error_type for e in errors),
                    key=lambda x: sum(1 for e in errors if e.error_type == x)
                ).name
            }

        return {
            'regime': regime,
            'total_errors': len(regime_errors),
            'strategy_performance': strategy_stats,
            'recommended_strategies': [
                s for s, stats in sorted(
                    strategy_stats.items(),
                    key=lambda x: x[1]['error_count']
                )[:3]
            ]
        }

    def save_error_history(self, filepath: str):
        """Save error history to file"""
        data = {
            'errors': [e.to_dict() for e in self.error_history],
            'statistics': self.get_error_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Error history saved to {filepath}")

    def load_error_history(self, filepath: str):
        """Load error history from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.error_history = [
            TradeError.from_dict(e) for e in data.get('errors', [])
        ]

        # Rebuild statistics
        self.total_errors = 0
        self.errors_by_type.clear()
        self.errors_by_strategy.clear()
        self.errors_by_regime.clear()
        self.error_counts.clear()

        for error in self.error_history:
            self._update_statistics(error)

        logger.info(f"Loaded {len(self.error_history)} errors from {filepath}")

    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.total_errors = 0
        self.errors_by_type.clear()
        self.errors_by_strategy.clear()
        self.errors_by_regime.clear()
        self.error_counts.clear()
        logger.info("Error history cleared")
