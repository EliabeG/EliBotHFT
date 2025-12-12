"""
Error Pattern Analyzer

This module identifies patterns in trading errors using machine learning
techniques. It detects recurring error patterns and provides predictions
for future error likelihood.

Key capabilities:
- Pattern detection using clustering and sequence analysis
- Error prediction using classification models
- Similarity matching for known error patterns
- Anomaly detection for unusual market conditions
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .error_classifier import TradeError, ErrorType, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents a detected error pattern"""

    pattern_id: str
    name: str
    description: str

    # Pattern characteristics
    error_types: List[ErrorType]
    common_features: Dict[str, Tuple[float, float]]  # Feature -> (mean, std)
    market_conditions: Dict[str, str]  # Regime, volatility level, etc.

    # Statistics
    occurrence_count: int
    avg_pnl_impact: float
    avg_severity: float

    # Associated strategies
    affected_strategies: Dict[str, int]  # Strategy -> count

    # Time patterns
    hour_distribution: Dict[int, int]
    day_distribution: Dict[int, int]

    # Metadata
    first_seen: datetime
    last_seen: datetime
    confidence: float  # How confident are we this is a real pattern

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'error_types': [e.name for e in self.error_types],
            'common_features': self.common_features,
            'market_conditions': self.market_conditions,
            'occurrence_count': self.occurrence_count,
            'avg_pnl_impact': self.avg_pnl_impact,
            'avg_severity': self.avg_severity,
            'affected_strategies': self.affected_strategies,
            'hour_distribution': self.hour_distribution,
            'day_distribution': self.day_distribution,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class PatternMatch:
    """Result of matching current conditions to known patterns"""

    pattern: ErrorPattern
    similarity: float  # 0-1 similarity score
    matching_features: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    predicted_error_types: List[Tuple[ErrorType, float]]  # Type, probability
    overall_risk: float  # 0-1 risk score


class PatternAnalyzer:
    """
    Analyzes trading errors to detect patterns and predict future errors.

    Uses a combination of:
    1. Rule-based pattern matching
    2. Clustering for pattern discovery
    3. Classification for error prediction
    4. Sequence analysis for temporal patterns
    """

    def __init__(
        self,
        min_pattern_occurrences: int = 5,
        similarity_threshold: float = 0.7,
        feature_importance_threshold: float = 0.3,
        max_patterns: int = 100
    ):
        """
        Initialize the pattern analyzer.

        Args:
            min_pattern_occurrences: Minimum errors to form a pattern
            similarity_threshold: Minimum similarity for pattern matching
            feature_importance_threshold: Threshold for feature significance
            max_patterns: Maximum number of patterns to maintain
        """
        self.min_occurrences = min_pattern_occurrences
        self.similarity_threshold = similarity_threshold
        self.feature_threshold = feature_importance_threshold
        self.max_patterns = max_patterns

        # Known patterns
        self.patterns: Dict[str, ErrorPattern] = {}

        # Error history for analysis
        self.error_buffer: List[TradeError] = []
        self.max_buffer_size = 10000

        # Feature statistics (for normalization)
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        # Cluster centroids (for pattern discovery)
        self.cluster_centroids: List[np.ndarray] = []
        self.cluster_labels: Dict[int, List[TradeError]] = defaultdict(list)

        # Sequence patterns
        self.error_sequences: List[List[ErrorType]] = []
        self.sequence_probabilities: Dict[Tuple[ErrorType, ...], float] = {}

        # Initialize built-in patterns
        self._initialize_builtin_patterns()

        logger.info("PatternAnalyzer initialized")

    def _initialize_builtin_patterns(self):
        """Initialize known error patterns based on trading knowledge"""

        # Pattern 1: High Volatility Slippage
        self.patterns['HV_SLIPPAGE'] = ErrorPattern(
            pattern_id='HV_SLIPPAGE',
            name='High Volatility Slippage',
            description='Excessive slippage during volatile market conditions',
            error_types=[ErrorType.SLIPPAGE_ERROR, ErrorType.VOLATILITY_SPIKE],
            common_features={
                'volatility_1m': (0.015, 0.005),
                'spread_pips': (2.0, 0.5),
                'entry_slippage_pips': (3.0, 1.0)
            },
            market_conditions={'regime': 'VOLATILE', 'liquidity': 'LOW'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=3.0,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.8
        )

        # Pattern 2: Stop Hunt / Premature Stop
        self.patterns['STOP_HUNT'] = ErrorPattern(
            pattern_id='STOP_HUNT',
            name='Stop Hunt Pattern',
            description='Stop loss triggered followed by price reversal',
            error_types=[ErrorType.PREMATURE_STOP, ErrorType.STOP_TOO_TIGHT],
            common_features={
                'distance_to_stop_pips': (5.0, 2.0),
                'volatility_ratio': (1.5, 0.3),
                'regime_confidence': (0.5, 0.2)
            },
            market_conditions={'regime': 'RANGING', 'time': 'SESSION_OVERLAP'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=3.5,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.75
        )

        # Pattern 3: False Breakout
        self.patterns['FALSE_BO'] = ErrorPattern(
            pattern_id='FALSE_BO',
            name='False Breakout',
            description='Breakout signal that reverses quickly',
            error_types=[ErrorType.FALSE_BREAKOUT, ErrorType.PREDICTION_ERROR],
            common_features={
                'signal_confidence': (0.7, 0.1),
                'momentum_5m': (0.002, 0.001),
                'book_imbalance': (0.3, 0.15)
            },
            market_conditions={'regime': 'RANGING', 'volume': 'LOW'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=3.0,
            affected_strategies={'breakout': 0},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.7
        )

        # Pattern 4: Regime Mismatch
        self.patterns['REGIME_MISMATCH'] = ErrorPattern(
            pattern_id='REGIME_MISMATCH',
            name='Strategy-Regime Mismatch',
            description='Strategy not suited for current market regime',
            error_types=[ErrorType.REGIME_MISMATCH, ErrorType.PREDICTION_ERROR],
            common_features={
                'regime_win_rate': (0.3, 0.1),
                'trend_strength': (0.3, 0.2)
            },
            market_conditions={'regime': 'TRANSITIONING'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=2.5,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.65
        )

        # Pattern 5: Overconfidence
        self.patterns['OVERCONF'] = ErrorPattern(
            pattern_id='OVERCONF',
            name='Signal Overconfidence',
            description='High confidence signal that resulted in loss',
            error_types=[ErrorType.CONFIDENCE_OVERESTIMATE, ErrorType.PREDICTION_ERROR],
            common_features={
                'signal_confidence': (0.9, 0.05),
                'num_confirming_strategies': (1.0, 0.5),
                'consecutive_wins': (3.0, 1.0)
            },
            market_conditions={},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=2.0,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.6
        )

        # Pattern 6: News Event Impact
        self.patterns['NEWS_IMPACT'] = ErrorPattern(
            pattern_id='NEWS_IMPACT',
            name='News Event Impact',
            description='Sudden market move likely due to news',
            error_types=[ErrorType.VOLATILITY_SPIKE, ErrorType.SLIPPAGE_ERROR, ErrorType.SPREAD_WIDENING],
            common_features={
                'volatility_ratio': (3.0, 1.0),
                'spread_pips': (5.0, 2.0),
                'price_range_1m': (20.0, 10.0)
            },
            market_conditions={'regime': 'VOLATILE', 'volatility': 'EXTREME'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=4.0,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.85
        )

        # Pattern 7: Liquidity Gap
        self.patterns['LIQ_GAP'] = ErrorPattern(
            pattern_id='LIQ_GAP',
            name='Liquidity Gap',
            description='Poor execution due to thin order book',
            error_types=[ErrorType.LIQUIDITY_GAP, ErrorType.SLIPPAGE_ERROR, ErrorType.PARTIAL_FILL],
            common_features={
                'bid_depth': (0.1, 0.05),
                'ask_depth': (0.1, 0.05),
                'spread_pips': (3.0, 1.0)
            },
            market_conditions={'liquidity': 'VERY_LOW', 'time': 'OFF_HOURS'},
            occurrence_count=0,
            avg_pnl_impact=0,
            avg_severity=3.0,
            affected_strategies={},
            hour_distribution={h: 0 for h in range(24)},
            day_distribution={d: 0 for d in range(7)},
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.75
        )

        logger.info(f"Initialized {len(self.patterns)} built-in patterns")

    def add_error(self, error: TradeError):
        """
        Add a new error to the buffer for analysis.

        Args:
            error: TradeError to analyze
        """
        self.error_buffer.append(error)

        # Maintain buffer size
        if len(self.error_buffer) > self.max_buffer_size:
            self.error_buffer = self.error_buffer[-self.max_buffer_size:]

        # Update pattern statistics
        self._update_pattern_matches(error)

        # Update feature statistics
        self._update_feature_stats(error)

        # Check for new patterns periodically
        if len(self.error_buffer) % 50 == 0:
            self._discover_new_patterns()

        # Update sequence analysis
        self._update_sequences(error)

    def _update_pattern_matches(self, error: TradeError):
        """Update pattern statistics with new error"""
        for pattern_id, pattern in self.patterns.items():
            if self._matches_pattern(error, pattern):
                pattern.occurrence_count += 1
                pattern.avg_pnl_impact = (
                    (pattern.avg_pnl_impact * (pattern.occurrence_count - 1) + error.pnl_impact)
                    / pattern.occurrence_count
                )
                pattern.avg_severity = (
                    (pattern.avg_severity * (pattern.occurrence_count - 1) + error.severity.value)
                    / pattern.occurrence_count
                )
                pattern.last_seen = error.timestamp

                # Update strategy counts
                if error.strategy_name in pattern.affected_strategies:
                    pattern.affected_strategies[error.strategy_name] += 1
                else:
                    pattern.affected_strategies[error.strategy_name] = 1

                # Update time distributions
                hour = error.timestamp.hour
                day = error.timestamp.weekday()
                pattern.hour_distribution[hour] = pattern.hour_distribution.get(hour, 0) + 1
                pattern.day_distribution[day] = pattern.day_distribution.get(day, 0) + 1

    def _matches_pattern(self, error: TradeError, pattern: ErrorPattern) -> bool:
        """Check if error matches a pattern"""
        # Check error type
        if error.error_type not in pattern.error_types:
            return False

        # Check common features
        match_count = 0
        total_features = len(pattern.common_features)

        for feature_name, (mean, std) in pattern.common_features.items():
            if feature_name in error.features:
                value = error.features[feature_name]
                # Check if within 2 standard deviations
                if abs(value - mean) <= 2 * std:
                    match_count += 1

        # Require at least 60% feature match
        if total_features > 0 and match_count / total_features < 0.6:
            return False

        return True

    def _update_feature_stats(self, error: TradeError):
        """Update running statistics for features"""
        for feature_name, value in error.features.items():
            if feature_name not in self.feature_means:
                self.feature_means[feature_name] = value
                self.feature_stds[feature_name] = 0.0
            else:
                # Running mean and std
                n = len(self.error_buffer)
                old_mean = self.feature_means[feature_name]
                self.feature_means[feature_name] = old_mean + (value - old_mean) / n

                if n > 1:
                    # Welford's algorithm for variance
                    old_std = self.feature_stds[feature_name]
                    variance = ((n - 2) * old_std ** 2 + (value - old_mean) * (value - self.feature_means[feature_name])) / (n - 1)
                    # Evitar sqrt de número negativo devido a imprecisão de ponto flutuante
                    self.feature_stds[feature_name] = np.sqrt(max(0, variance))

    def _update_sequences(self, error: TradeError):
        """Update error sequence analysis"""
        # Keep track of error type sequences
        if len(self.error_buffer) >= 2:
            recent_types = [e.error_type for e in self.error_buffer[-5:]]
            if len(recent_types) >= 2:
                # Record 2-grams
                for i in range(len(recent_types) - 1):
                    seq = (recent_types[i], recent_types[i + 1])
                    self.sequence_probabilities[seq] = self.sequence_probabilities.get(seq, 0) + 1

                # Record 3-grams
                if len(recent_types) >= 3:
                    for i in range(len(recent_types) - 2):
                        seq = (recent_types[i], recent_types[i + 1], recent_types[i + 2])
                        self.sequence_probabilities[seq] = self.sequence_probabilities.get(seq, 0) + 1

    def _discover_new_patterns(self):
        """Discover new patterns from error buffer using clustering"""
        if len(self.error_buffer) < self.min_occurrences * 2:
            return

        # Group errors by type
        errors_by_type: Dict[ErrorType, List[TradeError]] = defaultdict(list)
        for error in self.error_buffer:
            errors_by_type[error.error_type].append(error)

        # Look for patterns within each error type
        for error_type, errors in errors_by_type.items():
            if len(errors) < self.min_occurrences:
                continue

            # Extract feature vectors
            feature_vectors = []
            common_features = set(errors[0].features.keys())
            for error in errors:
                common_features &= set(error.features.keys())

            if not common_features:
                continue

            feature_names = sorted(common_features)

            for error in errors:
                vec = [error.features.get(f, 0) for f in feature_names]
                feature_vectors.append(vec)

            feature_matrix = np.array(feature_vectors)

            # Normalize features
            means = np.mean(feature_matrix, axis=0)
            stds = np.std(feature_matrix, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
            normalized = (feature_matrix - means) / stds

            # Simple clustering: find dense regions
            clusters = self._simple_cluster(normalized, min_cluster_size=self.min_occurrences)

            # Create patterns from significant clusters
            for cluster_indices in clusters:
                if len(cluster_indices) >= self.min_occurrences:
                    self._create_pattern_from_cluster(
                        [errors[i] for i in cluster_indices],
                        feature_names
                    )

    def _simple_cluster(self, data: np.ndarray, min_cluster_size: int) -> List[List[int]]:
        """
        Simple distance-based clustering.
        Returns list of index lists for each cluster.
        """
        if len(data) < min_cluster_size:
            return []

        clusters = []
        used = set()

        for i in range(len(data)):
            if i in used:
                continue

            # Find neighbors within threshold
            cluster = [i]
            for j in range(i + 1, len(data)):
                if j in used:
                    continue

                dist = np.linalg.norm(data[i] - data[j])
                if dist < 2.0:  # Within 2 std in normalized space
                    cluster.append(j)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                used.update(cluster)

        return clusters

    def _create_pattern_from_cluster(self, errors: List[TradeError], feature_names: List[str]):
        """Create a new pattern from a cluster of errors"""
        # Generate pattern ID
        pattern_id = f"DISCOVERED_{len(self.patterns)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Check if similar pattern exists
        for existing in self.patterns.values():
            if self._patterns_similar(errors, existing):
                return  # Don't create duplicate

        # Calculate common features
        common_features = {}
        for feature in feature_names:
            values = [e.features.get(feature, 0) for e in errors]
            common_features[feature] = (float(np.mean(values)), float(np.std(values)))

        # Get error types
        error_types = list(set(e.error_type for e in errors))

        # Get affected strategies
        strategy_counts = defaultdict(int)
        for e in errors:
            strategy_counts[e.strategy_name] += 1

        # Get time distributions
        hour_dist = defaultdict(int)
        day_dist = defaultdict(int)
        for e in errors:
            hour_dist[e.timestamp.hour] += 1
            day_dist[e.timestamp.weekday()] += 1

        # Determine market conditions
        regimes = [e.market_regime for e in errors]
        most_common_regime = max(set(regimes), key=regimes.count) if regimes else "UNKNOWN"

        # Create pattern
        pattern = ErrorPattern(
            pattern_id=pattern_id,
            name=f"Discovered Pattern ({error_types[0].name})",
            description=f"Automatically discovered pattern from {len(errors)} errors",
            error_types=error_types,
            common_features=common_features,
            market_conditions={'regime': most_common_regime},
            occurrence_count=len(errors),
            avg_pnl_impact=float(np.mean([e.pnl_impact for e in errors])),
            avg_severity=float(np.mean([e.severity.value for e in errors])),
            affected_strategies=dict(strategy_counts),
            hour_distribution=dict(hour_dist),
            day_distribution=dict(day_dist),
            first_seen=min(e.timestamp for e in errors),
            last_seen=max(e.timestamp for e in errors),
            confidence=min(0.9, len(errors) / 20)  # More errors = higher confidence
        )

        self.patterns[pattern_id] = pattern
        logger.info(f"Discovered new pattern: {pattern_id} ({len(errors)} errors)")

        # Maintain max patterns limit
        if len(self.patterns) > self.max_patterns:
            self._prune_patterns()

    def _patterns_similar(self, errors: List[TradeError], pattern: ErrorPattern) -> bool:
        """Check if a cluster of errors is similar to an existing pattern"""
        # Check error type overlap
        error_types = set(e.error_type for e in errors)
        pattern_types = set(pattern.error_types)

        if not error_types & pattern_types:
            return False

        # Check feature similarity
        for feature, (mean, std) in pattern.common_features.items():
            values = [e.features.get(feature, 0) for e in errors if feature in e.features]
            if values:
                cluster_mean = np.mean(values)
                if abs(cluster_mean - mean) < 2 * std:
                    return True

        return False

    def _prune_patterns(self):
        """Remove least useful patterns to maintain limit"""
        # Score patterns by usefulness
        pattern_scores = []
        for pattern_id, pattern in self.patterns.items():
            if pattern_id.startswith('DISCOVERED_'):
                # Score based on occurrence and recency
                recency = (datetime.now() - pattern.last_seen).days
                score = pattern.occurrence_count / (1 + recency)
                pattern_scores.append((pattern_id, score))

        # Sort by score and remove lowest
        pattern_scores.sort(key=lambda x: x[1])
        to_remove = len(self.patterns) - self.max_patterns

        for i in range(min(to_remove, len(pattern_scores))):
            pattern_id = pattern_scores[i][0]
            del self.patterns[pattern_id]
            logger.info(f"Pruned pattern: {pattern_id}")

    def analyze_current_conditions(
        self,
        features: Dict[str, float],
        strategy_name: str,
        market_regime: str
    ) -> List[PatternMatch]:
        """
        Analyze current market conditions against known patterns.

        Args:
            features: Current feature values
            strategy_name: Strategy being considered
            market_regime: Current market regime

        Returns:
            List of pattern matches sorted by risk
        """
        matches = []

        for pattern_id, pattern in self.patterns.items():
            similarity, matching_features = self._calculate_pattern_similarity(
                features, pattern
            )

            if similarity >= self.similarity_threshold:
                # Calculate risk factors
                risk_factors = self._identify_risk_factors(
                    features, pattern, strategy_name, market_regime
                )

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    pattern, strategy_name, risk_factors
                )

                # Predict error types
                predicted_errors = self._predict_error_types(pattern, features)

                # Calculate overall risk
                overall_risk = self._calculate_overall_risk(
                    similarity, pattern, predicted_errors
                )

                match = PatternMatch(
                    pattern=pattern,
                    similarity=similarity,
                    matching_features=matching_features,
                    risk_factors=risk_factors,
                    recommendations=recommendations,
                    predicted_error_types=predicted_errors,
                    overall_risk=overall_risk
                )
                matches.append(match)

        # Sort by risk (highest first)
        matches.sort(key=lambda x: x.overall_risk, reverse=True)

        return matches

    def _calculate_pattern_similarity(
        self,
        features: Dict[str, float],
        pattern: ErrorPattern
    ) -> Tuple[float, List[str]]:
        """Calculate similarity between current features and pattern"""
        matching_features = []
        total_weight = 0
        weighted_match = 0

        for feature_name, (mean, std) in pattern.common_features.items():
            if feature_name in features:
                value = features[feature_name]

                # Calculate z-score
                z = abs(value - mean) / std if std > 0 else 0

                # Weight by how well it matches (inverse z-score)
                weight = 1.0 / (1.0 + z)
                total_weight += 1
                weighted_match += weight

                if z < 2:  # Within 2 std
                    matching_features.append(feature_name)

        similarity = weighted_match / total_weight if total_weight > 0 else 0

        return similarity, matching_features

    def _identify_risk_factors(
        self,
        features: Dict[str, float],
        pattern: ErrorPattern,
        strategy_name: str,
        market_regime: str
    ) -> List[str]:
        """Identify current risk factors based on pattern"""
        risk_factors = []

        # Check if strategy is in affected list
        if strategy_name in pattern.affected_strategies:
            risk_factors.append(
                f"Strategy '{strategy_name}' historically affected by this pattern"
            )

        # Check regime match
        if pattern.market_conditions.get('regime') == market_regime:
            risk_factors.append(f"Current regime ({market_regime}) matches pattern")

        # Check time-based risks
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()

        if pattern.hour_distribution.get(current_hour, 0) > pattern.occurrence_count / 12:
            risk_factors.append(f"Current hour ({current_hour}) is high-risk for this pattern")

        if pattern.day_distribution.get(current_day, 0) > pattern.occurrence_count / 4:
            risk_factors.append(f"Current day is high-risk for this pattern")

        # Check feature-specific risks
        for feature_name, (mean, std) in pattern.common_features.items():
            if feature_name in features:
                value = features[feature_name]
                z = abs(value - mean) / std if std > 0 else 0

                if z < 1:  # Very close match
                    if 'volatility' in feature_name.lower():
                        risk_factors.append(f"Volatility matches error pattern")
                    elif 'spread' in feature_name.lower():
                        risk_factors.append(f"Spread conditions match error pattern")
                    elif 'slippage' in feature_name.lower():
                        risk_factors.append(f"Slippage risk elevated")

        return risk_factors

    def _generate_recommendations(
        self,
        pattern: ErrorPattern,
        strategy_name: str,
        risk_factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on pattern match"""
        recommendations = []

        # Pattern-specific recommendations
        if pattern.pattern_id == 'HV_SLIPPAGE':
            recommendations.extend([
                "Reduce position size during high volatility",
                "Use limit orders instead of market orders",
                "Widen slippage tolerance or avoid trading"
            ])
        elif pattern.pattern_id == 'STOP_HUNT':
            recommendations.extend([
                "Widen stop loss based on ATR",
                "Use time-based stops as alternative",
                "Avoid round number stop levels"
            ])
        elif pattern.pattern_id == 'FALSE_BO':
            recommendations.extend([
                "Wait for breakout confirmation",
                "Require volume confirmation",
                "Use smaller position for initial entry"
            ])
        elif pattern.pattern_id == 'REGIME_MISMATCH':
            recommendations.extend([
                f"Consider reducing weight for {strategy_name}",
                "Switch to regime-appropriate strategy",
                "Wait for clearer regime signal"
            ])
        elif pattern.pattern_id == 'OVERCONF':
            recommendations.extend([
                "Require additional confirmation signals",
                "Cap position size regardless of confidence",
                "Review signal calibration"
            ])
        elif pattern.pattern_id == 'NEWS_IMPACT':
            recommendations.extend([
                "Avoid trading during scheduled news events",
                "Reduce exposure before announcements",
                "Use wider stops if trading"
            ])
        elif pattern.pattern_id == 'LIQ_GAP':
            recommendations.extend([
                "Trade during peak liquidity hours only",
                "Reduce position size significantly",
                "Use limit orders with patience"
            ])
        else:
            # Generic recommendations based on pattern severity
            if pattern.avg_severity >= 3:
                recommendations.append("High-severity pattern - consider avoiding trade")
            recommendations.append("Monitor trade closely if entering")
            recommendations.append("Use reduced position size")

        return recommendations

    def _predict_error_types(
        self,
        pattern: ErrorPattern,
        features: Dict[str, float]
    ) -> List[Tuple[ErrorType, float]]:
        """Predict most likely error types"""
        predictions = []

        # Weight by pattern occurrence and feature match
        for error_type in pattern.error_types:
            # Base probability from pattern
            base_prob = 1.0 / len(pattern.error_types)

            # Adjust based on feature similarity
            feature_prob = 0.5
            for feature_name, (mean, std) in pattern.common_features.items():
                if feature_name in features:
                    z = abs(features[feature_name] - mean) / std if std > 0 else 0
                    feature_prob += 0.1 * (1 / (1 + z))

            prob = min(0.95, base_prob * feature_prob * (pattern.confidence))
            predictions.append((error_type, prob))

        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def _calculate_overall_risk(
        self,
        similarity: float,
        pattern: ErrorPattern,
        predicted_errors: List[Tuple[ErrorType, float]]
    ) -> float:
        """Calculate overall risk score (0-1)"""
        # Base risk from similarity
        risk = similarity * 0.4

        # Add severity component
        risk += (pattern.avg_severity / 5.0) * 0.3

        # Add prediction confidence component
        if predicted_errors:
            max_prob = max(p for _, p in predicted_errors)
            risk += max_prob * 0.3

        # Clamp to [0, 1]
        return min(1.0, max(0.0, risk))

    def predict_next_error(self, recent_errors: List[TradeError]) -> Optional[Tuple[ErrorType, float]]:
        """
        Predict the next likely error type based on sequence analysis.

        Args:
            recent_errors: List of recent errors

        Returns:
            Tuple of (predicted_type, probability) or None
        """
        if len(recent_errors) < 2:
            return None

        recent_types = tuple(e.error_type for e in recent_errors[-2:])

        # Look for matching sequences
        candidates = []
        total = sum(self.sequence_probabilities.values()) or 1

        for seq, count in self.sequence_probabilities.items():
            if len(seq) >= 2 and seq[:-1] == recent_types[-len(seq) + 1:]:
                prob = count / total
                candidates.append((seq[-1], prob))

        if candidates:
            # Return highest probability prediction
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]

        return None

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all patterns"""
        return {
            'total_patterns': len(self.patterns),
            'built_in_patterns': sum(1 for p in self.patterns if not p.startswith('DISCOVERED_')),
            'discovered_patterns': sum(1 for p in self.patterns if p.startswith('DISCOVERED_')),
            'total_errors_analyzed': len(self.error_buffer),
            'patterns': {
                pid: {
                    'name': p.name,
                    'occurrences': p.occurrence_count,
                    'avg_impact': p.avg_pnl_impact,
                    'avg_severity': p.avg_severity,
                    'confidence': p.confidence
                }
                for pid, p in self.patterns.items()
            }
        }

    def save_patterns(self, filepath: str):
        """Save patterns to file"""
        data = {
            'patterns': {pid: p.to_dict() for pid, p in self.patterns.items()},
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'sequence_probabilities': {
                str(k): v for k, v in self.sequence_probabilities.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Patterns saved to {filepath}")

    def load_patterns(self, filepath: str):
        """Load patterns from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.feature_means = data.get('feature_means', {})
        self.feature_stds = data.get('feature_stds', {})

        # Reconstruct patterns
        for pid, pdata in data.get('patterns', {}).items():
            self.patterns[pid] = ErrorPattern(
                pattern_id=pdata['pattern_id'],
                name=pdata['name'],
                description=pdata['description'],
                error_types=[ErrorType[e] for e in pdata['error_types']],
                common_features=pdata['common_features'],
                market_conditions=pdata['market_conditions'],
                occurrence_count=pdata['occurrence_count'],
                avg_pnl_impact=pdata['avg_pnl_impact'],
                avg_severity=pdata['avg_severity'],
                affected_strategies=pdata['affected_strategies'],
                hour_distribution={int(k): v for k, v in pdata['hour_distribution'].items()},
                day_distribution={int(k): v for k, v in pdata['day_distribution'].items()},
                first_seen=datetime.fromisoformat(pdata['first_seen']),
                last_seen=datetime.fromisoformat(pdata['last_seen']),
                confidence=pdata['confidence']
            )

        logger.info(f"Loaded {len(self.patterns)} patterns from {filepath}")
