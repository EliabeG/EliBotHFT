"""
Adaptive Strategy Optimizer

This module dynamically optimizes strategy parameters and weights based on:
- Error patterns and their frequency
- Market regime performance
- Real-time learning feedback
- Risk-adjusted returns

Key capabilities:
- Multi-armed bandit for strategy selection
- Thompson sampling for exploration/exploitation
- Bayesian optimization for parameter tuning
- Online gradient descent for continuous adaptation
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json

from .error_classifier import TradeError, ErrorType
from .pattern_analyzer import PatternMatch

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods available"""
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"  # Upper Confidence Bound
    EPSILON_GREEDY = "epsilon_greedy"
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN = "bayesian"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    name: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    errors: int = 0
    error_by_type: Dict[str, int] = field(default_factory=dict)

    # Bayesian parameters (Beta distribution for Thompson Sampling)
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0   # Prior failures

    # Performance by regime
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Recent performance (sliding window)
    recent_wins: int = 0
    recent_losses: int = 0
    recent_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def recent_win_rate(self) -> float:
        total = self.recent_wins + self.recent_losses
        return self.recent_wins / total if total > 0 else 0.5

    @property
    def error_rate(self) -> float:
        total = self.wins + self.losses
        return self.errors / total if total > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        total = self.wins + self.losses
        return self.total_pnl / total if total > 0 else 0.0

    def sample_thompson(self) -> float:
        """Sample from Beta distribution for Thompson Sampling"""
        return np.random.beta(self.alpha, self.beta)


@dataclass
class OptimizationResult:
    """Result of an optimization step"""
    timestamp: datetime
    strategy_weights: Dict[str, float]
    parameter_updates: Dict[str, Dict[str, float]]
    reason: str
    confidence: float
    expected_improvement: float
    risk_adjustment: float


class AdaptiveOptimizer:
    """
    Adaptive optimizer that continuously adjusts strategy weights
    and parameters based on performance and error analysis.

    Uses multiple techniques:
    1. Multi-armed bandit for strategy selection
    2. Thompson sampling for exploration/exploitation balance
    3. Gradient descent for parameter optimization
    4. Risk-adjusted scoring
    """

    def __init__(
        self,
        strategy_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.THOMPSON_SAMPLING,
        learning_rate: float = 0.1,
        exploration_bonus: float = 1.0,
        risk_aversion: float = 0.5,
        window_size: int = 100,
        min_samples_per_strategy: int = 10
    ):
        """
        Initialize the adaptive optimizer.

        Args:
            strategy_names: List of strategy names to optimize
            initial_weights: Initial weight distribution
            optimization_method: Method to use for optimization
            learning_rate: Learning rate for gradient updates
            exploration_bonus: Bonus for exploration (UCB)
            risk_aversion: How much to penalize high-variance strategies
            window_size: Size of sliding window for recent performance
            min_samples_per_strategy: Minimum samples before optimization
        """
        self.strategy_names = strategy_names
        self.optimization_method = optimization_method
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self.risk_aversion = risk_aversion
        self.window_size = window_size
        self.min_samples = min_samples_per_strategy

        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            self.weights = {name: 1.0 / len(strategy_names) for name in strategy_names}

        # Normalize weights
        self._normalize_weights()

        # Performance tracking
        self.performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(name=name)
            for name in strategy_names
        }

        # Sliding window for recent trades
        self.recent_trades: deque = deque(maxlen=window_size)

        # Regime-specific weights
        self.regime_weights: Dict[str, Dict[str, float]] = {}

        # Parameter bounds for each strategy
        self.parameter_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Current parameters
        self.current_parameters: Dict[str, Dict[str, float]] = {}

        # Optimization history
        self.optimization_history: List[OptimizationResult] = []

        # Error impact tracking
        self.error_impacts: Dict[str, Dict[ErrorType, float]] = defaultdict(lambda: defaultdict(float))

        # Callbacks
        self.on_weight_update: Optional[Callable[[Dict[str, float]], None]] = None
        self.on_parameter_update: Optional[Callable[[str, Dict[str, float]], None]] = None

        logger.info(f"AdaptiveOptimizer initialized with {len(strategy_names)} strategies")

    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def set_parameter_bounds(
        self,
        strategy_name: str,
        parameter_name: str,
        min_value: float,
        max_value: float,
        initial_value: Optional[float] = None
    ):
        """
        Set bounds for a strategy parameter.

        Args:
            strategy_name: Name of the strategy
            parameter_name: Name of the parameter
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            initial_value: Initial value (default: midpoint)
        """
        if strategy_name not in self.parameter_bounds:
            self.parameter_bounds[strategy_name] = {}
            self.current_parameters[strategy_name] = {}

        self.parameter_bounds[strategy_name][parameter_name] = (min_value, max_value)
        self.current_parameters[strategy_name][parameter_name] = (
            initial_value if initial_value is not None
            else (min_value + max_value) / 2
        )

    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        error: Optional[TradeError] = None,
        market_regime: str = "UNKNOWN"
    ):
        """
        Record the result of a trade.

        Args:
            strategy_name: Name of strategy that generated the trade
            pnl: Profit/loss from the trade
            error: TradeError if one occurred
            market_regime: Current market regime
        """
        if strategy_name not in self.performance:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return

        perf = self.performance[strategy_name]

        # Update overall performance
        is_win = pnl > 0
        if is_win:
            perf.wins += 1
            perf.alpha += 1  # Bayesian update
        else:
            perf.losses += 1
            perf.beta += 1

        perf.total_pnl += pnl

        # Update error tracking
        if error:
            perf.errors += 1
            error_name = error.error_type.name
            perf.error_by_type[error_name] = perf.error_by_type.get(error_name, 0) + 1
            self.error_impacts[strategy_name][error.error_type] += abs(error.pnl_impact)

        # Update regime-specific performance
        if market_regime not in perf.regime_performance:
            perf.regime_performance[market_regime] = {
                'wins': 0, 'losses': 0, 'pnl': 0.0
            }

        regime_perf = perf.regime_performance[market_regime]
        regime_perf['wins' if is_win else 'losses'] += 1
        regime_perf['pnl'] += pnl

        # Store in recent trades
        self.recent_trades.append({
            'strategy': strategy_name,
            'pnl': pnl,
            'is_win': is_win,
            'error': error.error_type.name if error else None,
            'regime': market_regime,
            'timestamp': datetime.now()
        })

        # Update recent performance
        self._update_recent_performance()

        # Trigger optimization if enough samples
        total_trades = sum(p.wins + p.losses for p in self.performance.values())
        if total_trades > 0 and total_trades % 10 == 0:  # Optimize every 10 trades
            self.optimize()

    def _update_recent_performance(self):
        """Update recent performance metrics from sliding window"""
        for name in self.strategy_names:
            perf = self.performance[name]
            perf.recent_wins = 0
            perf.recent_losses = 0
            perf.recent_pnl = 0.0

        for trade in self.recent_trades:
            strategy = trade['strategy']
            if strategy in self.performance:
                perf = self.performance[strategy]
                if trade['is_win']:
                    perf.recent_wins += 1
                else:
                    perf.recent_losses += 1
                perf.recent_pnl += trade['pnl']

    def optimize(self) -> OptimizationResult:
        """
        Run optimization step to update weights and parameters.

        Returns:
            OptimizationResult with changes made
        """
        # Choose optimization method
        if self.optimization_method == OptimizationMethod.THOMPSON_SAMPLING:
            new_weights, reason = self._thompson_sampling_optimize()
        elif self.optimization_method == OptimizationMethod.UCB:
            new_weights, reason = self._ucb_optimize()
        elif self.optimization_method == OptimizationMethod.EPSILON_GREEDY:
            new_weights, reason = self._epsilon_greedy_optimize()
        elif self.optimization_method == OptimizationMethod.GRADIENT_DESCENT:
            new_weights, reason = self._gradient_descent_optimize()
        else:
            new_weights, reason = self._thompson_sampling_optimize()

        # Apply risk adjustment
        new_weights = self._apply_risk_adjustment(new_weights)

        # Apply error-based penalties
        new_weights = self._apply_error_penalties(new_weights)

        # Calculate expected improvement
        expected_improvement = self._calculate_expected_improvement(new_weights)

        # Optimize parameters
        parameter_updates = self._optimize_parameters()

        # Create result
        result = OptimizationResult(
            timestamp=datetime.now(),
            strategy_weights=new_weights.copy(),
            parameter_updates=parameter_updates,
            reason=reason,
            confidence=self._calculate_optimization_confidence(),
            expected_improvement=expected_improvement,
            risk_adjustment=self.risk_aversion
        )

        # Update weights
        old_weights = self.weights.copy()
        self.weights = new_weights
        self._normalize_weights()

        # Store in history
        self.optimization_history.append(result)

        # Notify callback
        if self.on_weight_update and old_weights != self.weights:
            self.on_weight_update(self.weights)

        logger.info(
            f"Optimization complete: {reason} "
            f"(confidence={result.confidence:.2f}, "
            f"expected_improvement={expected_improvement:.4f})"
        )

        return result

    def _thompson_sampling_optimize(self) -> Tuple[Dict[str, float], str]:
        """
        Optimize using Thompson Sampling.

        Each strategy is modeled as a Beta distribution.
        Sample from each and use samples as weights.
        """
        samples = {}
        for name, perf in self.performance.items():
            samples[name] = perf.sample_thompson()

        # Normalize to get weights
        total = sum(samples.values())
        if total > 0:
            new_weights = {k: v / total for k, v in samples.items()}
        else:
            new_weights = self.weights.copy()

        return new_weights, "Thompson Sampling update"

    def _ucb_optimize(self) -> Tuple[Dict[str, float], str]:
        """
        Optimize using Upper Confidence Bound.

        UCB = mean + exploration_bonus * sqrt(log(total) / n)
        """
        total_trades = sum(p.wins + p.losses for p in self.performance.values())

        if total_trades == 0:
            return self.weights.copy(), "UCB - insufficient data"

        ucb_scores = {}
        for name, perf in self.performance.items():
            n = perf.wins + perf.losses

            if n == 0:
                ucb_scores[name] = float('inf')  # Encourage exploration
            else:
                mean = perf.win_rate
                exploration = self.exploration_bonus * np.sqrt(np.log(total_trades + 1) / n)
                ucb_scores[name] = mean + exploration

        # Convert UCB scores to weights
        total = sum(ucb_scores.values())
        if total > 0 and total != float('inf'):
            new_weights = {k: v / total for k, v in ucb_scores.items()}
        else:
            # Use uniform weights if all inf
            new_weights = {k: 1.0 / len(ucb_scores) for k in ucb_scores}

        return new_weights, "UCB update"

    def _epsilon_greedy_optimize(self, epsilon: float = 0.1) -> Tuple[Dict[str, float], str]:
        """
        Optimize using epsilon-greedy.

        With probability epsilon, explore (uniform).
        Otherwise, exploit (weight by performance).
        """
        if np.random.random() < epsilon:
            # Explore - uniform weights
            new_weights = {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}
            return new_weights, "Epsilon-greedy exploration"

        # Exploit - weight by performance
        scores = {}
        for name, perf in self.performance.items():
            # Combine win rate and average PnL
            score = perf.recent_win_rate * 0.5 + (perf.avg_pnl / 100 + 0.5) * 0.5
            scores[name] = max(0.01, score)  # Minimum weight

        total = sum(scores.values())
        new_weights = {k: v / total for k, v in scores.items()}

        return new_weights, "Epsilon-greedy exploitation"

    def _gradient_descent_optimize(self) -> Tuple[Dict[str, float], str]:
        """
        Optimize using online gradient descent.

        Update weights based on recent performance gradient.
        """
        new_weights = self.weights.copy()

        for name, perf in self.performance.items():
            # Calculate gradient (recent performance vs overall)
            if perf.wins + perf.losses < self.min_samples:
                continue

            overall_wr = perf.win_rate
            recent_wr = perf.recent_win_rate

            gradient = recent_wr - overall_wr

            # Also factor in error rate
            error_penalty = -perf.error_rate * 0.5

            # Update weight
            update = self.learning_rate * (gradient + error_penalty)
            new_weights[name] = max(0.01, new_weights[name] + update)

        # Normalize
        total = sum(new_weights.values())
        new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights, "Gradient descent update"

    def _apply_risk_adjustment(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk adjustment to weights.

        Penalize strategies with high variance/drawdown.
        """
        adjusted = {}

        for name, weight in weights.items():
            perf = self.performance[name]

            # Calculate variance proxy from recent trades
            recent_pnls = [
                t['pnl'] for t in self.recent_trades
                if t['strategy'] == name
            ]

            if len(recent_pnls) >= 2:
                variance = np.var(recent_pnls)
                # Risk-adjusted weight
                risk_penalty = self.risk_aversion * np.sqrt(variance) / 100
                adjusted[name] = max(0.01, weight - risk_penalty)
            else:
                adjusted[name] = weight

        # Normalize
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}

    def _apply_error_penalties(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply penalties based on error patterns.

        Reduce weight for strategies with high error rates.
        """
        penalized = weights.copy()

        for name in self.strategy_names:
            perf = self.performance[name]

            if perf.error_rate > 0.2:  # More than 20% error rate
                penalty = 0.5 * perf.error_rate
                penalized[name] = max(0.01, penalized[name] * (1 - penalty))

            # Extra penalty for severe errors
            severe_errors = sum(
                count for err_type, count in perf.error_by_type.items()
                if err_type in ['PREDICTION_ERROR', 'VOLATILITY_SPIKE', 'SLIPPAGE_ERROR']
            )
            if severe_errors > 5:
                penalized[name] *= 0.9

        # Normalize
        total = sum(penalized.values())
        return {k: v / total for k, v in penalized.items()}

    def _optimize_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Optimize strategy parameters using gradient-free optimization.

        Returns:
            Dictionary of strategy -> parameter -> new_value
        """
        updates = {}

        for strategy_name, params in self.current_parameters.items():
            if strategy_name not in self.parameter_bounds:
                continue

            perf = self.performance.get(strategy_name)
            if not perf or perf.wins + perf.losses < self.min_samples:
                continue

            updates[strategy_name] = {}

            for param_name, current_value in params.items():
                bounds = self.parameter_bounds[strategy_name].get(param_name)
                if not bounds:
                    continue

                min_val, max_val = bounds

                # Adjust based on error patterns
                error_rate = perf.error_rate

                # Heuristic parameter adjustments
                if param_name in ['stop_loss_pips', 'distance_to_stop']:
                    # If many premature stops, widen stop
                    stop_errors = perf.error_by_type.get('PREMATURE_STOP', 0)
                    stop_errors += perf.error_by_type.get('STOP_TOO_TIGHT', 0)

                    if stop_errors > 2:
                        # Increase stop distance
                        new_value = min(max_val, current_value * 1.1)
                        updates[strategy_name][param_name] = new_value
                        self.current_parameters[strategy_name][param_name] = new_value

                elif param_name in ['take_profit_pips', 'target_pips']:
                    # If many late exits, reduce target
                    late_exits = perf.error_by_type.get('LATE_EXIT', 0)

                    if late_exits > 2:
                        new_value = max(min_val, current_value * 0.9)
                        updates[strategy_name][param_name] = new_value
                        self.current_parameters[strategy_name][param_name] = new_value

                elif param_name in ['position_size', 'lot_size']:
                    # If high error rate, reduce size
                    if error_rate > 0.3:
                        new_value = max(min_val, current_value * 0.9)
                        updates[strategy_name][param_name] = new_value
                        self.current_parameters[strategy_name][param_name] = new_value

            # Notify callback
            if updates[strategy_name] and self.on_parameter_update:
                self.on_parameter_update(strategy_name, updates[strategy_name])

        return updates

    def _calculate_expected_improvement(self, new_weights: Dict[str, float]) -> float:
        """Calculate expected improvement from new weights"""
        old_expected = 0
        new_expected = 0

        for name in self.strategy_names:
            perf = self.performance[name]
            old_w = self.weights.get(name, 0)
            new_w = new_weights.get(name, 0)

            # Expected value is win_rate * avg_win - (1 - win_rate) * avg_loss
            # Simplified to just expected PnL
            expected_pnl = perf.avg_pnl

            old_expected += old_w * expected_pnl
            new_expected += new_w * expected_pnl

        return new_expected - old_expected

    def _calculate_optimization_confidence(self) -> float:
        """Calculate confidence in the optimization"""
        # Based on amount of data
        total_trades = sum(p.wins + p.losses for p in self.performance.values())

        if total_trades < self.min_samples * len(self.strategy_names):
            return 0.3  # Low confidence

        # Check data distribution
        min_trades = min(p.wins + p.losses for p in self.performance.values())

        if min_trades < self.min_samples:
            return 0.5

        # Higher confidence with more data
        confidence = min(0.95, 0.6 + total_trades / 1000)

        return confidence

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get optimized weights for a specific regime.

        Args:
            regime: Market regime name

        Returns:
            Weights optimized for the regime
        """
        if regime in self.regime_weights:
            return self.regime_weights[regime]

        # Calculate from performance data
        regime_scores = {}

        for name, perf in self.performance.items():
            if regime in perf.regime_performance:
                rp = perf.regime_performance[regime]
                total = rp['wins'] + rp['losses']
                if total > 0:
                    win_rate = rp['wins'] / total
                    avg_pnl = rp['pnl'] / total
                    # Combined score
                    regime_scores[name] = win_rate * 0.6 + (avg_pnl / 100 + 0.5) * 0.4
                else:
                    regime_scores[name] = 0.5
            else:
                # Use overall performance
                regime_scores[name] = self.weights.get(name, 0.5)

        # Normalize
        total = sum(regime_scores.values())
        if total > 0:
            regime_weights = {k: v / total for k, v in regime_scores.items()}
        else:
            regime_weights = self.weights.copy()

        self.regime_weights[regime] = regime_weights
        return regime_weights

    def get_strategy_recommendation(
        self,
        current_regime: str,
        risk_level: str,
        pattern_matches: Optional[List[PatternMatch]] = None
    ) -> Tuple[str, float]:
        """
        Get recommended strategy for current conditions.

        Args:
            current_regime: Current market regime
            risk_level: Current risk level
            pattern_matches: Pattern matches from analyzer

        Returns:
            Tuple of (strategy_name, confidence)
        """
        # Get regime-specific weights
        weights = self.get_regime_weights(current_regime)

        # Adjust for risk level
        if risk_level in ['HIGH', 'CRITICAL']:
            # Favor conservative strategies
            conservative = ['mean_reversion', 'market_making']
            for strategy in conservative:
                if strategy in weights:
                    weights[strategy] *= 1.2

        # Adjust for pattern matches
        if pattern_matches:
            for match in pattern_matches[:3]:  # Top 3 patterns
                # Reduce weight for affected strategies
                for strategy in match.pattern.affected_strategies:
                    if strategy in weights:
                        weights[strategy] *= (1 - match.overall_risk * 0.3)

        # Normalize
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Select best strategy
        best_strategy = max(weights, key=weights.get)
        confidence = weights[best_strategy]

        return best_strategy, confidence

    def get_position_size_multiplier(
        self,
        strategy_name: str,
        error_risk: float
    ) -> float:
        """
        Get position size multiplier based on strategy performance and error risk.

        Args:
            strategy_name: Name of strategy
            error_risk: Predicted error risk (0-1)

        Returns:
            Multiplier for position size (0.1 - 1.5)
        """
        perf = self.performance.get(strategy_name)

        if not perf or perf.wins + perf.losses < self.min_samples:
            return 1.0  # Default multiplier

        # Base multiplier from performance
        win_rate = perf.win_rate
        base_multiplier = 0.5 + win_rate  # 0.5 to 1.5

        # Adjust for error risk
        risk_adjustment = 1 - error_risk * 0.5  # 0.5 to 1.0

        # Adjust for error rate
        error_adjustment = 1 - perf.error_rate * 0.5  # 0.5 to 1.0

        # Combined multiplier
        multiplier = base_multiplier * risk_adjustment * error_adjustment

        # Clamp to valid range
        return max(0.1, min(1.5, multiplier))

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            'current_weights': self.weights,
            'optimization_count': len(self.optimization_history),
            'strategy_performance': {
                name: {
                    'wins': perf.wins,
                    'losses': perf.losses,
                    'win_rate': perf.win_rate,
                    'recent_win_rate': perf.recent_win_rate,
                    'error_rate': perf.error_rate,
                    'avg_pnl': perf.avg_pnl,
                    'total_pnl': perf.total_pnl,
                }
                for name, perf in self.performance.items()
            },
            'regime_weights': self.regime_weights,
            'total_trades': sum(p.wins + p.losses for p in self.performance.values()),
        }

    def save_state(self, filepath: str):
        """Save optimizer state"""
        state = {
            'weights': self.weights,
            'performance': {
                name: {
                    'wins': p.wins,
                    'losses': p.losses,
                    'total_pnl': p.total_pnl,
                    'errors': p.errors,
                    'error_by_type': p.error_by_type,
                    'alpha': p.alpha,
                    'beta': p.beta,
                    'regime_performance': p.regime_performance,
                }
                for name, p in self.performance.items()
            },
            'regime_weights': self.regime_weights,
            'current_parameters': self.current_parameters,
            'parameter_bounds': self.parameter_bounds,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Optimizer state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load optimizer state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.weights = state.get('weights', self.weights)
        self.regime_weights = state.get('regime_weights', {})
        self.current_parameters = state.get('current_parameters', {})
        self.parameter_bounds = state.get('parameter_bounds', {})

        for name, pdata in state.get('performance', {}).items():
            if name in self.performance:
                p = self.performance[name]
                p.wins = pdata.get('wins', 0)
                p.losses = pdata.get('losses', 0)
                p.total_pnl = pdata.get('total_pnl', 0.0)
                p.errors = pdata.get('errors', 0)
                p.error_by_type = pdata.get('error_by_type', {})
                p.alpha = pdata.get('alpha', 1.0)
                p.beta = pdata.get('beta', 1.0)
                p.regime_performance = pdata.get('regime_performance', {})

        logger.info(f"Optimizer state loaded from {filepath}")

    def reset(self):
        """Reset optimizer to initial state"""
        self.weights = {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}

        for perf in self.performance.values():
            perf.wins = 0
            perf.losses = 0
            perf.total_pnl = 0.0
            perf.errors = 0
            perf.error_by_type.clear()
            perf.alpha = 1.0
            perf.beta = 1.0
            perf.regime_performance.clear()
            perf.recent_wins = 0
            perf.recent_losses = 0
            perf.recent_pnl = 0.0

        self.recent_trades.clear()
        self.regime_weights.clear()
        self.optimization_history.clear()
        self.error_impacts.clear()

        logger.info("AdaptiveOptimizer reset")
