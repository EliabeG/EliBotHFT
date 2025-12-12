"""
Online Learning and Concept Drift Detection

This module implements advanced online learning capabilities:
1. Incremental Learning algorithms
2. Concept Drift Detection (ADWIN, DDM, EDDM, Page-Hinkley)
3. Adaptive Window Management
4. Stream Processing for real-time data
5. Forgetting mechanisms for non-stationary data
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import json
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of concept drift"""
    NONE = auto()
    SUDDEN = auto()      # Abrupt change
    GRADUAL = auto()     # Smooth transition
    INCREMENTAL = auto() # Small continuous changes
    RECURRING = auto()   # Periodic patterns
    OUTLIER = auto()     # Temporary anomaly


@dataclass
class DriftDetectionResult:
    """Result from drift detection"""
    drift_detected: bool
    drift_type: DriftType
    confidence: float
    warning_level: bool
    statistics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class BaseDriftDetector(ABC):
    """Base class for drift detectors"""

    @abstractmethod
    def update(self, value: float) -> DriftDetectionResult:
        """Update detector with new value"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state"""
        pass


class ADWIN:
    """
    ADaptive WINdowing (ADWIN) algorithm for drift detection.
    Automatically adjusts window size based on data distribution.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5, min_window: int = 10):
        """
        Initialize ADWIN detector.

        Args:
            delta: Confidence parameter (lower = more sensitive)
            max_buckets: Maximum buckets per level
            min_window: Minimum window size
        """
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window

        self.reset()

        logger.info(f"ADWIN initialized with delta={delta}")

    def reset(self):
        """Reset detector state"""
        self.window = deque()
        self.sum = 0.0
        self.variance = 0.0
        self.width = 0
        self.bucket_list = []
        self.total = 0

    def update(self, value: float) -> DriftDetectionResult:
        """
        Update ADWIN with new value.

        Args:
            value: New observation

        Returns:
            DriftDetectionResult
        """
        self._insert_element(value)
        drift_detected = self._detect_change()

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.SUDDEN if drift_detected else DriftType.NONE,
            confidence=1.0 - self.delta if drift_detected else 0.0,
            warning_level=self.width > self.min_window * 2,
            statistics={
                'window_size': self.width,
                'mean': self.sum / self.width if self.width > 0 else 0,
                'total_samples': self.total
            }
        )

    def _insert_element(self, value: float):
        """Insert new element into window"""
        self.window.append(value)
        self.width += 1
        self.total += 1
        self.sum += value

        # Update variance using Welford's algorithm
        if self.width > 1:
            delta = value - (self.sum / self.width)
            self.variance += delta * delta * (self.width - 1) / self.width

    def _detect_change(self) -> bool:
        """Detect if there's a significant change"""
        if self.width < self.min_window * 2:
            return False

        changed = False

        # Try different split points
        for split in range(self.min_window, self.width - self.min_window):
            # Calculate statistics for two windows
            w0 = list(self.window)[:split]
            w1 = list(self.window)[split:]

            n0, n1 = len(w0), len(w1)
            if n0 < self.min_window or n1 < self.min_window:
                continue

            m0 = np.mean(w0)
            m1 = np.mean(w1)

            # Calculate cut threshold
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt(2.0 / m * np.log(2.0 / self.delta))

            if abs(m0 - m1) > epsilon:
                # Drift detected - shrink window
                for _ in range(split):
                    if self.window:
                        removed = self.window.popleft()
                        self.sum -= removed
                        self.width -= 1
                changed = True
                break

        return changed


class DDM:
    """
    Drift Detection Method (DDM).
    Uses error rate statistics to detect drift.
    """

    def __init__(
        self,
        min_samples: int = 30,
        warning_level: float = 2.0,
        drift_level: float = 3.0
    ):
        """
        Initialize DDM detector.

        Args:
            min_samples: Minimum samples before detection
            warning_level: Standard deviations for warning
            drift_level: Standard deviations for drift
        """
        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.reset()

        logger.info("DDM detector initialized")

    def reset(self):
        """Reset detector state"""
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.in_warning = False

    def update(self, prediction_correct: bool) -> DriftDetectionResult:
        """
        Update DDM with prediction result.

        Args:
            prediction_correct: Whether prediction was correct

        Returns:
            DriftDetectionResult
        """
        self.n += 1

        # Update error rate
        if prediction_correct:
            self.p = self.p + (0 - self.p) / self.n
        else:
            self.p = self.p + (1 - self.p) / self.n

        # Standard deviation of error
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)

        if self.n < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                confidence=0.0,
                warning_level=False,
                statistics={'n': self.n, 'error_rate': self.p, 'std': self.s}
            )

        # Update minimums
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        # Check for drift
        drift_detected = False
        warning_level = False
        drift_type = DriftType.NONE

        if self.p + self.s >= self.p_min + self.drift_level * self.s_min:
            drift_detected = True
            drift_type = DriftType.SUDDEN
            self.reset()
        elif self.p + self.s >= self.p_min + self.warning_level * self.s_min:
            warning_level = True
            self.in_warning = True

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=min(1.0, (self.p + self.s - self.p_min) / (self.s_min + 1e-10)),
            warning_level=warning_level,
            statistics={
                'n': self.n,
                'error_rate': self.p,
                'std': self.s,
                'p_min': self.p_min,
                's_min': self.s_min
            }
        )


class EDDM:
    """
    Early Drift Detection Method (EDDM).
    More sensitive than DDM for gradual drift.
    """

    def __init__(
        self,
        min_samples: int = 30,
        warning_level: float = 0.95,
        drift_level: float = 0.90
    ):
        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.reset()

        logger.info("EDDM detector initialized")

    def reset(self):
        """Reset detector state"""
        self.n = 0
        self.num_errors = 0
        self.last_error = 0
        self.mean_distance = 0.0
        self.std_distance = 0.0
        self.max_distance = 0.0
        self.max_std = 0.0
        self.in_warning = False

    def update(self, prediction_correct: bool) -> DriftDetectionResult:
        """
        Update EDDM with prediction result.

        Args:
            prediction_correct: Whether prediction was correct

        Returns:
            DriftDetectionResult
        """
        self.n += 1

        if not prediction_correct:
            self.num_errors += 1

            # Distance between errors
            distance = self.n - self.last_error
            self.last_error = self.n

            # Update statistics
            old_mean = self.mean_distance
            self.mean_distance = self.mean_distance + (distance - self.mean_distance) / self.num_errors

            if self.num_errors > 1:
                self.std_distance = np.sqrt(
                    self.std_distance**2 +
                    (distance - old_mean) * (distance - self.mean_distance)
                )

            # Update maximum
            current_value = self.mean_distance + 2 * self.std_distance
            if current_value > self.max_distance + 2 * self.max_std:
                self.max_distance = self.mean_distance
                self.max_std = self.std_distance

        if self.n < self.min_samples or self.num_errors < 2:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                confidence=0.0,
                warning_level=False,
                statistics={'n': self.n, 'num_errors': self.num_errors}
            )

        # Calculate ratio
        current_value = self.mean_distance + 2 * self.std_distance
        max_value = self.max_distance + 2 * self.max_std

        if max_value > 0:
            ratio = current_value / max_value
        else:
            ratio = 1.0

        # Check for drift
        drift_detected = False
        warning_level = False
        drift_type = DriftType.NONE

        if ratio < self.drift_level:
            drift_detected = True
            drift_type = DriftType.GRADUAL
            self.reset()
        elif ratio < self.warning_level:
            warning_level = True
            self.in_warning = True

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=1.0 - ratio,
            warning_level=warning_level,
            statistics={
                'n': self.n,
                'num_errors': self.num_errors,
                'mean_distance': self.mean_distance,
                'ratio': ratio
            }
        )


class PageHinkley:
    """
    Page-Hinkley test for drift detection.
    Detects changes in mean of a process.
    """

    def __init__(
        self,
        min_samples: int = 30,
        delta: float = 0.005,
        threshold: float = 50,
        alpha: float = 1 - 0.0001
    ):
        self.min_samples = min_samples
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.reset()

        logger.info("Page-Hinkley detector initialized")

    def reset(self):
        """Reset detector state"""
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.m = 0.0
        self.M = 0.0

    def update(self, value: float) -> DriftDetectionResult:
        """
        Update Page-Hinkley with new value.

        Args:
            value: New observation

        Returns:
            DriftDetectionResult
        """
        self.n += 1
        self.sum += value

        # Update mean
        self.mean = self.sum / self.n

        # Update cumulative sums
        self.m = self.alpha * self.m + (value - self.mean - self.delta)
        self.M = max(self.M, self.m)

        # Page-Hinkley statistic
        ph = self.M - self.m

        drift_detected = False
        drift_type = DriftType.NONE

        if self.n >= self.min_samples and ph > self.threshold:
            drift_detected = True
            drift_type = DriftType.SUDDEN
            self.reset()

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=min(1.0, ph / self.threshold) if self.threshold > 0 else 0.0,
            warning_level=ph > self.threshold * 0.7,
            statistics={
                'n': self.n,
                'mean': self.mean,
                'ph_statistic': ph
            }
        )


class MultiDriftDetector:
    """
    Combines multiple drift detectors for robust detection.
    Uses voting or weighted combination.
    """

    def __init__(
        self,
        detectors: Optional[List[BaseDriftDetector]] = None,
        voting_threshold: float = 0.5
    ):
        if detectors is None:
            # Default detectors
            self.detectors = {
                'adwin': ADWIN(),
                'ddm': DDM(),
                'eddm': EDDM(),
                'page_hinkley': PageHinkley()
            }
        else:
            self.detectors = {f'detector_{i}': d for i, d in enumerate(detectors)}

        self.voting_threshold = voting_threshold
        self.drift_history = deque(maxlen=1000)

        logger.info(f"MultiDriftDetector initialized with {len(self.detectors)} detectors")

    def update(
        self,
        value: Optional[float] = None,
        prediction_correct: Optional[bool] = None
    ) -> DriftDetectionResult:
        """
        Update all detectors.

        Args:
            value: Numeric value for ADWIN/Page-Hinkley
            prediction_correct: Prediction correctness for DDM/EDDM

        Returns:
            Combined DriftDetectionResult
        """
        results = {}
        drift_votes = 0
        warning_votes = 0
        total_confidence = 0.0

        for name, detector in self.detectors.items():
            if isinstance(detector, (ADWIN, PageHinkley)) and value is not None:
                result = detector.update(value)
            elif isinstance(detector, (DDM, EDDM)) and prediction_correct is not None:
                result = detector.update(prediction_correct)
            else:
                continue

            results[name] = result

            if result.drift_detected:
                drift_votes += 1
            if result.warning_level:
                warning_votes += 1
            total_confidence += result.confidence

        n_detectors = len(results)
        if n_detectors == 0:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.NONE,
                confidence=0.0,
                warning_level=False,
                statistics={}
            )

        # Voting decision
        drift_detected = drift_votes / n_detectors >= self.voting_threshold
        warning_level = warning_votes / n_detectors >= self.voting_threshold

        # Determine drift type
        drift_types = [r.drift_type for r in results.values() if r.drift_detected]
        if drift_types:
            from collections import Counter
            drift_type = Counter(drift_types).most_common(1)[0][0]
        else:
            drift_type = DriftType.NONE

        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=total_confidence / n_detectors,
            warning_level=warning_level,
            statistics={
                'drift_votes': drift_votes,
                'warning_votes': warning_votes,
                'n_detectors': n_detectors,
                'individual_results': {k: v.statistics for k, v in results.items()}
            }
        )

        self.drift_history.append(result)
        return result

    def reset(self):
        """Reset all detectors"""
        for detector in self.detectors.values():
            if hasattr(detector, 'reset'):
                detector.reset()


class OnlineLearner:
    """
    Online learning wrapper that adapts to concept drift.
    """

    def __init__(
        self,
        base_model: Any,
        drift_detector: Optional[MultiDriftDetector] = None,
        window_size: int = 1000,
        retrain_on_drift: bool = True
    ):
        self.base_model = base_model
        self.drift_detector = drift_detector or MultiDriftDetector()
        self.window_size = window_size
        self.retrain_on_drift = retrain_on_drift

        self.data_window = deque(maxlen=window_size)
        self.label_window = deque(maxlen=window_size)
        self.n_updates = 0
        self.n_drifts = 0
        self.is_trained = False

        logger.info("OnlineLearner initialized")

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Incremental update with new data.

        Args:
            X: Features
            y: Labels

        Returns:
            Update statistics
        """
        # Store data
        for i in range(len(X)):
            self.data_window.append(X[i])
            self.label_window.append(y[i])

        self.n_updates += len(X)

        # Check for drift
        drift_result = None
        if self.is_trained:
            predictions = self.base_model.predict(X)
            for i in range(len(predictions)):
                pred_class = np.argmax(predictions[i])
                true_class = np.argmax(y[i]) if y.ndim > 1 else y[i]
                correct = pred_class == true_class

                drift_result = self.drift_detector.update(
                    value=float(correct),
                    prediction_correct=correct
                )

                if drift_result.drift_detected and self.retrain_on_drift:
                    self._handle_drift()
                    break

        # Train/update model
        if len(self.data_window) >= 100:
            X_train = np.array(list(self.data_window))
            y_train = np.array(list(self.label_window))

            if hasattr(self.base_model, 'partial_fit'):
                self.base_model.partial_fit(X_train, y_train)
            else:
                self.base_model.fit(X_train, y_train)

            self.is_trained = True

        return {
            'n_updates': self.n_updates,
            'n_drifts': self.n_drifts,
            'drift_detected': drift_result.drift_detected if drift_result else False,
            'window_size': len(self.data_window)
        }

    def _handle_drift(self):
        """Handle detected drift"""
        logger.warning("Drift detected - retraining model")
        self.n_drifts += 1

        # Option 1: Reset and retrain on recent data
        if len(self.data_window) >= 50:
            X_recent = np.array(list(self.data_window)[-int(self.window_size/2):])
            y_recent = np.array(list(self.label_window)[-int(self.window_size/2):])

            if hasattr(self.base_model, 'fit'):
                self.base_model.fit(X_recent, y_recent)

        # Reset drift detector
        self.drift_detector.reset()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            return np.zeros((len(X), 2))
        return self.base_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            return np.ones((len(X), 2)) * 0.5
        return self.base_model.predict_proba(X)


class IncrementalSGD:
    """
    Incremental Stochastic Gradient Descent classifier.
    Supports online learning with mini-batches.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        regularization: float = 0.0001,
        momentum: float = 0.9
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.momentum = momentum

        # Initialize weights
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros(output_size)

        # Momentum terms
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

        self.n_samples_seen = 0

        logger.info(f"IncrementalSGD initialized: {input_size} -> {output_size}")

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incremental training step.

        Args:
            X: Features (batch_size, input_size) or (input_size,)
            y: Labels (batch_size, output_size) or (output_size,)

        Returns:
            Loss value
        """
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        batch_size = X.shape[0]
        self.n_samples_seen += batch_size

        # Forward pass
        logits = X @ self.weights + self.bias

        # Use sigmoid for binary (single output), softmax for multiclass
        if self.output_size == 1:
            proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(proba + 1e-10) + (1 - y) * np.log(1 - proba + 1e-10))
        else:
            proba = self._softmax(logits)
            # Categorical cross-entropy loss
            loss = -np.mean(np.sum(y * np.log(proba + 1e-10), axis=1))

        # Backward pass
        d_logits = (proba - y) / batch_size

        d_weights = X.T @ d_logits + self.regularization * self.weights
        d_bias = np.sum(d_logits, axis=0)

        # Momentum update
        self.v_weights = self.momentum * self.v_weights - self.learning_rate * d_weights
        self.v_bias = self.momentum * self.v_bias - self.learning_rate * d_bias

        self.weights += self.v_weights
        self.bias += self.v_bias

        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> float:
        """Full training"""
        for _ in range(epochs):
            loss = self.partial_fit(X, y)
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        logits = X @ self.weights + self.bias
        if self.output_size == 1:
            # Binary classification: use sigmoid and return [1-p, p]
            p = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            return np.column_stack([1 - p.flatten(), p.flatten()])
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PassiveAggressiveLearner:
    """
    Passive-Aggressive online learning algorithm.
    Aggressive updates only when making errors.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        C: float = 1.0,
        mode: str = 'pa-ii'
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.C = C
        self.mode = mode

        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)

        logger.info(f"PassiveAggressiveLearner initialized: {mode}")

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incremental update.

        Args:
            X: Features
            y: Labels (one-hot)

        Returns:
            Loss value
        """
        total_loss = 0.0

        for i in range(len(X)):
            x = X[i:i+1]
            yi = y[i:i+1]

            # Predict
            scores = x @ self.weights + self.bias
            pred = np.argmax(scores)
            true = np.argmax(yi)

            if pred != true:
                # Calculate hinge loss
                margin = scores[0, true] - scores[0, pred]
                loss = max(0, 1 - margin)
                total_loss += loss

                if loss > 0:
                    # Calculate update
                    x_norm_sq = np.sum(x ** 2)

                    if self.mode == 'pa':
                        tau = loss / (x_norm_sq + 1e-10)
                    elif self.mode == 'pa-i':
                        tau = min(self.C, loss / (x_norm_sq + 1e-10))
                    else:  # pa-ii
                        tau = loss / (x_norm_sq + 1 / (2 * self.C))

                    # Update
                    update = tau * x.T
                    self.weights[:, true] += update.flatten()
                    self.weights[:, pred] -= update.flatten()
                    self.bias[true] += tau
                    self.bias[pred] -= tau

        return total_loss / len(X)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> float:
        """Full training"""
        for _ in range(epochs):
            indices = np.random.permutation(len(X))
            loss = self.partial_fit(X[indices], y[indices])
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (softmax of scores)"""
        scores = X @ self.weights + self.bias
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        scores = X @ self.weights + self.bias
        return np.argmax(scores, axis=1)


class AdaptiveWindowManager:
    """
    Manages adaptive windows for streaming data.
    Adjusts window size based on drift detection.
    """

    def __init__(
        self,
        initial_size: int = 1000,
        min_size: int = 100,
        max_size: int = 10000,
        growth_rate: float = 1.2,
        shrink_rate: float = 0.5
    ):
        self.initial_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.growth_rate = growth_rate
        self.shrink_rate = shrink_rate

        self.current_size = initial_size
        self.data = deque(maxlen=max_size)
        self.labels = deque(maxlen=max_size)

        self.drift_detector = ADWIN()
        self.n_adjustments = 0

        logger.info(f"AdaptiveWindowManager initialized: size={initial_size}")

    def add(self, x: np.ndarray, y: np.ndarray, error: float = 0.0) -> bool:
        """
        Add data to window.

        Args:
            x: Feature vector
            y: Label
            error: Prediction error

        Returns:
            Whether drift was detected
        """
        self.data.append(x)
        self.labels.append(y)

        # Check for drift
        drift_result = self.drift_detector.update(error)

        if drift_result.drift_detected:
            self._shrink_window()
            return True

        # Grow window if stable
        if len(self.data) >= self.current_size:
            self._grow_window()

        return False

    def _shrink_window(self):
        """Shrink window after drift"""
        new_size = max(self.min_size, int(self.current_size * self.shrink_rate))

        if new_size < len(self.data):
            # Keep only recent data
            recent_data = list(self.data)[-new_size:]
            recent_labels = list(self.labels)[-new_size:]

            self.data = deque(recent_data, maxlen=self.max_size)
            self.labels = deque(recent_labels, maxlen=self.max_size)

        self.current_size = new_size
        self.n_adjustments += 1
        logger.info(f"Window shrunk to {new_size}")

    def _grow_window(self):
        """Grow window when stable"""
        new_size = min(self.max_size, int(self.current_size * self.growth_rate))
        self.current_size = new_size
        self.data = deque(self.data, maxlen=self.max_size)
        self.labels = deque(self.labels, maxlen=self.max_size)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current window data"""
        if len(self.data) == 0:
            return np.array([]), np.array([])
        return np.array(list(self.data)), np.array(list(self.labels))

    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get most recent n samples"""
        n = min(n, len(self.data))
        if n == 0:
            return np.array([]), np.array([])
        return np.array(list(self.data)[-n:]), np.array(list(self.labels)[-n:])


class StreamProcessor:
    """
    Processes streaming data for online learning.
    Handles batching, buffering, and preprocessing.
    """

    def __init__(
        self,
        learner: OnlineLearner,
        batch_size: int = 32,
        buffer_size: int = 100,
        preprocess_fn: Optional[Callable] = None
    ):
        self.learner = learner
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.preprocess_fn = preprocess_fn

        self.buffer_x = []
        self.buffer_y = []

        self.n_processed = 0
        self.processing_times = deque(maxlen=100)

        logger.info(f"StreamProcessor initialized: batch_size={batch_size}")

    def process(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process streaming data.

        Args:
            x: Feature vector
            y: Label

        Returns:
            Update result if batch processed, None otherwise
        """
        start_time = time.time()

        # Preprocess
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)

        # Add to buffer
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        self.n_processed += 1

        result = None

        # Process batch when buffer is full
        if len(self.buffer_x) >= self.batch_size:
            X = np.array(self.buffer_x)
            Y = np.array(self.buffer_y)

            result = self.learner.partial_fit(X, Y)

            # Clear buffer
            self.buffer_x = []
            self.buffer_y = []

        # Track processing time
        self.processing_times.append(time.time() - start_time)

        return result

    def flush(self) -> Optional[Dict[str, Any]]:
        """Process remaining data in buffer"""
        if len(self.buffer_x) > 0:
            X = np.array(self.buffer_x)
            Y = np.array(self.buffer_y)

            result = self.learner.partial_fit(X, Y)

            self.buffer_x = []
            self.buffer_y = []

            return result
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'n_processed': self.n_processed,
            'buffer_size': len(self.buffer_x),
            'avg_processing_time': np.mean(list(self.processing_times)) if self.processing_times else 0,
            'learner_stats': {
                'n_updates': self.learner.n_updates,
                'n_drifts': self.learner.n_drifts
            }
        }


class ForgetfulLearner:
    """
    Learner with forgetting mechanism for non-stationary data.
    Uses exponential decay for old samples.
    """

    def __init__(
        self,
        base_model: Any,
        decay_rate: float = 0.99,
        min_weight: float = 0.01
    ):
        self.base_model = base_model
        self.decay_rate = decay_rate
        self.min_weight = min_weight

        self.sample_weights = []
        self.samples_x = []
        self.samples_y = []

        logger.info(f"ForgetfulLearner initialized: decay_rate={decay_rate}")

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incremental update with forgetting.

        Args:
            X: Features
            y: Labels

        Returns:
            Loss value
        """
        # Decay existing weights
        self.sample_weights = [max(self.min_weight, w * self.decay_rate) for w in self.sample_weights]

        # Add new samples
        for i in range(len(X)):
            self.samples_x.append(X[i])
            self.samples_y.append(y[i])
            self.sample_weights.append(1.0)

        # Remove samples with very low weights
        keep_indices = [i for i, w in enumerate(self.sample_weights) if w > self.min_weight]

        self.samples_x = [self.samples_x[i] for i in keep_indices]
        self.samples_y = [self.samples_y[i] for i in keep_indices]
        self.sample_weights = [self.sample_weights[i] for i in keep_indices]

        # Train model with weighted samples
        if len(self.samples_x) > 0:
            X_train = np.array(self.samples_x)
            y_train = np.array(self.samples_y)
            weights = np.array(self.sample_weights)

            # Normalize weights
            weights = weights / np.sum(weights)

            # Weighted training (if supported)
            if hasattr(self.base_model, 'fit'):
                return self.base_model.fit(X_train, y_train)

        return 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.base_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        return self.base_model.predict_proba(X)
