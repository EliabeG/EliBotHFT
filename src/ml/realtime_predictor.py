"""
Real-time Prediction Pipeline for HFT Trading

This module implements low-latency prediction infrastructure:
1. Prediction Caching and Batching
2. Model Serving with versioning
3. Feature Pipeline with preprocessing
4. Latency Monitoring
5. Prediction Queuing
6. Confidence Calibration
7. Multi-model Routing
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time
import threading
import queue
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions"""
    ERROR_PROBABILITY = auto()
    ERROR_TYPE = auto()
    TRADE_OUTCOME = auto()
    RISK_LEVEL = auto()
    STRATEGY_SELECTION = auto()
    POSITION_SIZE = auto()


@dataclass
class PredictionRequest:
    """Request for prediction"""
    request_id: str
    prediction_type: PredictionType
    features: np.ndarray
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent


@dataclass
class PredictionResponse:
    """Response from prediction"""
    request_id: str
    prediction_type: PredictionType
    prediction: np.ndarray
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeaturePipeline:
    """
    Feature preprocessing pipeline for real-time predictions.
    """

    def __init__(
        self,
        feature_names: List[str],
        normalization: str = 'standard',  # 'standard', 'minmax', 'robust', 'none'
        handle_missing: str = 'zero'  # 'zero', 'mean', 'median', 'forward_fill'
    ):
        self.feature_names = feature_names
        self.normalization = normalization
        self.handle_missing = handle_missing

        # Statistics for normalization
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None
        self.medians: Optional[np.ndarray] = None
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None

        # Running statistics for online updates
        self.n_samples = 0
        self.running_sum: Optional[np.ndarray] = None
        self.running_sq_sum: Optional[np.ndarray] = None

        self.is_fitted = False

        logger.info(f"FeaturePipeline initialized: {normalization} normalization")

    def fit(self, X: np.ndarray):
        """Fit pipeline on training data"""
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0) + 1e-8
        self.mins = np.min(X, axis=0)
        self.maxs = np.max(X, axis=0)
        self.medians = np.median(X, axis=0)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)

        self.running_sum = np.sum(X, axis=0)
        self.running_sq_sum = np.sum(X ** 2, axis=0)
        self.n_samples = len(X)

        self.is_fitted = True

        logger.info(f"FeaturePipeline fitted on {len(X)} samples")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        if not self.is_fitted:
            logger.warning("Pipeline not fitted, returning raw features")
            return X

        X = X.copy()

        # Handle missing values
        if self.handle_missing == 'zero':
            X = np.nan_to_num(X, nan=0.0)
        elif self.handle_missing == 'mean':
            for i in range(X.shape[1] if X.ndim > 1 else 1):
                mask = np.isnan(X[:, i] if X.ndim > 1 else X)
                if X.ndim > 1:
                    X[mask, i] = self.means[i]
                else:
                    X[mask] = self.means[i]
        elif self.handle_missing == 'median':
            for i in range(X.shape[1] if X.ndim > 1 else 1):
                mask = np.isnan(X[:, i] if X.ndim > 1 else X)
                if X.ndim > 1:
                    X[mask, i] = self.medians[i]
                else:
                    X[mask] = self.medians[i]

        # Normalize
        if self.normalization == 'standard':
            X = (X - self.means) / self.stds
        elif self.normalization == 'minmax':
            X = (X - self.mins) / (self.maxs - self.mins + 1e-8)
        elif self.normalization == 'robust':
            iqr = self.q3 - self.q1 + 1e-8
            X = (X - self.medians) / iqr

        # Clip extreme values
        X = np.clip(X, -10, 10)

        return X

    def update_online(self, x: np.ndarray):
        """Update statistics with new sample (online learning)"""
        if not self.is_fitted:
            return

        self.n_samples += 1
        self.running_sum += x
        self.running_sq_sum += x ** 2

        # Update means and stds
        self.means = self.running_sum / self.n_samples
        variance = (self.running_sq_sum / self.n_samples) - (self.means ** 2)
        self.stds = np.sqrt(np.maximum(variance, 0)) + 1e-8

    def save(self, filepath: str):
        """Save pipeline state"""
        state = {
            'feature_names': self.feature_names,
            'normalization': self.normalization,
            'handle_missing': self.handle_missing,
            'means': self.means.tolist() if self.means is not None else None,
            'stds': self.stds.tolist() if self.stds is not None else None,
            'mins': self.mins.tolist() if self.mins is not None else None,
            'maxs': self.maxs.tolist() if self.maxs is not None else None,
            'medians': self.medians.tolist() if self.medians is not None else None,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load(self, filepath: str):
        """Load pipeline state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.feature_names = state['feature_names']
        self.normalization = state['normalization']
        self.handle_missing = state['handle_missing']
        self.means = np.array(state['means']) if state['means'] else None
        self.stds = np.array(state['stds']) if state['stds'] else None
        self.mins = np.array(state['mins']) if state['mins'] else None
        self.maxs = np.array(state['maxs']) if state['maxs'] else None
        self.medians = np.array(state['medians']) if state['medians'] else None
        self.is_fitted = state['is_fitted']


class PredictionCache:
    """
    Cache for recent predictions to avoid redundant computation.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_ms: float = 100.0,  # Time to live in milliseconds
        similarity_threshold: float = 0.01
    ):
        self.max_size = max_size
        self.ttl_ms = ttl_ms
        self.similarity_threshold = similarity_threshold

        self.cache: Dict[str, Tuple[np.ndarray, float, float]] = {}  # key -> (prediction, confidence, timestamp)
        self.feature_cache: Dict[str, np.ndarray] = {}  # key -> features

        self.hits = 0
        self.misses = 0

        logger.info(f"PredictionCache initialized: max_size={max_size}, ttl={ttl_ms}ms")

    def _feature_hash(self, features: np.ndarray) -> str:
        """Generate hash for feature vector"""
        # Quantize features for approximate matching
        quantized = np.round(features / self.similarity_threshold) * self.similarity_threshold
        return hash(quantized.tobytes())

    def get(self, features: np.ndarray, prediction_type: PredictionType) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get cached prediction if available.

        Args:
            features: Feature vector
            prediction_type: Type of prediction

        Returns:
            (prediction, confidence) or None if not cached
        """
        key = f"{prediction_type.name}_{self._feature_hash(features)}"

        if key in self.cache:
            prediction, confidence, timestamp = self.cache[key]

            # Check TTL
            age_ms = (time.time() - timestamp) * 1000
            if age_ms < self.ttl_ms:
                self.hits += 1
                return prediction, confidence

            # Expired, remove
            del self.cache[key]
            if key in self.feature_cache:
                del self.feature_cache[key]

        self.misses += 1
        return None

    def put(
        self,
        features: np.ndarray,
        prediction_type: PredictionType,
        prediction: np.ndarray,
        confidence: float
    ):
        """Cache a prediction"""
        key = f"{prediction_type.name}_{self._feature_hash(features)}"

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
            del self.cache[oldest_key]
            if oldest_key in self.feature_cache:
                del self.feature_cache[oldest_key]

        self.cache[key] = (prediction, confidence, time.time())
        self.feature_cache[key] = features

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.feature_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class ConfidenceCalibrator:
    """
    Calibrate prediction confidence using isotonic regression.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_means: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, probabilities: np.ndarray, actuals: np.ndarray):
        """
        Fit calibrator on validation data.

        Args:
            probabilities: Predicted probabilities
            actuals: Actual outcomes (0 or 1)
        """
        # Create bins
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.bin_means = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (probabilities >= self.bin_edges[i]) & (probabilities < self.bin_edges[i + 1])
            if np.sum(mask) > 0:
                self.bin_means[i] = np.mean(actuals[mask])
            else:
                self.bin_means[i] = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2

        self.is_fitted = True

        logger.info("ConfidenceCalibrator fitted")

    def calibrate(self, probability: float) -> float:
        """
        Calibrate a single probability.

        Args:
            probability: Raw probability

        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            return probability

        # Find bin
        bin_idx = np.digitize(probability, self.bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        return self.bin_means[bin_idx]

    def calibrate_batch(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate batch of probabilities"""
        return np.array([self.calibrate(p) for p in probabilities])


class ModelServer:
    """
    Serve ML models with versioning and routing.
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}  # model_name -> {version -> model}
        self.active_versions: Dict[str, str] = {}  # model_name -> active_version
        self.model_stats: Dict[str, Dict[str, Any]] = {}

        logger.info("ModelServer initialized")

    def register_model(
        self,
        name: str,
        model: Any,
        version: str,
        make_active: bool = True
    ):
        """
        Register a model.

        Args:
            name: Model name
            model: Model object
            version: Version string
            make_active: Whether to make this the active version
        """
        if name not in self.models:
            self.models[name] = {}
            self.model_stats[name] = {}

        self.models[name][version] = model
        self.model_stats[name][version] = {
            'n_predictions': 0,
            'total_latency_ms': 0,
            'errors': 0
        }

        if make_active:
            self.active_versions[name] = version

        logger.info(f"Registered model {name} v{version}")

    def predict(
        self,
        model_name: str,
        features: np.ndarray,
        version: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Make prediction with specified model.

        Args:
            model_name: Name of model
            features: Feature vector/matrix
            version: Specific version (uses active if None)

        Returns:
            (prediction, latency_ms)
        """
        version = version or self.active_versions.get(model_name)

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.models[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        model = self.models[model_name][version]
        stats = self.model_stats[model_name][version]

        start_time = time.time()
        try:
            prediction = model.predict(features)
            latency_ms = (time.time() - start_time) * 1000

            stats['n_predictions'] += 1
            stats['total_latency_ms'] += latency_ms

            return prediction, latency_ms
        except Exception as e:
            stats['errors'] += 1
            raise e

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models"""
        result = {}
        for model_name, versions in self.model_stats.items():
            result[model_name] = {
                'active_version': self.active_versions.get(model_name),
                'versions': {
                    v: {
                        'n_predictions': s['n_predictions'],
                        'avg_latency_ms': s['total_latency_ms'] / s['n_predictions'] if s['n_predictions'] > 0 else 0,
                        'errors': s['errors']
                    }
                    for v, s in versions.items()
                }
            }
        return result


class LatencyMonitor:
    """
    Monitor and track prediction latency.
    """

    def __init__(self, window_size: int = 1000, alert_threshold_ms: float = 10.0):
        self.window_size = window_size
        self.alert_threshold_ms = alert_threshold_ms

        self.latencies: deque = deque(maxlen=window_size)
        self.alerts: List[Dict[str, Any]] = []

        self.total_predictions = 0
        self.total_latency_ms = 0.0

        logger.info(f"LatencyMonitor initialized: alert_threshold={alert_threshold_ms}ms")

    def record(self, latency_ms: float, prediction_type: str = "unknown"):
        """Record a latency measurement"""
        self.latencies.append((time.time(), latency_ms, prediction_type))
        self.total_predictions += 1
        self.total_latency_ms += latency_ms

        # Check for alert
        if latency_ms > self.alert_threshold_ms:
            self.alerts.append({
                'timestamp': time.time(),
                'latency_ms': latency_ms,
                'prediction_type': prediction_type,
                'message': f'High latency: {latency_ms:.2f}ms (threshold: {self.alert_threshold_ms}ms)'
            })

    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latencies:
            return {'avg_latency_ms': 0, 'p50_latency_ms': 0, 'p99_latency_ms': 0, 'max_latency_ms': 0}

        latency_values = [l[1] for l in self.latencies]

        return {
            'avg_latency_ms': np.mean(latency_values),
            'p50_latency_ms': np.percentile(latency_values, 50),
            'p95_latency_ms': np.percentile(latency_values, 95),
            'p99_latency_ms': np.percentile(latency_values, 99),
            'max_latency_ms': np.max(latency_values),
            'min_latency_ms': np.min(latency_values),
            'total_predictions': self.total_predictions
        }

    def get_recent_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-n:]


class PredictionQueue:
    """
    Priority queue for prediction requests.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self.pending: Dict[str, PredictionRequest] = {}

        self.total_enqueued = 0
        self.total_processed = 0
        self.dropped = 0

    def enqueue(self, request: PredictionRequest) -> bool:
        """
        Add request to queue.

        Returns:
            True if enqueued, False if dropped
        """
        try:
            # Priority is negative so higher priority values come first
            self.queue.put_nowait((-request.priority, request.timestamp, request))
            self.pending[request.request_id] = request
            self.total_enqueued += 1
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("Prediction queue full, request dropped")
            return False

    def dequeue(self, timeout: float = 0.001) -> Optional[PredictionRequest]:
        """Get next request from queue"""
        try:
            _, _, request = self.queue.get(timeout=timeout)
            if request.request_id in self.pending:
                del self.pending[request.request_id]
            self.total_processed += 1
            return request
        except queue.Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': self.queue.qsize(),
            'pending': len(self.pending),
            'total_enqueued': self.total_enqueued,
            'total_processed': self.total_processed,
            'dropped': self.dropped
        }


class RealtimePredictor:
    """
    Complete real-time prediction system.
    """

    def __init__(
        self,
        feature_pipeline: Optional[FeaturePipeline] = None,
        cache_enabled: bool = True,
        cache_ttl_ms: float = 100.0,
        async_enabled: bool = False
    ):
        self.feature_pipeline = feature_pipeline
        self.cache_enabled = cache_enabled
        self.async_enabled = async_enabled

        # Components
        self.model_server = ModelServer()
        self.cache = PredictionCache(ttl_ms=cache_ttl_ms) if cache_enabled else None
        self.latency_monitor = LatencyMonitor()
        self.confidence_calibrator = ConfidenceCalibrator()

        # Async components
        if async_enabled:
            self.prediction_queue = PredictionQueue()
            self.response_callbacks: Dict[str, Callable] = {}
            self._running = False
            self._worker_thread: Optional[threading.Thread] = None

        # Model routing
        self.prediction_type_models: Dict[PredictionType, str] = {}

        logger.info(f"RealtimePredictor initialized: cache={cache_enabled}, async={async_enabled}")

    def register_model(
        self,
        model: Any,
        name: str,
        version: str,
        prediction_types: List[PredictionType],
        make_active: bool = True
    ):
        """Register a model for specific prediction types"""
        self.model_server.register_model(name, model, version, make_active)

        for pred_type in prediction_types:
            self.prediction_type_models[pred_type] = name

    def predict(
        self,
        features: np.ndarray,
        prediction_type: PredictionType,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResponse:
        """
        Make a prediction.

        Args:
            features: Raw features
            prediction_type: Type of prediction needed
            context: Additional context

        Returns:
            PredictionResponse
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000000)}"

        # Preprocess features
        if self.feature_pipeline is not None:
            features = self.feature_pipeline.transform(features.reshape(1, -1))[0]

        # Check cache
        if self.cache_enabled and self.cache is not None:
            cached = self.cache.get(features, prediction_type)
            if cached is not None:
                prediction, confidence = cached
                latency_ms = (time.time() - start_time) * 1000
                self.latency_monitor.record(latency_ms, f"{prediction_type.name}_cached")

                return PredictionResponse(
                    request_id=request_id,
                    prediction_type=prediction_type,
                    prediction=prediction,
                    confidence=confidence,
                    model_version="cached",
                    latency_ms=latency_ms,
                    metadata={'source': 'cache'}
                )

        # Get model for this prediction type
        model_name = self.prediction_type_models.get(prediction_type)
        if model_name is None:
            raise ValueError(f"No model registered for {prediction_type.name}")

        # Make prediction
        prediction, model_latency = self.model_server.predict(model_name, features.reshape(1, -1))

        # Extract confidence
        if prediction.ndim > 1:
            confidence = float(np.max(prediction))
            prediction = prediction[0]
        else:
            confidence = 0.5

        # Calibrate confidence
        if self.confidence_calibrator.is_fitted:
            confidence = self.confidence_calibrator.calibrate(confidence)

        # Cache result
        if self.cache_enabled and self.cache is not None:
            self.cache.put(features, prediction_type, prediction, confidence)

        # Record latency
        total_latency = (time.time() - start_time) * 1000
        self.latency_monitor.record(total_latency, prediction_type.name)

        return PredictionResponse(
            request_id=request_id,
            prediction_type=prediction_type,
            prediction=prediction,
            confidence=confidence,
            model_version=self.model_server.active_versions.get(model_name, "unknown"),
            latency_ms=total_latency,
            metadata={
                'model_name': model_name,
                'model_latency_ms': model_latency
            }
        )

    def predict_batch(
        self,
        features_batch: np.ndarray,
        prediction_type: PredictionType
    ) -> List[PredictionResponse]:
        """Make batch predictions"""
        responses = []

        # Preprocess all features
        if self.feature_pipeline is not None:
            features_batch = self.feature_pipeline.transform(features_batch)

        # Get model
        model_name = self.prediction_type_models.get(prediction_type)
        if model_name is None:
            raise ValueError(f"No model registered for {prediction_type.name}")

        # Batch predict
        start_time = time.time()
        predictions, model_latency = self.model_server.predict(model_name, features_batch)
        total_latency = (time.time() - start_time) * 1000

        # Create responses
        for i, (features, prediction) in enumerate(zip(features_batch, predictions)):
            if prediction.ndim > 0:
                confidence = float(np.max(prediction))
            else:
                confidence = 0.5

            responses.append(PredictionResponse(
                request_id=f"batch_{int(start_time * 1000000)}_{i}",
                prediction_type=prediction_type,
                prediction=prediction,
                confidence=confidence,
                model_version=self.model_server.active_versions.get(model_name, "unknown"),
                latency_ms=total_latency / len(features_batch),
                metadata={'batch_size': len(features_batch)}
            ))

        self.latency_monitor.record(total_latency, f"{prediction_type.name}_batch")

        return responses

    def predict_async(
        self,
        features: np.ndarray,
        prediction_type: PredictionType,
        callback: Callable[[PredictionResponse], None],
        priority: int = 0
    ) -> str:
        """
        Asynchronous prediction.

        Args:
            features: Features
            prediction_type: Type of prediction
            callback: Function to call with response
            priority: Request priority (higher = more urgent)

        Returns:
            Request ID
        """
        if not self.async_enabled:
            raise RuntimeError("Async prediction not enabled")

        request_id = f"async_{int(time.time() * 1000000)}"

        request = PredictionRequest(
            request_id=request_id,
            prediction_type=prediction_type,
            features=features,
            priority=priority
        )

        self.response_callbacks[request_id] = callback
        self.prediction_queue.enqueue(request)

        return request_id

    def start_async_worker(self):
        """Start async prediction worker"""
        if not self.async_enabled:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Async prediction worker started")

    def stop_async_worker(self):
        """Stop async prediction worker"""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        logger.info("Async prediction worker stopped")

    def _async_worker(self):
        """Worker thread for async predictions"""
        while self._running:
            request = self.prediction_queue.dequeue(timeout=0.01)
            if request is None:
                continue

            try:
                response = self.predict(
                    request.features,
                    request.prediction_type,
                    request.context
                )

                callback = self.response_callbacks.pop(request.request_id, None)
                if callback is not None:
                    callback(response)

            except Exception as e:
                logger.error(f"Async prediction error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'latency': self.latency_monitor.get_stats(),
            'models': self.model_server.get_model_stats(),
        }

        if self.cache_enabled and self.cache is not None:
            stats['cache'] = self.cache.get_stats()

        if self.async_enabled:
            stats['queue'] = self.prediction_queue.get_stats()

        return stats


class MultiModelRouter:
    """
    Route predictions to different models based on context.
    """

    def __init__(self, predictor: RealtimePredictor):
        self.predictor = predictor
        self.routing_rules: List[Tuple[Callable, str]] = []  # (condition, model_name)

    def add_rule(self, condition: Callable[[Dict[str, Any]], bool], model_name: str):
        """
        Add routing rule.

        Args:
            condition: Function that takes context and returns True if rule matches
            model_name: Model to route to if condition is true
        """
        self.routing_rules.append((condition, model_name))

    def route(self, context: Dict[str, Any]) -> str:
        """
        Determine model to use based on context.

        Args:
            context: Prediction context

        Returns:
            Model name to use
        """
        for condition, model_name in self.routing_rules:
            try:
                if condition(context):
                    return model_name
            except Exception as e:
                logger.warning(f"Routing rule error: {e}")

        # Default to first registered model
        if self.predictor.model_server.active_versions:
            return list(self.predictor.model_server.active_versions.keys())[0]

        raise ValueError("No models registered for routing")


class EnsemblePredictor:
    """
    Make predictions from multiple models and combine them.
    """

    def __init__(
        self,
        predictor: RealtimePredictor,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        combination: str = 'weighted_mean'  # 'mean', 'weighted_mean', 'voting', 'stacking'
    ):
        self.predictor = predictor
        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)
        self.combination = combination

    def predict(
        self,
        features: np.ndarray,
        prediction_type: PredictionType
    ) -> PredictionResponse:
        """Make ensemble prediction"""
        start_time = time.time()

        # Get predictions from all models
        predictions = []
        confidences = []

        for model_name, weight in zip(self.model_names, self.weights):
            # Temporarily override model routing
            original_model = self.predictor.prediction_type_models.get(prediction_type)
            self.predictor.prediction_type_models[prediction_type] = model_name

            response = self.predictor.predict(features, prediction_type)

            # Restore original
            if original_model is not None:
                self.predictor.prediction_type_models[prediction_type] = original_model

            predictions.append(response.prediction * weight)
            confidences.append(response.confidence * weight)

        # Combine predictions
        if self.combination == 'weighted_mean':
            final_prediction = np.sum(predictions, axis=0)
            final_confidence = np.sum(confidences)
        elif self.combination == 'mean':
            final_prediction = np.mean(predictions, axis=0)
            final_confidence = np.mean(confidences)
        elif self.combination == 'voting':
            # Hard voting
            votes = np.array([np.argmax(p) for p in predictions])
            final_prediction = np.zeros_like(predictions[0])
            final_prediction[np.bincount(votes).argmax()] = 1.0
            final_confidence = np.mean(confidences)
        else:
            final_prediction = np.mean(predictions, axis=0)
            final_confidence = np.mean(confidences)

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            request_id=f"ensemble_{int(start_time * 1000000)}",
            prediction_type=prediction_type,
            prediction=final_prediction,
            confidence=final_confidence,
            model_version="ensemble",
            latency_ms=latency_ms,
            metadata={
                'models': self.model_names,
                'weights': self.weights,
                'combination': self.combination
            }
        )
