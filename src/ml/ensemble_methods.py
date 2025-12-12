"""
Ensemble Methods for Robust ML Predictions

This module implements advanced ensemble techniques:
1. Bagging (Bootstrap Aggregating)
2. Boosting (AdaBoost, Gradient Boosting)
3. Stacking (Meta-learning)
4. Random Forest-like feature bagging
5. Model Selection and Combination
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating ensemble predictions"""
    MEAN = auto()
    WEIGHTED_MEAN = auto()
    MEDIAN = auto()
    VOTING = auto()
    SOFT_VOTING = auto()
    STACKING = auto()


@dataclass
class BaseModel(ABC):
    """Abstract base model for ensemble members"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class SimpleNeuralNet:
    """Simple neural network for ensemble members"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        learning_rate: float = 0.01
    ):
        self.learning_rate = learning_rate

        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * scale)
            self.biases.append(np.zeros(sizes[i+1]))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> float:
        """Train the network"""
        for epoch in range(epochs):
            # Forward pass
            activations = [X]
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = activations[-1] @ W + b
                if i < len(self.weights) - 1:
                    a = self._relu(z)
                else:
                    a = self._softmax(z)
                activations.append(a)

            # Compute loss
            pred = activations[-1]
            loss = -np.mean(np.sum(y * np.log(pred + 1e-10), axis=1))

            # Backward pass
            d = pred - y
            for i in reversed(range(len(self.weights))):
                dW = activations[i].T @ d / X.shape[0]
                db = np.mean(d, axis=0)

                self.weights[i] -= self.learning_rate * dW
                self.biases[i] -= self.learning_rate * db

                if i > 0:
                    d = (d @ self.weights[i].T) * (activations[i] > 0)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
        return a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict(X), axis=1)


class DecisionStump:
    """Simple decision stump for boosting"""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.polarity = 1
        self.alpha = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Fit to minimize weighted error"""
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature_idx] < threshold] = -1
                    else:
                        predictions[X[:, feature_idx] >= threshold] = -1

                    # Weighted error
                    error = np.sum(weights * (predictions != y))

                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

        return min_error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] >= self.threshold] = -1

        return predictions


class BaggingEnsemble:
    """
    Bagging (Bootstrap Aggregating) Ensemble.
    Reduces variance by training on bootstrap samples.
    """

    def __init__(
        self,
        base_model_fn: Callable,
        n_estimators: int = 10,
        sample_ratio: float = 0.8,
        feature_ratio: float = 1.0,
        aggregation: AggregationMethod = AggregationMethod.SOFT_VOTING
    ):
        self.base_model_fn = base_model_fn
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.feature_ratio = feature_ratio
        self.aggregation = aggregation

        self.models = []
        self.feature_indices = []
        self.oob_score_ = 0.0

        logger.info(f"BaggingEnsemble initialized with {n_estimators} estimators")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingEnsemble':
        """Train the ensemble"""
        n_samples, n_features = X.shape
        n_sample_size = int(n_samples * self.sample_ratio)
        n_feature_size = int(n_features * self.feature_ratio)

        oob_predictions = np.zeros((n_samples, y.shape[1] if y.ndim > 1 else 2))
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # Bootstrap sample
            sample_indices = np.random.choice(n_samples, n_sample_size, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), sample_indices)

            # Feature bagging
            if self.feature_ratio < 1.0:
                feature_indices = np.random.choice(n_features, n_feature_size, replace=False)
            else:
                feature_indices = np.arange(n_features)

            self.feature_indices.append(feature_indices)

            # Train model
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]

            model = self.base_model_fn()
            model.fit(X_sample, y_sample)
            self.models.append(model)

            # OOB predictions
            if len(oob_indices) > 0:
                X_oob = X[oob_indices][:, feature_indices]
                oob_pred = model.predict_proba(X_oob)
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

            logger.debug(f"Trained estimator {i+1}/{self.n_estimators}")

        # Calculate OOB score
        valid_mask = oob_counts > 0
        if np.any(valid_mask):
            oob_pred_final = oob_predictions[valid_mask] / oob_counts[valid_mask, np.newaxis]
            y_valid = y[valid_mask] if y.ndim > 1 else np.eye(2)[y[valid_mask].astype(int)]
            self.oob_score_ = np.mean(np.argmax(oob_pred_final, axis=1) == np.argmax(y_valid, axis=1))

        logger.info(f"Bagging ensemble trained, OOB score: {self.oob_score_:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        predictions = []

        for model, features in zip(self.models, self.feature_indices):
            X_subset = X[:, features]
            pred = model.predict_proba(X_subset)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.aggregation == AggregationMethod.MEAN:
            return np.mean(predictions, axis=0)
        elif self.aggregation == AggregationMethod.MEDIAN:
            return np.median(predictions, axis=0)
        elif self.aggregation == AggregationMethod.SOFT_VOTING:
            return np.mean(predictions, axis=0)
        else:
            return np.mean(predictions, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return np.argmax(self.predict_proba(X), axis=1)


class AdaBoostEnsemble:
    """
    AdaBoost (Adaptive Boosting) Ensemble.
    Sequentially trains weak learners with adaptive sample weights.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.stumps = []
        self.alphas = []

        logger.info(f"AdaBoostEnsemble initialized with {n_estimators} estimators")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostEnsemble':
        """Train the ensemble"""
        n_samples = X.shape[0]

        # Convert y to {-1, 1}
        if y.ndim > 1:
            y_binary = 2 * np.argmax(y, axis=1) - 1
        else:
            y_binary = 2 * y - 1

        # Initialize weights
        weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # Train weak learner
            stump = DecisionStump()
            error = stump.fit(X, y_binary, weights)

            # Avoid division by zero
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate alpha
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha

            # Update weights
            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y_binary * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)

            logger.debug(f"Trained stump {i+1}/{self.n_estimators}, alpha={alpha:.4f}")

        logger.info(f"AdaBoost ensemble trained with {len(self.stumps)} stumps")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for stump, alpha in zip(self.stumps, self.alphas):
            predictions += alpha * stump.predict(X)

        return np.sign(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for stump, alpha in zip(self.stumps, self.alphas):
            scores += alpha * stump.predict(X)

        # Convert to probabilities using sigmoid
        proba_pos = 1 / (1 + np.exp(-2 * scores))
        return np.column_stack([1 - proba_pos, proba_pos])


class GradientBoostingEnsemble:
    """
    Gradient Boosting Ensemble.
    Sequentially fits models to residuals.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

        self.trees = []
        self.initial_prediction = 0.0

        logger.info(f"GradientBoostingEnsemble initialized")

    def _build_tree(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        depth: int = 0
    ) -> Dict:
        """Build a simple regression tree"""
        if depth >= self.max_depth or len(X) < 2:
            return {'value': np.mean(residuals)}

        n_samples, n_features = X.shape
        best_gain = 0
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]

                # Calculate gain (variance reduction)
                var_before = np.var(residuals) * len(residuals)
                var_after = np.var(left_residuals) * len(left_residuals) + \
                           np.var(right_residuals) * len(right_residuals)
                gain = var_before - var_after

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None:
            return {'value': np.mean(residuals)}

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': self._build_tree(
                X[best_split['left_mask']],
                residuals[best_split['left_mask']],
                depth + 1
            ),
            'right': self._build_tree(
                X[best_split['right_mask']],
                residuals[best_split['right_mask']],
                depth + 1
            )
        }

    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """Predict with a single tree"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            node = tree
            while 'value' not in node:
                if X[i, node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions[i] = node['value']

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingEnsemble':
        """Train the ensemble"""
        # For binary classification, use log-odds
        if y.ndim > 1:
            y_binary = np.argmax(y, axis=1)
        else:
            y_binary = y.astype(float)

        # Initial prediction (log-odds of positive class)
        p = np.mean(y_binary)
        self.initial_prediction = np.log(p / (1 - p + 1e-10) + 1e-10)

        predictions = np.full(len(y_binary), self.initial_prediction)

        for i in range(self.n_estimators):
            # Convert to probabilities
            proba = 1 / (1 + np.exp(-predictions))

            # Compute pseudo-residuals
            residuals = y_binary - proba

            # Subsample
            if self.subsample < 1.0:
                n_sample = int(len(X) * self.subsample)
                indices = np.random.choice(len(X), n_sample, replace=False)
                X_sample = X[indices]
                residuals_sample = residuals[indices]
            else:
                X_sample = X
                residuals_sample = residuals

            # Fit tree to residuals
            tree = self._build_tree(X_sample, residuals_sample)
            self.trees.append(tree)

            # Update predictions
            tree_pred = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_pred

            if (i + 1) % 10 == 0:
                logger.debug(f"Trained tree {i+1}/{self.n_estimators}")

        logger.info(f"Gradient Boosting trained with {len(self.trees)} trees")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        predictions = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(tree, X)

        proba_pos = 1 / (1 + np.exp(-predictions))
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StackingEnsemble:
    """
    Stacking Ensemble with meta-learner.
    Combines base model predictions using a meta-model.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Any,
        use_proba: bool = True,
        cv_folds: int = 5
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
        self.cv_folds = cv_folds

        logger.info(f"StackingEnsemble initialized with {len(base_models)} base models")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Train the stacking ensemble using cross-validation"""
        n_samples = X.shape[0]
        n_classes = y.shape[1] if y.ndim > 1 else 2

        # Generate meta-features using cross-validation
        if self.use_proba:
            meta_features = np.zeros((n_samples, len(self.base_models) * n_classes))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models)))

        fold_size = n_samples // self.cv_folds

        for fold in range(self.cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n_samples

            val_indices = np.arange(val_start, val_end)
            train_indices = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train = y[train_indices]

            for i, model in enumerate(self.base_models):
                # Clone and train model
                model_copy = type(model)(**{
                    k: v for k, v in model.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }) if hasattr(model, '__dict__') else model

                try:
                    model_copy.fit(X_train, y_train)

                    if self.use_proba:
                        pred = model_copy.predict_proba(X_val)
                        start_idx = i * n_classes
                        end_idx = start_idx + n_classes
                        meta_features[val_indices, start_idx:end_idx] = pred
                    else:
                        pred = model_copy.predict_class(X_val)
                        meta_features[val_indices, i] = pred
                except Exception as e:
                    logger.warning(f"Error training model {i} in fold {fold}: {e}")

        # Train base models on full data
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                logger.warning(f"Error training base model: {e}")

        # Train meta-model
        self.meta_model.fit(meta_features, y)

        logger.info("Stacking ensemble trained")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        n_samples = X.shape[0]
        n_classes = 2  # Assuming binary classification

        if self.use_proba:
            meta_features = np.zeros((n_samples, len(self.base_models) * n_classes))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models)))

        for i, model in enumerate(self.base_models):
            try:
                if self.use_proba:
                    pred = model.predict_proba(X)
                    start_idx = i * n_classes
                    end_idx = start_idx + n_classes
                    meta_features[:, start_idx:end_idx] = pred
                else:
                    meta_features[:, i] = model.predict_class(X)
            except Exception as e:
                logger.warning(f"Error predicting with model {i}: {e}")

        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return np.argmax(self.predict_proba(X), axis=1)


class WeightedEnsemble:
    """
    Dynamically weighted ensemble based on recent performance.
    Adapts weights to changing market conditions.
    """

    def __init__(
        self,
        models: List[Any],
        window_size: int = 100,
        min_weight: float = 0.05
    ):
        self.models = models
        self.window_size = window_size
        self.min_weight = min_weight

        self.weights = np.ones(len(models)) / len(models)
        self.performance_history = [deque(maxlen=window_size) for _ in models]

        logger.info(f"WeightedEnsemble initialized with {len(models)} models")

    def update_weights(self, X: np.ndarray, y_true: np.ndarray):
        """Update weights based on recent performance"""
        for i, model in enumerate(self.models):
            try:
                pred = model.predict_class(X)
                y_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
                accuracy = np.mean(pred == y_class)
                self.performance_history[i].append(accuracy)
            except Exception as e:
                logger.warning(f"Error updating weights for model {i}: {e}")

        # Calculate new weights based on recent performance
        recent_performance = np.array([
            np.mean(list(history)) if len(history) > 0 else 0.5
            for history in self.performance_history
        ])

        # Softmax weighting
        exp_perf = np.exp(recent_performance * 5)  # Temperature = 0.2
        self.weights = exp_perf / np.sum(exp_perf)

        # Apply minimum weight
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights /= np.sum(self.weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted probability predictions"""
        predictions = []

        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict_proba(X)
                predictions.append(pred * weight)
            except Exception as e:
                logger.warning(f"Error predicting: {e}")

        return np.sum(predictions, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_model_weights(self) -> Dict[int, float]:
        """Get current model weights"""
        return {i: w for i, w in enumerate(self.weights)}


class DiversityEnsemble:
    """
    Ensemble that maximizes model diversity.
    Uses negative correlation learning.
    """

    def __init__(
        self,
        base_model_fn: Callable,
        n_estimators: int = 5,
        diversity_weight: float = 0.5
    ):
        self.base_model_fn = base_model_fn
        self.n_estimators = n_estimators
        self.diversity_weight = diversity_weight

        self.models = []

        logger.info(f"DiversityEnsemble initialized")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DiversityEnsemble':
        """Train ensemble with diversity penalty"""
        for i in range(self.n_estimators):
            model = self.base_model_fn()

            if i == 0:
                # First model trains normally
                model.fit(X, y)
            else:
                # Subsequent models train with diversity penalty
                # Get predictions from existing models
                existing_preds = np.mean([
                    m.predict_proba(X) for m in self.models
                ], axis=0)

                # Modify targets to encourage diversity
                diversity_target = y - self.diversity_weight * (existing_preds - 0.5)
                diversity_target = np.clip(diversity_target, 0, 1)
                diversity_target /= np.sum(diversity_target, axis=1, keepdims=True)

                model.fit(X, diversity_target)

            self.models.append(model)
            logger.debug(f"Trained diverse model {i+1}/{self.n_estimators}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        predictions = [model.predict_proba(X) for model in self.models]
        return np.mean(predictions, axis=0)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return np.argmax(self.predict_proba(X), axis=1)

    def calculate_diversity(self, X: np.ndarray) -> float:
        """Calculate ensemble diversity"""
        predictions = [model.predict_class(X) for model in self.models]
        predictions = np.array(predictions)

        # Calculate pairwise disagreement
        n_pairs = 0
        total_disagreement = 0

        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                disagreement = np.mean(predictions[i] != predictions[j])
                total_disagreement += disagreement
                n_pairs += 1

        return total_disagreement / n_pairs if n_pairs > 0 else 0


class EnsembleSelector:
    """
    Automatically selects and combines the best ensemble configuration.
    Uses validation performance to guide selection.
    """

    def __init__(
        self,
        candidate_ensembles: List[Any],
        validation_ratio: float = 0.2
    ):
        self.candidate_ensembles = candidate_ensembles
        self.validation_ratio = validation_ratio

        self.best_ensemble = None
        self.best_score = 0
        self.ensemble_scores = {}

        logger.info(f"EnsembleSelector initialized with {len(candidate_ensembles)} candidates")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleSelector':
        """Train and select best ensemble"""
        # Split data
        n_val = int(len(X) * self.validation_ratio)
        indices = np.random.permutation(len(X))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Evaluate each candidate
        for i, ensemble in enumerate(self.candidate_ensembles):
            try:
                ensemble.fit(X_train, y_train)
                pred = ensemble.predict_class(X_val)
                y_class = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
                score = np.mean(pred == y_class)

                self.ensemble_scores[i] = score

                if score > self.best_score:
                    self.best_score = score
                    self.best_ensemble = ensemble

                logger.info(f"Ensemble {i}: score = {score:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating ensemble {i}: {e}")
                self.ensemble_scores[i] = 0

        # Retrain best ensemble on full data
        if self.best_ensemble is not None:
            self.best_ensemble.fit(X, y)

        logger.info(f"Best ensemble selected with score {self.best_score:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from best ensemble"""
        if self.best_ensemble is None:
            raise ValueError("No ensemble selected. Call fit() first.")
        return self.best_ensemble.predict_proba(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        return np.argmax(self.predict_proba(X), axis=1)
