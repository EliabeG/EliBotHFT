"""
Machine Learning Engine for Error Learning

This module implements the core ML algorithms for learning from trading errors
and improving future performance. It includes:

1. Supervised Learning: Error classification and outcome prediction
2. Reinforcement Learning: Strategy weight optimization
3. Online Learning: Continuous model updates
4. Ensemble Methods: Combining multiple models

The engine learns:
- Which conditions lead to specific errors
- Optimal strategy weights for different regimes
- Position sizing based on error risk
- Signal confidence calibration
"""

import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque
import json
import pickle

from .error_classifier import TradeError, ErrorType, ErrorSeverity
from .feature_extractor import TradeFeatures, MarketFeatures

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models"""
    ERROR_CLASSIFIER = "error_classifier"
    OUTCOME_PREDICTOR = "outcome_predictor"
    RISK_SCORER = "risk_scorer"
    STRATEGY_SELECTOR = "strategy_selector"
    POSITION_SIZER = "position_sizer"
    CONFIDENCE_CALIBRATOR = "confidence_calibrator"


@dataclass
class LearningConfig:
    """Configuration for the learning engine"""

    # Learning rates
    base_learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    learning_rate_decay: float = 0.995

    # Experience replay
    replay_buffer_size: int = 10000
    min_samples_to_train: int = 100
    batch_size: int = 32

    # Reinforcement learning
    discount_factor: float = 0.95
    exploration_rate: float = 0.2
    exploration_decay: float = 0.99
    min_exploration: float = 0.05

    # Model parameters
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float = 0.2
    regularization: float = 0.001

    # Training
    training_interval: int = 50  # Train every N experiences
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Ensemble
    num_ensemble_models: int = 5
    ensemble_method: str = "weighted_average"  # or "voting"

    # Online learning
    online_learning_enabled: bool = True
    online_update_frequency: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            'base_learning_rate': self.base_learning_rate,
            'min_learning_rate': self.min_learning_rate,
            'learning_rate_decay': self.learning_rate_decay,
            'replay_buffer_size': self.replay_buffer_size,
            'min_samples_to_train': self.min_samples_to_train,
            'batch_size': self.batch_size,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration': self.min_exploration,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'regularization': self.regularization,
            'training_interval': self.training_interval,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'num_ensemble_models': self.num_ensemble_models,
            'ensemble_method': self.ensemble_method,
            'online_learning_enabled': self.online_learning_enabled,
            'online_update_frequency': self.online_update_frequency,
        }


@dataclass
class Experience:
    """Represents a single learning experience"""
    state: np.ndarray          # Feature vector at time of action
    action: int                # Action taken (strategy index, signal decision, etc.)
    reward: float              # Reward received (PnL, error penalty, etc.)
    next_state: np.ndarray     # State after action
    done: bool                 # Episode ended
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingResult:
    """Result of a training session"""
    model_type: ModelType
    loss: float
    accuracy: float
    validation_loss: float
    validation_accuracy: float
    epochs_trained: int
    samples_used: int
    training_time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


class SimpleNeuralNetwork:
    """
    Simple neural network implementation using NumPy.
    This is a fallback when PyTorch/TensorFlow are not available.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate: float = 0.01,
        dropout_rate: float = 0.2
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Adam optimizer state
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through the network"""
        activations = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b

            if i < len(self.weights) - 1:
                # Hidden layer with ReLU
                a = self.relu(z)

                # Dropout during training
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                    a = a * mask
            else:
                # Output layer with softmax for classification
                a = self.softmax(z)

            activations.append(a)

        return activations[-1], activations

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward pass - compute gradients"""
        m = X.shape[0]
        dweights = []
        dbiases = []

        # Output layer gradient (cross-entropy with softmax)
        delta = activations[-1] - y

        for i in reversed(range(len(self.weights))):
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            dweights.insert(0, dw)
            dbiases.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])

        return dweights, dbiases

    def update_weights(self, dweights: List[np.ndarray], dbiases: List[np.ndarray]):
        """Update weights using Adam optimizer"""
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for i in range(len(self.weights)):
            # Update momentum
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * dweights[i]
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * dbiases[i]

            # Update velocity
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (dweights[i] ** 2)
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (dbiases[i] ** 2)

            # Bias correction
            m_w_hat = self.m_weights[i] / (1 - beta1 ** self.t)
            m_b_hat = self.m_biases[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_weights[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_biases[i] / (1 - beta2 ** self.t)

            # Update weights
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on a batch of data"""
        output, activations = self.forward(X, training=True)
        dweights, dbiases = self.backward(X, y, activations)
        self.update_weights(dweights, dbiases)

        # Calculate loss (cross-entropy)
        epsilon = 1e-15
        loss = -np.mean(np.sum(y * np.log(output + epsilon), axis=1))
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        output, _ = self.forward(X, training=False)
        return output

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)


class ReinforcementLearner:
    """
    Reinforcement learning for strategy weight optimization.
    Uses Q-learning with function approximation.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: LearningConfig
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Q-network
        self.q_network = SimpleNeuralNetwork(
            input_size=state_size,
            hidden_layers=config.hidden_layers,
            output_size=action_size,
            learning_rate=config.base_learning_rate
        )

        # Target network (for stable learning)
        self.target_network = SimpleNeuralNetwork(
            input_size=state_size,
            hidden_layers=config.hidden_layers,
            output_size=action_size,
            learning_rate=config.base_learning_rate
        )
        self._copy_weights(self.q_network, self.target_network)

        # Experience replay
        self.replay_buffer: deque = deque(maxlen=config.replay_buffer_size)

        # Training state
        self.exploration_rate = config.exploration_rate
        self.training_steps = 0
        self.target_update_freq = 100

    def _copy_weights(self, source: SimpleNeuralNetwork, target: SimpleNeuralNetwork):
        """Copy weights from source to target network"""
        for i in range(len(source.weights)):
            target.weights[i] = source.weights[i].copy()
            target.biases[i] = source.biases[i].copy()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)

        q_values = self.q_network.predict(state.reshape(1, -1))[0]
        return int(np.argmax(q_values))

    def store_experience(self, experience: Experience):
        """Store experience in replay buffer"""
        self.replay_buffer.append(experience)

    def train(self) -> Optional[float]:
        """Train on a batch of experiences"""
        if len(self.replay_buffer) < self.config.min_samples_to_train:
            return None

        # Sample batch
        batch_indices = np.random.choice(
            len(self.replay_buffer),
            min(self.config.batch_size, len(self.replay_buffer)),
            replace=False
        )
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Prepare data
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Calculate target Q-values
        current_q = self.q_network.predict(states)
        next_q = self.target_network.predict(next_states)

        # Double DQN: use main network to select actions
        next_actions = np.argmax(self.q_network.predict(next_states), axis=1)

        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.config.discount_factor * next_q[i, next_actions[i]]

        # Train
        loss = self.q_network.train_batch(states, targets)

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self._copy_weights(self.q_network, self.target_network)

        # Decay exploration
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay
        )

        return loss

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        return self.q_network.predict(state.reshape(1, -1))[0]


class LearningEngine:
    """
    Main machine learning engine that coordinates all learning components.

    This engine:
    1. Collects and processes trading experiences
    2. Trains multiple models for different purposes
    3. Provides predictions for error risk, strategy selection, etc.
    4. Continuously improves through online learning
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()

        # Feature dimensions (will be set when first data arrives)
        self.feature_size: Optional[int] = None
        self.num_error_types = len(ErrorType)
        self.num_strategies = 6  # From existing codebase

        # Models
        self.error_classifier: Optional[SimpleNeuralNetwork] = None
        self.outcome_predictor: Optional[SimpleNeuralNetwork] = None
        self.risk_scorer: Optional[SimpleNeuralNetwork] = None
        self.rl_learner: Optional[ReinforcementLearner] = None

        # Ensemble models
        self.ensemble_models: Dict[ModelType, List[SimpleNeuralNetwork]] = {}

        # Experience storage
        self.experiences: deque = deque(maxlen=self.config.replay_buffer_size)
        self.error_experiences: List[Tuple[np.ndarray, int]] = []  # (features, error_type)
        self.outcome_experiences: List[Tuple[np.ndarray, float]] = []  # (features, outcome)

        # Training state
        self.experience_count = 0
        self.training_count = 0
        self.current_learning_rate = self.config.base_learning_rate

        # Performance tracking
        self.training_history: List[TrainingResult] = []
        self.prediction_accuracy: Dict[ModelType, deque] = {
            model_type: deque(maxlen=100)
            for model_type in ModelType
        }

        # Feature normalization
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        # Callbacks
        self.on_training_complete: Optional[Callable[[TrainingResult], None]] = None
        self.on_prediction: Optional[Callable[[ModelType, Any], None]] = None

        logger.info("LearningEngine initialized")

    def _initialize_models(self, feature_size: int):
        """Initialize all models with proper dimensions"""
        self.feature_size = feature_size

        # Error classifier: predicts error type
        self.error_classifier = SimpleNeuralNetwork(
            input_size=feature_size,
            hidden_layers=self.config.hidden_layers,
            output_size=self.num_error_types,
            learning_rate=self.config.base_learning_rate,
            dropout_rate=self.config.dropout_rate
        )

        # Outcome predictor: predicts success/failure
        self.outcome_predictor = SimpleNeuralNetwork(
            input_size=feature_size,
            hidden_layers=self.config.hidden_layers,
            output_size=2,  # Binary: success/failure
            learning_rate=self.config.base_learning_rate,
            dropout_rate=self.config.dropout_rate
        )

        # Risk scorer: predicts risk level
        self.risk_scorer = SimpleNeuralNetwork(
            input_size=feature_size,
            hidden_layers=self.config.hidden_layers,
            output_size=5,  # 5 severity levels
            learning_rate=self.config.base_learning_rate,
            dropout_rate=self.config.dropout_rate
        )

        # RL learner for strategy selection
        # State: market features, Action: strategy weights adjustment
        self.rl_learner = ReinforcementLearner(
            state_size=feature_size,
            action_size=self.num_strategies * 3,  # 3 actions per strategy: increase/decrease/keep
            config=self.config
        )

        # Initialize ensembles
        for model_type in [ModelType.ERROR_CLASSIFIER, ModelType.OUTCOME_PREDICTOR]:
            self.ensemble_models[model_type] = [
                SimpleNeuralNetwork(
                    input_size=feature_size,
                    hidden_layers=self.config.hidden_layers,
                    output_size=self.num_error_types if model_type == ModelType.ERROR_CLASSIFIER else 2,
                    learning_rate=self.config.base_learning_rate * (0.5 + np.random.random()),
                    dropout_rate=self.config.dropout_rate
                )
                for _ in range(self.config.num_ensemble_models)
            ]

        logger.info(f"Models initialized with feature size {feature_size}")

    def add_trade_experience(
        self,
        features: np.ndarray,
        error: Optional[TradeError],
        pnl: float,
        strategy_index: int
    ):
        """
        Add a trade experience for learning.

        Args:
            features: Feature vector at time of trade
            error: TradeError if trade had an error, None otherwise
            pnl: Profit/loss from the trade
            strategy_index: Index of strategy that generated signal
        """
        # Initialize models if needed
        if self.feature_size is None:
            self._initialize_models(len(features))

        # Normalize features
        features = self._normalize_features(features)

        # Store for supervised learning
        if error:
            error_type_idx = list(ErrorType).index(error.error_type)
            self.error_experiences.append((features, error_type_idx))

        # Store outcome experience
        outcome = 1 if pnl > 0 else 0
        self.outcome_experiences.append((features, outcome))

        # Store for RL
        # Reward = normalized PnL with penalty for errors
        reward = pnl / 100.0  # Normalize
        if error:
            reward -= error.severity.value * 0.1  # Penalty for errors

        experience = Experience(
            state=features,
            action=strategy_index,
            reward=reward,
            next_state=features,  # Will be updated with next state
            done=False
        )
        self.experiences.append(experience)

        if self.rl_learner:
            self.rl_learner.store_experience(experience)

        self.experience_count += 1

        # Train periodically
        if self.experience_count % self.config.training_interval == 0:
            self._train_all_models()

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics"""
        features = np.array(features).flatten()

        if self.feature_means is None:
            self.feature_means = features.copy()
            self.feature_stds = np.ones_like(features)
            return features

        # Update running statistics
        n = self.experience_count + 1
        old_mean = self.feature_means
        self.feature_means = old_mean + (features - old_mean) / n

        if n > 1:
            self.feature_stds = np.sqrt(
                ((n - 2) * self.feature_stds ** 2 + (features - old_mean) * (features - self.feature_means))
                / (n - 1)
            )
            self.feature_stds = np.maximum(self.feature_stds, 1e-8)  # Avoid division by zero

        # Normalize
        return (features - self.feature_means) / self.feature_stds

    def _train_all_models(self):
        """Train all models on accumulated experience"""
        start_time = datetime.now()

        results = []

        # Train error classifier
        if len(self.error_experiences) >= self.config.min_samples_to_train:
            result = self._train_error_classifier()
            if result:
                results.append(result)

        # Train outcome predictor
        if len(self.outcome_experiences) >= self.config.min_samples_to_train:
            result = self._train_outcome_predictor()
            if result:
                results.append(result)

        # Train RL learner
        if self.rl_learner:
            rl_loss = self.rl_learner.train()
            if rl_loss is not None:
                logger.debug(f"RL training loss: {rl_loss:.4f}")

        # Update learning rate
        self.current_learning_rate = max(
            self.config.min_learning_rate,
            self.current_learning_rate * self.config.learning_rate_decay
        )

        self.training_count += 1

        for result in results:
            self.training_history.append(result)
            if self.on_training_complete:
                self.on_training_complete(result)

        logger.info(
            f"Training iteration {self.training_count} completed "
            f"({len(results)} models trained)"
        )

    def _train_error_classifier(self) -> Optional[TrainingResult]:
        """Train the error classification model"""
        if not self.error_classifier:
            return None

        # Prepare data
        X = np.array([x[0] for x in self.error_experiences])
        y_indices = np.array([x[1] for x in self.error_experiences])

        # One-hot encode labels
        y = np.zeros((len(y_indices), self.num_error_types))
        y[np.arange(len(y_indices)), y_indices] = 1

        # Split train/validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train
        start_time = datetime.now()
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(100):  # Max epochs
            # Shuffle
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Train in batches
            epoch_losses = []
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_shuffled[i:i + self.config.batch_size]
                batch_y = y_shuffled[i:i + self.config.batch_size]
                loss = self.error_classifier.train_batch(batch_X, batch_y)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            # Validation
            val_pred = self.error_classifier.predict(X_val)
            val_loss = -np.mean(np.sum(y_val * np.log(val_pred + 1e-15), axis=1))

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        # Calculate final metrics
        train_pred = self.error_classifier.predict_class(X_train)
        train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))

        val_pred = self.error_classifier.predict_class(X_val)
        val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))

        training_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResult(
            model_type=ModelType.ERROR_CLASSIFIER,
            loss=float(np.mean(losses[-10:])),
            accuracy=float(train_acc),
            validation_loss=float(best_val_loss),
            validation_accuracy=float(val_acc),
            epochs_trained=epoch + 1,
            samples_used=len(X_train),
            training_time_ms=training_time
        )

    def _train_outcome_predictor(self) -> Optional[TrainingResult]:
        """Train the outcome prediction model"""
        if not self.outcome_predictor:
            return None

        # Prepare data
        X = np.array([x[0] for x in self.outcome_experiences])
        y_raw = np.array([x[1] for x in self.outcome_experiences])

        # One-hot encode
        y = np.zeros((len(y_raw), 2))
        y[np.arange(len(y_raw)), y_raw.astype(int)] = 1

        # Split
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train
        start_time = datetime.now()
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(100):
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_shuffled[i:i + self.config.batch_size]
                batch_y = y_shuffled[i:i + self.config.batch_size]
                loss = self.outcome_predictor.train_batch(batch_X, batch_y)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            # Validation
            val_pred = self.outcome_predictor.predict(X_val)
            val_loss = -np.mean(np.sum(y_val * np.log(val_pred + 1e-15), axis=1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        # Final metrics
        train_pred = self.outcome_predictor.predict_class(X_train)
        train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))

        val_pred = self.outcome_predictor.predict_class(X_val)
        val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))

        training_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return TrainingResult(
            model_type=ModelType.OUTCOME_PREDICTOR,
            loss=float(np.mean(losses[-10:])),
            accuracy=float(train_acc),
            validation_loss=float(best_val_loss),
            validation_accuracy=float(val_acc),
            epochs_trained=epoch + 1,
            samples_used=len(X_train),
            training_time_ms=training_time
        )

    def predict_error_type(self, features: np.ndarray) -> List[Tuple[ErrorType, float]]:
        """
        Predict likely error types for given features.

        Args:
            features: Feature vector

        Returns:
            List of (ErrorType, probability) tuples, sorted by probability
        """
        if self.error_classifier is None or self.feature_size is None:
            return []

        features = self._normalize_features(features)
        probs = self.error_classifier.predict(features.reshape(1, -1))[0]

        predictions = [
            (error_type, float(prob))
            for error_type, prob in zip(ErrorType, probs)
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        if self.on_prediction:
            self.on_prediction(ModelType.ERROR_CLASSIFIER, predictions)

        return predictions

    def predict_outcome(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict trade outcome (success/failure).

        Args:
            features: Feature vector

        Returns:
            Tuple of (predicted_success, confidence)
        """
        if self.outcome_predictor is None or self.feature_size is None:
            return True, 0.5

        features = self._normalize_features(features)
        probs = self.outcome_predictor.predict(features.reshape(1, -1))[0]

        success = probs[1] > probs[0]
        confidence = float(max(probs))

        if self.on_prediction:
            self.on_prediction(ModelType.OUTCOME_PREDICTOR, (success, confidence))

        return success, confidence

    def predict_error_risk(self, features: np.ndarray) -> float:
        """
        Predict overall error risk (0-1).

        Args:
            features: Feature vector

        Returns:
            Risk score between 0 and 1
        """
        if self.error_classifier is None:
            return 0.5

        # Get error type predictions
        error_probs = self.predict_error_type(features)

        # Weight by severity
        risk = 0.0
        for error_type, prob in error_probs:
            # Assign severity weights
            severity_weights = {
                ErrorType.PREDICTION_ERROR: 0.8,
                ErrorType.FALSE_BREAKOUT: 0.7,
                ErrorType.SLIPPAGE_ERROR: 0.6,
                ErrorType.STOP_TOO_TIGHT: 0.6,
                ErrorType.VOLATILITY_SPIKE: 0.7,
                ErrorType.REGIME_MISMATCH: 0.5,
            }
            weight = severity_weights.get(error_type, 0.5)
            risk += prob * weight

        # Also factor in outcome prediction
        success, confidence = self.predict_outcome(features)
        if not success:
            risk = risk * 0.5 + (1 - confidence) * 0.5

        return min(1.0, risk)

    def get_strategy_action(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Get recommended strategy action using RL.

        Args:
            features: Current market state features

        Returns:
            Tuple of (action_index, q_values)
        """
        if self.rl_learner is None or self.feature_size is None:
            return 0, np.zeros(self.num_strategies * 3)

        features = self._normalize_features(features)
        action = self.rl_learner.select_action(features, training=True)
        q_values = self.rl_learner.get_q_values(features)

        return action, q_values

    def get_strategy_weight_adjustments(self, features: np.ndarray) -> Dict[int, float]:
        """
        Get recommended weight adjustments for each strategy.

        Args:
            features: Current market state

        Returns:
            Dict of strategy_index -> adjustment (-1, 0, or 1)
        """
        action, q_values = self.get_strategy_action(features)

        adjustments = {}
        for i in range(self.num_strategies):
            # Each strategy has 3 actions: decrease (0), keep (1), increase (2)
            strategy_q = q_values[i * 3:(i + 1) * 3]
            best_action = np.argmax(strategy_q)

            if best_action == 0:
                adjustments[i] = -0.1  # Decrease weight
            elif best_action == 2:
                adjustments[i] = 0.1   # Increase weight
            else:
                adjustments[i] = 0.0   # Keep same

        return adjustments

    def ensemble_predict_error(self, features: np.ndarray) -> List[Tuple[ErrorType, float]]:
        """
        Predict error using ensemble of models.

        Args:
            features: Feature vector

        Returns:
            List of (ErrorType, probability) from ensemble
        """
        if ModelType.ERROR_CLASSIFIER not in self.ensemble_models:
            return self.predict_error_type(features)

        models = self.ensemble_models[ModelType.ERROR_CLASSIFIER]
        if not models:
            return self.predict_error_type(features)

        features = self._normalize_features(features)
        all_probs = []

        for model in models:
            probs = model.predict(features.reshape(1, -1))[0]
            all_probs.append(probs)

        # Average predictions
        avg_probs = np.mean(all_probs, axis=0)

        predictions = [
            (error_type, float(prob))
            for error_type, prob in zip(ErrorType, avg_probs)
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'experience_count': self.experience_count,
            'training_count': self.training_count,
            'current_learning_rate': self.current_learning_rate,
            'error_samples': len(self.error_experiences),
            'outcome_samples': len(self.outcome_experiences),
            'replay_buffer_size': len(self.experiences),
        }

        if self.rl_learner:
            stats['rl'] = {
                'exploration_rate': self.rl_learner.exploration_rate,
                'training_steps': self.rl_learner.training_steps,
                'replay_buffer': len(self.rl_learner.replay_buffer)
            }

        # Recent training results
        if self.training_history:
            recent = self.training_history[-10:]
            stats['recent_training'] = {
                'avg_loss': np.mean([r.loss for r in recent]),
                'avg_accuracy': np.mean([r.accuracy for r in recent]),
                'avg_val_accuracy': np.mean([r.validation_accuracy for r in recent]),
            }

        return stats

    def save_models(self, filepath: str):
        """Save all models to file"""
        data = {
            'config': self.config.to_dict(),
            'feature_size': self.feature_size,
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None,
            'experience_count': self.experience_count,
            'training_count': self.training_count,
            'current_learning_rate': self.current_learning_rate,
        }

        # Save model weights
        if self.error_classifier:
            data['error_classifier'] = {
                'weights': [w.tolist() for w in self.error_classifier.weights],
                'biases': [b.tolist() for b in self.error_classifier.biases]
            }

        if self.outcome_predictor:
            data['outcome_predictor'] = {
                'weights': [w.tolist() for w in self.outcome_predictor.weights],
                'biases': [b.tolist() for b in self.outcome_predictor.biases]
            }

        if self.rl_learner:
            data['rl_learner'] = {
                'q_weights': [w.tolist() for w in self.rl_learner.q_network.weights],
                'q_biases': [b.tolist() for b in self.rl_learner.q_network.biases],
                'exploration_rate': self.rl_learner.exploration_rate,
                'training_steps': self.rl_learner.training_steps
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load models from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.feature_size = data.get('feature_size')
        if data.get('feature_means'):
            self.feature_means = np.array(data['feature_means'])
        if data.get('feature_stds'):
            self.feature_stds = np.array(data['feature_stds'])

        self.experience_count = data.get('experience_count', 0)
        self.training_count = data.get('training_count', 0)
        self.current_learning_rate = data.get('current_learning_rate', self.config.base_learning_rate)

        # Initialize models if we have feature size
        if self.feature_size:
            self._initialize_models(self.feature_size)

            # Load error classifier weights
            if 'error_classifier' in data and self.error_classifier:
                ec_data = data['error_classifier']
                self.error_classifier.weights = [np.array(w) for w in ec_data['weights']]
                self.error_classifier.biases = [np.array(b) for b in ec_data['biases']]

            # Load outcome predictor weights
            if 'outcome_predictor' in data and self.outcome_predictor:
                op_data = data['outcome_predictor']
                self.outcome_predictor.weights = [np.array(w) for w in op_data['weights']]
                self.outcome_predictor.biases = [np.array(b) for b in op_data['biases']]

            # Load RL learner
            if 'rl_learner' in data and self.rl_learner:
                rl_data = data['rl_learner']
                self.rl_learner.q_network.weights = [np.array(w) for w in rl_data['q_weights']]
                self.rl_learner.q_network.biases = [np.array(b) for b in rl_data['q_biases']]
                self.rl_learner.exploration_rate = rl_data.get('exploration_rate', self.config.exploration_rate)
                self.rl_learner.training_steps = rl_data.get('training_steps', 0)
                self.rl_learner._copy_weights(self.rl_learner.q_network, self.rl_learner.target_network)

        logger.info(f"Models loaded from {filepath}")

    def reset(self):
        """Reset all learning state"""
        self.experiences.clear()
        self.error_experiences.clear()
        self.outcome_experiences.clear()
        self.experience_count = 0
        self.training_count = 0
        self.current_learning_rate = self.config.base_learning_rate
        self.feature_means = None
        self.feature_stds = None
        self.training_history.clear()

        # Reinitialize models if feature size is known
        if self.feature_size:
            self._initialize_models(self.feature_size)

        logger.info("LearningEngine reset")
