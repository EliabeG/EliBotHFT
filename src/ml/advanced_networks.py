"""
Advanced Neural Networks for HFT Trading

This module implements sophisticated neural network architectures:
1. LSTM-like Recurrent Networks for sequence modeling
2. Attention Mechanisms for feature importance
3. Transformer-inspired architectures
4. Residual connections for deep networks
5. Dropout and regularization for robustness
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import json

logger = logging.getLogger(__name__)


class ActivationFunction(Enum):
    """Supported activation functions"""
    RELU = auto()
    LEAKY_RELU = auto()
    ELU = auto()
    SELU = auto()
    TANH = auto()
    SIGMOID = auto()
    SOFTMAX = auto()
    GELU = auto()
    SWISH = auto()


@dataclass
class NetworkConfig:
    """Configuration for neural networks"""
    input_size: int = 50
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_size: int = 10
    activation: ActivationFunction = ActivationFunction.RELU
    output_activation: ActivationFunction = ActivationFunction.SOFTMAX
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    use_residual: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0


def apply_activation(x: np.ndarray, activation: ActivationFunction) -> np.ndarray:
    """Apply activation function"""
    if activation == ActivationFunction.RELU:
        return np.maximum(0, x)
    elif activation == ActivationFunction.LEAKY_RELU:
        return np.where(x > 0, x, 0.01 * x)
    elif activation == ActivationFunction.ELU:
        return np.where(x > 0, x, np.exp(x) - 1)
    elif activation == ActivationFunction.SELU:
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elif activation == ActivationFunction.TANH:
        return np.tanh(x)
    elif activation == ActivationFunction.SIGMOID:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation == ActivationFunction.SOFTMAX:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    elif activation == ActivationFunction.GELU:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    elif activation == ActivationFunction.SWISH:
        return x * (1 / (1 + np.exp(-x)))
    return x


def activation_derivative(x: np.ndarray, activation: ActivationFunction) -> np.ndarray:
    """Compute activation derivative"""
    if activation == ActivationFunction.RELU:
        return (x > 0).astype(float)
    elif activation == ActivationFunction.LEAKY_RELU:
        return np.where(x > 0, 1, 0.01)
    elif activation == ActivationFunction.ELU:
        return np.where(x > 0, 1, np.exp(x))
    elif activation == ActivationFunction.TANH:
        return 1 - np.tanh(x)**2
    elif activation == ActivationFunction.SIGMOID:
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)
    elif activation == ActivationFunction.GELU:
        # Approximation
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
               0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
               np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    return np.ones_like(x)


class BatchNormalization:
    """Batch Normalization layer"""

    def __init__(self, size: int, momentum: float = 0.9, epsilon: float = 1e-5):
        self.size = size
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)

        # Running statistics
        self.running_mean = np.zeros(size)
        self.running_var = np.ones(size)

        # Cache for backward pass
        self.cache = {}

        # Gradients
        self.d_gamma = np.zeros(size)
        self.d_beta = np.zeros(size)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.epsilon)

            # Cache for backward
            self.cache = {
                'x': x, 'mean': mean, 'var': var, 'x_norm': x_norm
            }
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * x_norm + self.beta

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass"""
        x = self.cache['x']
        mean = self.cache['mean']
        var = self.cache['var']
        x_norm = self.cache['x_norm']

        N = x.shape[0]

        # Gradients for gamma and beta
        self.d_gamma = np.sum(d_out * x_norm, axis=0)
        self.d_beta = np.sum(d_out, axis=0)

        # Gradient for input
        dx_norm = d_out * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.epsilon), axis=0) + \
                dvar * np.sum(-2 * (x - mean), axis=0) / N

        dx = dx_norm / np.sqrt(var + self.epsilon) + \
             dvar * 2 * (x - mean) / N + dmean / N

        return dx


class DropoutLayer:
    """Dropout layer for regularization"""

    def __init__(self, rate: float = 0.2):
        self.rate = rate
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout"""
        if training and self.rate > 0:
            self.mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * self.mask
        return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass"""
        if self.mask is not None:
            return d_out * self.mask
        return d_out


class AttentionLayer:
    """
    Self-Attention mechanism for feature importance weighting.
    Implements scaled dot-product attention.
    """

    def __init__(self, input_size: int, num_heads: int = 4, head_dim: int = 16):
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        # Initialize weights for Q, K, V projections
        scale = np.sqrt(2.0 / input_size)
        self.W_q = np.random.randn(input_size, self.total_dim) * scale
        self.W_k = np.random.randn(input_size, self.total_dim) * scale
        self.W_v = np.random.randn(input_size, self.total_dim) * scale
        self.W_o = np.random.randn(self.total_dim, input_size) * scale

        # Gradients
        self.d_W_q = np.zeros_like(self.W_q)
        self.d_W_k = np.zeros_like(self.W_k)
        self.d_W_v = np.zeros_like(self.W_v)
        self.d_W_o = np.zeros_like(self.W_o)

        # Cache
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of attention.
        x: (batch_size, seq_len, input_size) or (batch_size, input_size)
        """
        # Handle 2D input by adding sequence dimension
        if x.ndim == 2:
            x = x[:, np.newaxis, :]

        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = x @ self.W_q  # (batch, seq, total_dim)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)

        # Softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

        # Apply attention to values
        context = attention_weights @ V  # (batch, heads, seq, head_dim)

        # Reshape and project
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.total_dim)
        output = context @ self.W_o

        # Cache for backward
        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V,
            'attention_weights': attention_weights, 'context': context
        }

        # Remove sequence dimension if input was 2D
        if seq_len == 1:
            output = output.squeeze(1)

        return output

    def get_attention_weights(self) -> np.ndarray:
        """Get attention weights from last forward pass"""
        return self.cache.get('attention_weights', None)


class LSTMCell:
    """
    LSTM Cell implementation for sequence modeling.
    Captures temporal dependencies in market data.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights (Xavier/Glorot)
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Gates: input, forget, cell, output
        self.W_i = np.random.randn(input_size, hidden_size) * scale
        self.W_f = np.random.randn(input_size, hidden_size) * scale
        self.W_c = np.random.randn(input_size, hidden_size) * scale
        self.W_o = np.random.randn(input_size, hidden_size) * scale

        self.U_i = np.random.randn(hidden_size, hidden_size) * scale
        self.U_f = np.random.randn(hidden_size, hidden_size) * scale
        self.U_c = np.random.randn(hidden_size, hidden_size) * scale
        self.U_o = np.random.randn(hidden_size, hidden_size) * scale

        self.b_i = np.zeros(hidden_size)
        self.b_f = np.ones(hidden_size)  # Initialize forget gate bias to 1
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

        # Gradients
        self.d_W_i = np.zeros_like(self.W_i)
        self.d_W_f = np.zeros_like(self.W_f)
        self.d_W_c = np.zeros_like(self.W_c)
        self.d_W_o = np.zeros_like(self.W_o)

        self.d_U_i = np.zeros_like(self.U_i)
        self.d_U_f = np.zeros_like(self.U_f)
        self.d_U_c = np.zeros_like(self.U_c)
        self.d_U_o = np.zeros_like(self.U_o)

        self.d_b_i = np.zeros_like(self.b_i)
        self.d_b_f = np.zeros_like(self.b_f)
        self.d_b_c = np.zeros_like(self.b_c)
        self.d_b_o = np.zeros_like(self.b_o)

    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of LSTM cell.

        Args:
            x: Input (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)

        Returns:
            h_next, c_next
        """
        # Gates
        i = self._sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        f = self._sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)
        o = self._sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)

        # Candidate cell state
        c_tilde = np.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)

        # New cell state
        c_next = f * c_prev + i * c_tilde

        # New hidden state
        h_next = o * np.tanh(c_next)

        return h_next, c_next

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRUCell:
    """
    GRU Cell - simpler alternative to LSTM.
    Faster training while maintaining good performance.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Reset and update gates
        self.W_r = np.random.randn(input_size, hidden_size) * scale
        self.W_z = np.random.randn(input_size, hidden_size) * scale
        self.W_h = np.random.randn(input_size, hidden_size) * scale

        self.U_r = np.random.randn(hidden_size, hidden_size) * scale
        self.U_z = np.random.randn(hidden_size, hidden_size) * scale
        self.U_h = np.random.randn(hidden_size, hidden_size) * scale

        self.b_r = np.zeros(hidden_size)
        self.b_z = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Forward pass"""
        r = self._sigmoid(x @ self.W_r + h_prev @ self.U_r + self.b_r)
        z = self._sigmoid(x @ self.W_z + h_prev @ self.U_z + self.b_z)

        h_tilde = np.tanh(x @ self.W_h + (r * h_prev) @ self.U_h + self.b_h)

        h_next = (1 - z) * h_prev + z * h_tilde

        return h_next

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class ResidualBlock:
    """
    Residual block with skip connections.
    Enables training of deeper networks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: ActivationFunction = ActivationFunction.RELU,
        dropout_rate: float = 0.1
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        # Two-layer residual block
        scale = np.sqrt(2.0 / input_size)
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, input_size) * scale
        self.b2 = np.zeros(input_size)

        # Batch norm
        self.bn1 = BatchNormalization(hidden_size)
        self.bn2 = BatchNormalization(input_size)

        # Dropout
        self.dropout = DropoutLayer(dropout_rate)

        # Gradients
        self.d_W1 = np.zeros_like(self.W1)
        self.d_b1 = np.zeros_like(self.b1)
        self.d_W2 = np.zeros_like(self.W2)
        self.d_b2 = np.zeros_like(self.b2)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with skip connection"""
        # First layer
        h = x @ self.W1 + self.b1
        h = self.bn1.forward(h, training)
        h = apply_activation(h, self.activation)
        h = self.dropout.forward(h, training)

        # Second layer
        h = h @ self.W2 + self.b2
        h = self.bn2.forward(h, training)

        # Skip connection
        out = x + h
        out = apply_activation(out, self.activation)

        return out


class DeepNetwork:
    """
    Deep Neural Network with advanced features:
    - Residual connections
    - Batch normalization
    - Dropout regularization
    - Multiple activation functions
    - Attention mechanism
    """

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.layers = []
        self.batch_norms = []
        self.dropouts = []
        self.attention = None

        # Build network
        self._build_network()

        # Adam optimizer state
        self.m = []  # First moment
        self.v = []  # Second moment
        self.t = 0   # Time step

        # Initialize optimizer state
        self._init_optimizer()

        logger.info(f"DeepNetwork initialized: {config.input_size} -> {config.output_size}")

    def _build_network(self):
        """Build network layers"""
        sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]

        for i in range(len(sizes) - 1):
            # Weight initialization (He initialization for ReLU)
            if self.config.activation in [ActivationFunction.RELU, ActivationFunction.LEAKY_RELU]:
                scale = np.sqrt(2.0 / sizes[i])
            else:
                scale = np.sqrt(2.0 / (sizes[i] + sizes[i+1]))

            W = np.random.randn(sizes[i], sizes[i+1]) * scale
            b = np.zeros(sizes[i+1])
            self.layers.append((W, b))

            # Batch normalization (except for output layer)
            if self.config.use_batch_norm and i < len(sizes) - 2:
                self.batch_norms.append(BatchNormalization(sizes[i+1]))
            else:
                self.batch_norms.append(None)

            # Dropout (except for output layer)
            if i < len(sizes) - 2:
                self.dropouts.append(DropoutLayer(self.config.dropout_rate))
            else:
                self.dropouts.append(None)

        # Optional attention layer
        if self.config.use_residual and len(self.config.hidden_sizes) > 0:
            self.attention = AttentionLayer(
                self.config.hidden_sizes[0],
                num_heads=4,
                head_dim=self.config.hidden_sizes[0] // 4
            )

    def _init_optimizer(self):
        """Initialize Adam optimizer state"""
        for W, b in self.layers:
            self.m.append((np.zeros_like(W), np.zeros_like(b)))
            self.v.append((np.zeros_like(W), np.zeros_like(b)))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array (batch_size, input_size)
            training: Whether in training mode

        Returns:
            Output array
        """
        self.activations = [x]  # Store for backward pass

        for i, (W, b) in enumerate(self.layers):
            # Linear transformation
            x = x @ W + b

            # Batch normalization
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i].forward(x, training)

            # Activation (use output activation for last layer)
            if i == len(self.layers) - 1:
                x = apply_activation(x, self.config.output_activation)
            else:
                x = apply_activation(x, self.config.activation)

                # Apply attention after first hidden layer
                if i == 0 and self.attention is not None:
                    x = self.attention.forward(x)

                # Dropout
                if self.dropouts[i] is not None:
                    x = self.dropouts[i].forward(x, training)

            self.activations.append(x)

        return x

    def backward(self, d_out: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Backward pass to compute gradients.

        Args:
            d_out: Gradient of loss with respect to output

        Returns:
            List of (dW, db) gradients for each layer
        """
        gradients = []

        for i in reversed(range(len(self.layers))):
            W, b = self.layers[i]
            a_prev = self.activations[i]

            # Gradient through dropout
            if self.dropouts[i] is not None:
                d_out = self.dropouts[i].backward(d_out)

            # Gradient through activation
            if i == len(self.layers) - 1:
                # Softmax + cross-entropy gradient is just (pred - target)
                pass
            else:
                d_out = d_out * activation_derivative(
                    self.activations[i+1], self.config.activation
                )

            # Gradient through batch norm
            if self.batch_norms[i] is not None:
                d_out = self.batch_norms[i].backward(d_out)

            # Gradients for weights and biases
            dW = a_prev.T @ d_out / d_out.shape[0]
            db = np.mean(d_out, axis=0)

            # Add weight decay (L2 regularization)
            dW += self.config.weight_decay * W

            # Gradient clipping
            dW = np.clip(dW, -self.config.gradient_clip, self.config.gradient_clip)

            gradients.insert(0, (dW, db))

            # Propagate gradient to previous layer
            d_out = d_out @ W.T

        return gradients

    def update_weights(self, gradients: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Update weights using Adam optimizer.

        Args:
            gradients: List of (dW, db) gradients
        """
        self.t += 1
        lr = self.config.learning_rate
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        for i, (dW, db) in enumerate(gradients):
            # Update first moment
            self.m[i] = (
                beta1 * self.m[i][0] + (1 - beta1) * dW,
                beta1 * self.m[i][1] + (1 - beta1) * db
            )

            # Update second moment
            self.v[i] = (
                beta2 * self.v[i][0] + (1 - beta2) * dW**2,
                beta2 * self.v[i][1] + (1 - beta2) * db**2
            )

            # Bias correction
            m_hat_W = self.m[i][0] / (1 - beta1**self.t)
            m_hat_b = self.m[i][1] / (1 - beta1**self.t)
            v_hat_W = self.v[i][0] / (1 - beta2**self.t)
            v_hat_b = self.v[i][1] / (1 - beta2**self.t)

            # Update weights
            W, b = self.layers[i]
            W -= lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
            self.layers[i] = (W, b)

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: str = 'cross_entropy'
    ) -> float:
        """
        Single training step.

        Args:
            x: Input batch
            y: Target batch (one-hot for classification)
            loss_fn: Loss function type

        Returns:
            Loss value
        """
        # Forward pass
        pred = self.forward(x, training=True)

        # Compute loss
        if loss_fn == 'cross_entropy':
            loss = -np.mean(np.sum(y * np.log(pred + 1e-10), axis=1))
            d_out = pred - y
        elif loss_fn == 'mse':
            loss = np.mean((pred - y)**2)
            d_out = 2 * (pred - y) / y.shape[0]
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Backward pass
        gradients = self.backward(d_out)

        # Update weights
        self.update_weights(gradients)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions (inference mode)"""
        return self.forward(x, training=False)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.predict(x)

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        """Get predicted classes"""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def save(self, filepath: str):
        """Save model weights"""
        data = {
            'config': {
                'input_size': self.config.input_size,
                'hidden_sizes': self.config.hidden_sizes,
                'output_size': self.config.output_size,
                'activation': self.config.activation.name,
                'output_activation': self.config.output_activation.name,
                'dropout_rate': self.config.dropout_rate,
                'use_batch_norm': self.config.use_batch_norm,
                'use_residual': self.config.use_residual,
                'learning_rate': self.config.learning_rate,
            },
            'layers': [(W.tolist(), b.tolist()) for W, b in self.layers],
            't': self.t
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model weights"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.layers = [(np.array(W), np.array(b)) for W, b in data['layers']]
        self.t = data.get('t', 0)
        logger.info(f"Model loaded from {filepath}")


class RecurrentNetwork:
    """
    Recurrent Neural Network with LSTM/GRU cells.
    Designed for sequential market data processing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        cell_type: str = 'lstm',
        bidirectional: bool = False,
        dropout_rate: float = 0.2
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        # Build cells
        self.cells = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            if cell_type == 'lstm':
                self.cells.append(LSTMCell(in_size, hidden_size))
            else:
                self.cells.append(GRUCell(in_size, hidden_size))

        # Output layer
        out_hidden = hidden_size * 2 if bidirectional else hidden_size
        scale = np.sqrt(2.0 / out_hidden)
        self.W_out = np.random.randn(out_hidden, output_size) * scale
        self.b_out = np.zeros(output_size)

        # Dropout
        self.dropout = DropoutLayer(dropout_rate)

        logger.info(
            f"RecurrentNetwork initialized: {input_size} -> {hidden_size} x {num_layers} -> {output_size}"
        )

    def forward(
        self,
        x: np.ndarray,
        h_init: Optional[np.ndarray] = None,
        c_init: Optional[np.ndarray] = None,
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through recurrent network.

        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_init: Initial hidden state
            c_init: Initial cell state (for LSTM)
            training: Whether in training mode

        Returns:
            output, final_hidden_state
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states
        if h_init is None:
            h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        else:
            h = [h_init[i] for i in range(self.num_layers)]

        if self.cell_type == 'lstm':
            if c_init is None:
                c = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
            else:
                c = [c_init[i] for i in range(self.num_layers)]

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            for i, cell in enumerate(self.cells):
                if self.cell_type == 'lstm':
                    h[i], c[i] = cell.forward(x_t, h[i], c[i])
                else:
                    h[i] = cell.forward(x_t, h[i])

                x_t = self.dropout.forward(h[i], training)

            outputs.append(h[-1])

        # Stack outputs
        output = np.stack(outputs, axis=1)  # (batch, seq_len, hidden)

        # Final output (use last hidden state)
        final_hidden = h[-1]
        logits = final_hidden @ self.W_out + self.b_out

        return logits, output

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        logits, _ = self.forward(x, training=False)
        return apply_activation(logits, ActivationFunction.SOFTMAX)


class TransformerBlock:
    """
    Transformer-style block with self-attention and feed-forward layers.
    Adapted for trading feature processing.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int = 4,
        ff_hidden: int = 128,
        dropout_rate: float = 0.1
    ):
        self.input_size = input_size

        # Self-attention
        self.attention = AttentionLayer(input_size, num_heads, input_size // num_heads)

        # Layer normalization (simplified)
        self.ln1_gamma = np.ones(input_size)
        self.ln1_beta = np.zeros(input_size)
        self.ln2_gamma = np.ones(input_size)
        self.ln2_beta = np.zeros(input_size)

        # Feed-forward network
        scale = np.sqrt(2.0 / input_size)
        self.W_ff1 = np.random.randn(input_size, ff_hidden) * scale
        self.b_ff1 = np.zeros(ff_hidden)
        self.W_ff2 = np.random.randn(ff_hidden, input_size) * scale
        self.b_ff2 = np.zeros(input_size)

        # Dropout
        self.dropout = DropoutLayer(dropout_rate)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass"""
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        attn_out = self.dropout.forward(attn_out, training)
        x = self._layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)

        # Feed-forward with residual
        ff_out = x @ self.W_ff1 + self.b_ff1
        ff_out = apply_activation(ff_out, ActivationFunction.GELU)
        ff_out = ff_out @ self.W_ff2 + self.b_ff2
        ff_out = self.dropout.forward(ff_out, training)
        x = self._layer_norm(x + ff_out, self.ln2_gamma, self.ln2_beta)

        return x


class EnsembleNetwork:
    """
    Ensemble of multiple networks for robust predictions.
    Combines predictions using various strategies.
    """

    def __init__(
        self,
        networks: List[DeepNetwork],
        weights: Optional[List[float]] = None,
        aggregation: str = 'mean'
    ):
        self.networks = networks
        self.weights = weights or [1.0 / len(networks)] * len(networks)
        self.aggregation = aggregation

        logger.info(f"EnsembleNetwork initialized with {len(networks)} networks")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        for net, weight in zip(self.networks, self.weights):
            pred = net.predict(x)
            predictions.append(pred * weight)

        if self.aggregation == 'mean':
            return np.sum(predictions, axis=0)
        elif self.aggregation == 'max':
            return np.max(predictions, axis=0)
        elif self.aggregation == 'voting':
            # Hard voting
            votes = np.array([np.argmax(p, axis=1) for p in predictions])
            result = np.zeros_like(predictions[0])
            for i in range(result.shape[0]):
                counts = np.bincount(votes[:, i], minlength=result.shape[1])
                result[i] = counts / len(predictions)
            return result

        return np.mean(predictions, axis=0)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Train all networks"""
        losses = []
        for net in self.networks:
            loss = net.train_step(x, y)
            losses.append(loss)
        return losses


class AdaptiveNetwork:
    """
    Network that adapts to changing market conditions.
    Implements online learning with concept drift detection.
    """

    def __init__(
        self,
        base_config: NetworkConfig,
        drift_threshold: float = 0.1,
        window_size: int = 100
    ):
        self.base_config = base_config
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Main network
        self.network = DeepNetwork(base_config)

        # Performance tracking
        self.recent_losses = deque(maxlen=window_size)
        self.baseline_loss = None
        self.drift_detected = False

        # Shadow network for comparison
        self.shadow_network = None

        logger.info("AdaptiveNetwork initialized")

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        """
        Training step with drift detection.

        Returns:
            loss, drift_detected
        """
        loss = self.network.train_step(x, y)
        self.recent_losses.append(loss)

        # Check for drift
        if len(self.recent_losses) >= self.window_size:
            current_loss = np.mean(list(self.recent_losses)[-self.window_size//2:])

            if self.baseline_loss is None:
                self.baseline_loss = current_loss
            else:
                drift = (current_loss - self.baseline_loss) / (self.baseline_loss + 1e-10)

                if drift > self.drift_threshold:
                    self.drift_detected = True
                    self._handle_drift()
                    self.baseline_loss = current_loss
                else:
                    self.drift_detected = False

        return loss, self.drift_detected

    def _handle_drift(self):
        """Handle detected concept drift"""
        logger.warning("Concept drift detected - adapting network")

        # Option 1: Reset learning rate
        self.network.config.learning_rate *= 2

        # Option 2: Create shadow network with fresh weights
        self.shadow_network = DeepNetwork(self.base_config)

        # Option 3: Reinitialize last layers
        if len(self.network.layers) > 2:
            W, b = self.network.layers[-1]
            scale = np.sqrt(2.0 / W.shape[0])
            self.network.layers[-1] = (
                np.random.randn(*W.shape) * scale,
                np.zeros_like(b)
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction"""
        pred = self.network.predict(x)

        # If shadow network exists, blend predictions
        if self.shadow_network is not None:
            shadow_pred = self.shadow_network.predict(x)
            pred = 0.7 * pred + 0.3 * shadow_pred

        return pred
