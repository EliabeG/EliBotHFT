"""
Self-Optimization System for ML Trading

This module implements automatic optimization:
1. Hyperparameter Tuning (Bayesian, Grid, Random)
2. Neural Architecture Search
3. Feature Selection
4. Strategy Weight Optimization
5. Risk Parameter Optimization
6. Online Hyperparameter Adaptation
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = auto()
    RANDOM_SEARCH = auto()
    BAYESIAN = auto()
    GENETIC = auto()
    PARTICLE_SWARM = auto()
    EVOLUTIONARY = auto()


@dataclass
class HyperParameter:
    """Definition of a hyperparameter"""
    name: str
    param_type: str  # 'float', 'int', 'categorical'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Any = None
    log_scale: bool = False

    def sample(self) -> Any:
        """Sample a value"""
        if self.param_type == 'categorical':
            return np.random.choice(self.choices)
        elif self.param_type == 'int':
            if self.log_scale:
                log_val = np.random.uniform(np.log(self.min_value), np.log(self.max_value))
                return int(np.exp(log_val))
            return np.random.randint(self.min_value, self.max_value + 1)
        else:  # float
            if self.log_scale:
                log_val = np.random.uniform(np.log(self.min_value), np.log(self.max_value))
                return np.exp(log_val)
            return np.random.uniform(self.min_value, self.max_value)


@dataclass
class OptimizationResult:
    """Result from optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    n_iterations: int
    total_time: float
    convergence_history: List[float]


class BaseOptimizer(ABC):
    """Base class for optimizers"""

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int
    ) -> OptimizationResult:
        pass


class GridSearchOptimizer(BaseOptimizer):
    """Grid search over parameter space"""

    def __init__(self, n_points_per_param: int = 5):
        self.n_points = n_points_per_param

    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Perform grid search"""
        start_time = time.time()

        # Generate grid
        param_grids = {}
        for param in param_space:
            if param.param_type == 'categorical':
                param_grids[param.name] = param.choices
            elif param.param_type == 'int':
                param_grids[param.name] = np.linspace(
                    param.min_value, param.max_value, self.n_points
                ).astype(int).tolist()
            else:
                if param.log_scale:
                    param_grids[param.name] = np.exp(np.linspace(
                        np.log(param.min_value), np.log(param.max_value), self.n_points
                    )).tolist()
                else:
                    param_grids[param.name] = np.linspace(
                        param.min_value, param.max_value, self.n_points
                    ).tolist()

        # Generate all combinations
        from itertools import product
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())

        all_results = []
        best_params = None
        best_score = float('-inf')
        convergence = []

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            try:
                score = objective_fn(params)
                all_results.append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                convergence.append(best_score)
            except Exception as e:
                logger.warning(f"Error evaluating params {params}: {e}")

            if len(all_results) >= n_iterations:
                break

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(all_results),
            total_time=time.time() - start_time,
            convergence_history=convergence
        )


class RandomSearchOptimizer(BaseOptimizer):
    """Random search over parameter space"""

    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Perform random search"""
        start_time = time.time()

        all_results = []
        best_params = None
        best_score = float('-inf')
        convergence = []

        for i in range(n_iterations):
            # Sample random parameters
            params = {p.name: p.sample() for p in param_space}

            try:
                score = objective_fn(params)
                all_results.append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                convergence.append(best_score)
            except Exception as e:
                logger.warning(f"Error evaluating params: {e}")

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(all_results),
            total_time=time.time() - start_time,
            convergence_history=convergence
        )


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Gaussian Process surrogate.
    """

    def __init__(
        self,
        n_initial_points: int = 10,
        exploration_weight: float = 2.0
    ):
        self.n_initial = n_initial_points
        self.exploration_weight = exploration_weight

    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Perform Bayesian optimization"""
        start_time = time.time()

        # Store observations
        X_observed = []
        y_observed = []
        all_results = []
        best_params = None
        best_score = float('-inf')
        convergence = []

        param_names = [p.name for p in param_space]

        # Helper to convert params to vector
        def params_to_vector(params: Dict) -> np.ndarray:
            vec = []
            for p in param_space:
                val = params[p.name]
                if p.param_type == 'categorical':
                    val = p.choices.index(val)
                elif p.log_scale:
                    val = np.log(val)
                # Normalize to [0, 1]
                if p.min_value is not None and p.max_value is not None:
                    if p.log_scale:
                        val = (val - np.log(p.min_value)) / (np.log(p.max_value) - np.log(p.min_value))
                    else:
                        val = (val - p.min_value) / (p.max_value - p.min_value)
                vec.append(val)
            return np.array(vec)

        def vector_to_params(vec: np.ndarray) -> Dict:
            params = {}
            for i, p in enumerate(param_space):
                val = vec[i]
                # Denormalize
                if p.min_value is not None and p.max_value is not None:
                    if p.log_scale:
                        val = val * (np.log(p.max_value) - np.log(p.min_value)) + np.log(p.min_value)
                        val = np.exp(val)
                    else:
                        val = val * (p.max_value - p.min_value) + p.min_value

                if p.param_type == 'int':
                    val = int(round(val))
                elif p.param_type == 'categorical':
                    val = p.choices[int(round(val)) % len(p.choices)]

                params[p.name] = val
            return params

        # Initial random sampling
        for _ in range(self.n_initial):
            params = {p.name: p.sample() for p in param_space}
            try:
                score = objective_fn(params)
                X_observed.append(params_to_vector(params))
                y_observed.append(score)
                all_results.append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                convergence.append(best_score)
            except Exception as e:
                logger.warning(f"Error in initial sampling: {e}")

        # Bayesian optimization iterations
        for i in range(n_iterations - self.n_initial):
            if len(X_observed) < 2:
                # Fall back to random sampling
                params = {p.name: p.sample() for p in param_space}
            else:
                # Fit GP surrogate and find next point
                X = np.array(X_observed)
                y = np.array(y_observed)

                # Find next point using acquisition function
                next_vec = self._acquisition_search(X, y, len(param_space))
                params = vector_to_params(next_vec)

            try:
                score = objective_fn(params)
                X_observed.append(params_to_vector(params))
                y_observed.append(score)
                all_results.append((params, score))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                convergence.append(best_score)
            except Exception as e:
                logger.warning(f"Error in BO iteration: {e}")

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(all_results),
            total_time=time.time() - start_time,
            convergence_history=convergence
        )

    def _acquisition_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_dims: int,
        n_candidates: int = 1000
    ) -> np.ndarray:
        """Find next point using UCB acquisition"""
        # Generate random candidates
        candidates = np.random.random((n_candidates, n_dims))

        # Compute GP predictions (simplified RBF kernel)
        best_acquisition = float('-inf')
        best_candidate = candidates[0]

        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-6

        for candidate in candidates:
            # Compute distances to observed points
            distances = np.linalg.norm(X - candidate, axis=1)

            # RBF kernel
            kernel_values = np.exp(-distances ** 2 / 2)

            # Predict mean and variance (simplified)
            weights = kernel_values / (np.sum(kernel_values) + 1e-6)
            pred_mean = np.sum(weights * y)
            pred_var = np.sum(weights * (y - pred_mean) ** 2) + 0.1

            # UCB acquisition
            acquisition = pred_mean + self.exploration_weight * np.sqrt(pred_var)

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_candidate = candidate

        return best_candidate


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm for optimization.
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Perform genetic optimization"""
        start_time = time.time()

        # Initialize population
        population = [
            {p.name: p.sample() for p in param_space}
            for _ in range(self.population_size)
        ]

        all_results = []
        best_params = None
        best_score = float('-inf')
        convergence = []

        n_elite = max(1, int(self.population_size * self.elite_fraction))

        for generation in range(n_iterations // self.population_size + 1):
            # Evaluate fitness
            fitness = []
            for individual in population:
                try:
                    score = objective_fn(individual)
                    fitness.append(score)
                    all_results.append((individual.copy(), score))

                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()

                except Exception as e:
                    fitness.append(float('-inf'))
                    logger.warning(f"Error evaluating individual: {e}")

            convergence.append(best_score)

            if len(all_results) >= n_iterations:
                break

            # Selection - keep elite
            sorted_indices = np.argsort(fitness)[::-1]
            elite = [population[i] for i in sorted_indices[:n_elite]]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, param_space)
                else:
                    child = parent1.copy()

                # Mutation
                child = self._mutate(child, param_space)

                new_population.append(child)

            population = new_population

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(all_results),
            total_time=time.time() - start_time,
            convergence_history=convergence
        )

    def _tournament_select(
        self,
        population: List[Dict],
        fitness: List[float],
        tournament_size: int = 3
    ) -> Dict:
        """Tournament selection"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness[i] for i in indices])]
        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: Dict,
        parent2: Dict,
        param_space: List[HyperParameter]
    ) -> Dict:
        """Uniform crossover"""
        child = {}
        for p in param_space:
            if np.random.random() < 0.5:
                child[p.name] = parent1[p.name]
            else:
                child[p.name] = parent2[p.name]
        return child

    def _mutate(self, individual: Dict, param_space: List[HyperParameter]) -> Dict:
        """Mutate individual"""
        for p in param_space:
            if np.random.random() < self.mutation_rate:
                individual[p.name] = p.sample()
        return individual


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization.
    """

    def __init__(
        self,
        n_particles: int = 30,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5
    ):
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optimize(
        self,
        objective_fn: Callable,
        param_space: List[HyperParameter],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Perform PSO"""
        start_time = time.time()
        n_dims = len(param_space)

        # Initialize particles
        positions = np.random.random((self.n_particles, n_dims))
        velocities = np.random.random((self.n_particles, n_dims)) * 0.1

        # Personal and global best
        personal_best_pos = positions.copy()
        personal_best_scores = np.full(self.n_particles, float('-inf'))
        global_best_pos = positions[0].copy()
        global_best_score = float('-inf')

        all_results = []
        convergence = []

        def vector_to_params(vec: np.ndarray) -> Dict:
            params = {}
            for i, p in enumerate(param_space):
                val = vec[i]
                if p.min_value is not None and p.max_value is not None:
                    if p.log_scale:
                        val = np.exp(val * (np.log(p.max_value) - np.log(p.min_value)) + np.log(p.min_value))
                    else:
                        val = val * (p.max_value - p.min_value) + p.min_value

                if p.param_type == 'int':
                    val = int(round(val))
                elif p.param_type == 'categorical':
                    val = p.choices[int(val * len(p.choices)) % len(p.choices)]

                params[p.name] = val
            return params

        for iteration in range(n_iterations):
            # Evaluate particles
            for i in range(self.n_particles):
                params = vector_to_params(positions[i])
                try:
                    score = objective_fn(params)
                    all_results.append((params, score))

                    # Update personal best
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_pos[i] = positions[i].copy()

                    # Update global best
                    if score > global_best_score:
                        global_best_score = score
                        global_best_pos = positions[i].copy()

                except Exception as e:
                    logger.warning(f"Error evaluating particle: {e}")

            convergence.append(global_best_score)

            # Update velocities and positions
            r1 = np.random.random((self.n_particles, n_dims))
            r2 = np.random.random((self.n_particles, n_dims))

            cognitive_component = self.cognitive * r1 * (personal_best_pos - positions)
            social_component = self.social * r2 * (global_best_pos - positions)

            velocities = self.inertia * velocities + cognitive_component + social_component
            velocities = np.clip(velocities, -0.5, 0.5)

            positions = positions + velocities
            positions = np.clip(positions, 0, 1)

        best_params = vector_to_params(global_best_pos)

        return OptimizationResult(
            best_params=best_params,
            best_score=global_best_score,
            all_results=all_results,
            n_iterations=len(all_results),
            total_time=time.time() - start_time,
            convergence_history=convergence
        )


class FeatureSelector:
    """
    Automatic feature selection.
    """

    def __init__(
        self,
        method: str = 'importance',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ):
        self.method = method
        self.n_features = n_features
        self.threshold = threshold

        self.selected_features: List[int] = []
        self.feature_importances: Dict[int, float] = {}

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any = None
    ) -> np.ndarray:
        """
        Select features.

        Args:
            X: Feature matrix
            y: Target
            model: Optional model with feature_importances_

        Returns:
            Selected feature indices
        """
        n_features = X.shape[1]

        if self.method == 'importance' and model is not None:
            # Use model-based importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # Calculate permutation importance
                importances = self._permutation_importance(X, y, model)

            self.feature_importances = {i: imp for i, imp in enumerate(importances)}

            # Select top features
            if self.n_features is not None:
                sorted_idx = np.argsort(importances)[::-1]
                self.selected_features = sorted_idx[:self.n_features].tolist()
            else:
                self.selected_features = [i for i, imp in enumerate(importances) if imp > self.threshold]

        elif self.method == 'correlation':
            # Remove highly correlated features
            correlation_matrix = np.corrcoef(X.T)
            selected = list(range(n_features))

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if j in selected and abs(correlation_matrix[i, j]) > 0.95:
                        selected.remove(j)

            self.selected_features = selected

        elif self.method == 'variance':
            # Remove low variance features
            variances = np.var(X, axis=0)
            threshold = np.percentile(variances, 10)
            self.selected_features = [i for i, v in enumerate(variances) if v > threshold]

        elif self.method == 'mutual_info':
            # Mutual information selection
            mi_scores = self._mutual_information(X, y)
            self.feature_importances = {i: mi for i, mi in enumerate(mi_scores)}

            if self.n_features is not None:
                sorted_idx = np.argsort(mi_scores)[::-1]
                self.selected_features = sorted_idx[:self.n_features].tolist()
            else:
                self.selected_features = [i for i, mi in enumerate(mi_scores) if mi > self.threshold]

        else:
            # Default: select all
            self.selected_features = list(range(n_features))

        return np.array(self.selected_features)

    def _permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        n_repeats: int = 5
    ) -> np.ndarray:
        """Calculate permutation importance"""
        # Base score
        base_pred = model.predict(X)
        base_score = np.mean(np.argmax(base_pred, axis=1) == np.argmax(y, axis=1))

        importances = []
        for i in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                pred = model.predict(X_permuted)
                score = np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
                scores.append(base_score - score)
            importances.append(np.mean(scores))

        return np.array(importances)

    def _mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate mutual information (simplified)"""
        n_features = X.shape[1]
        mi_scores = []

        y_labels = np.argmax(y, axis=1) if y.ndim > 1 else y

        for i in range(n_features):
            # Discretize feature
            feature = X[:, i]
            bins = np.percentile(feature, [25, 50, 75])
            feature_discrete = np.digitize(feature, bins)

            # Calculate MI
            mi = self._calc_mi(feature_discrete, y_labels)
            mi_scores.append(mi)

        return np.array(mi_scores)

    def _calc_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information"""
        # Joint and marginal probabilities
        xy_counts = {}
        x_counts = {}
        y_counts = {}
        n = len(x)

        for xi, yi in zip(x, y):
            xy_counts[(xi, yi)] = xy_counts.get((xi, yi), 0) + 1
            x_counts[xi] = x_counts.get(xi, 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1

        mi = 0.0
        for (xi, yi), count in xy_counts.items():
            p_xy = count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

        return mi

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to selected features"""
        return X[:, self.selected_features]


class NeuralArchitectureSearch:
    """
    Simple Neural Architecture Search.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        min_layers: int = 1,
        max_layers: int = 5,
        min_units: int = 16,
        max_units: int = 256
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_units = min_units
        self.max_units = max_units

    def search(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Search for best architecture.

        Args:
            train_fn: Function to train model given architecture
            eval_fn: Function to evaluate model
            n_trials: Number of architectures to try

        Returns:
            Best architecture configuration
        """
        best_architecture = None
        best_score = float('-inf')
        all_architectures = []

        for trial in range(n_trials):
            # Sample architecture
            n_layers = np.random.randint(self.min_layers, self.max_layers + 1)
            hidden_sizes = [
                np.random.randint(self.min_units, self.max_units + 1)
                for _ in range(n_layers)
            ]

            # Optional: decreasing layer sizes
            if np.random.random() < 0.5:
                hidden_sizes = sorted(hidden_sizes, reverse=True)

            architecture = {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_sizes': hidden_sizes,
                'dropout_rate': np.random.uniform(0.0, 0.5),
                'use_batch_norm': np.random.random() < 0.5,
                'activation': np.random.choice(['relu', 'leaky_relu', 'elu', 'gelu'])
            }

            try:
                # Train and evaluate
                model = train_fn(architecture)
                score = eval_fn(model)

                all_architectures.append((architecture, score))

                if score > best_score:
                    best_score = score
                    best_architecture = architecture.copy()

                logger.info(f"NAS Trial {trial+1}/{n_trials}: score={score:.4f}")
            except Exception as e:
                logger.warning(f"NAS trial failed: {e}")

        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'all_architectures': all_architectures
        }


class OnlineHyperparameterTuner:
    """
    Online hyperparameter tuning during trading.
    Adapts parameters based on recent performance.
    """

    def __init__(
        self,
        param_space: List[HyperParameter],
        window_size: int = 100,
        adaptation_rate: float = 0.1
    ):
        self.param_space = param_space
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # Current parameters
        self.current_params = {p.name: p.default or p.sample() for p in param_space}

        # Performance tracking
        self.performance_history = deque(maxlen=window_size)
        self.param_history = deque(maxlen=window_size)

        # UCB-like bandit for parameter selection
        self.param_arms: Dict[str, Dict[str, List[float]]] = {
            p.name: {} for p in param_space
        }

        logger.info("OnlineHyperparameterTuner initialized")

    def update(self, performance: float):
        """
        Update with new performance observation.

        Args:
            performance: Performance metric (higher is better)
        """
        self.performance_history.append(performance)
        self.param_history.append(self.current_params.copy())

        # Update arm statistics
        for param_name, value in self.current_params.items():
            key = str(value)
            if key not in self.param_arms[param_name]:
                self.param_arms[param_name][key] = []
            self.param_arms[param_name][key].append(performance)

        # Check if we should adapt
        if len(self.performance_history) >= self.window_size:
            recent_perf = np.mean(list(self.performance_history)[-self.window_size//2:])
            older_perf = np.mean(list(self.performance_history)[:self.window_size//2])

            # If performance degraded, try new parameters
            if recent_perf < older_perf * 0.9:
                self._adapt_parameters()

    def _adapt_parameters(self):
        """Adapt parameters using UCB"""
        for param in self.param_space:
            param_name = param.name
            arms = self.param_arms[param_name]

            if len(arms) < 2:
                # Explore new value
                self.current_params[param_name] = param.sample()
            else:
                # UCB selection
                total_pulls = sum(len(rewards) for rewards in arms.values())
                best_ucb = float('-inf')
                best_value = self.current_params[param_name]

                for value_str, rewards in arms.items():
                    n_pulls = len(rewards)
                    mean_reward = np.mean(rewards)
                    ucb = mean_reward + np.sqrt(2 * np.log(total_pulls) / n_pulls)

                    if ucb > best_ucb:
                        best_ucb = ucb
                        # Convert back from string
                        if param.param_type == 'float':
                            best_value = float(value_str)
                        elif param.param_type == 'int':
                            best_value = int(value_str)
                        else:
                            best_value = value_str

                # Exploration: sometimes try random
                if np.random.random() < self.adaptation_rate:
                    best_value = param.sample()

                self.current_params[param_name] = best_value

        logger.info(f"Parameters adapted: {self.current_params}")

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.current_params.copy()


class SelfOptimizingSystem:
    """
    Complete self-optimizing ML system.
    """

    def __init__(
        self,
        model: Any,
        param_space: List[HyperParameter],
        optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        retrain_frequency: int = 1000,
        optimization_budget: int = 50
    ):
        self.model = model
        self.param_space = param_space
        self.optimization_method = optimization_method
        self.retrain_frequency = retrain_frequency
        self.optimization_budget = optimization_budget

        # Initialize optimizer
        if optimization_method == OptimizationMethod.GRID_SEARCH:
            self.optimizer = GridSearchOptimizer()
        elif optimization_method == OptimizationMethod.RANDOM_SEARCH:
            self.optimizer = RandomSearchOptimizer()
        elif optimization_method == OptimizationMethod.BAYESIAN:
            self.optimizer = BayesianOptimizer()
        elif optimization_method == OptimizationMethod.GENETIC:
            self.optimizer = GeneticOptimizer()
        elif optimization_method == OptimizationMethod.PARTICLE_SWARM:
            self.optimizer = ParticleSwarmOptimizer()
        else:
            self.optimizer = RandomSearchOptimizer()

        # Feature selector
        self.feature_selector = FeatureSelector()

        # Online tuner
        self.online_tuner = OnlineHyperparameterTuner(param_space)

        # Tracking
        self.n_samples = 0
        self.optimization_history: List[OptimizationResult] = []
        self.best_params: Dict[str, Any] = {}

        logger.info(f"SelfOptimizingSystem initialized with {optimization_method.name}")

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> OptimizationResult:
        """
        Run full optimization.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data

        Returns:
            Optimization result
        """
        # Feature selection first
        selected_features = self.feature_selector.select(X_train, y_train, self.model)
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]

        # Define objective function
        def objective(params: Dict) -> float:
            try:
                # Apply parameters to model
                self._apply_params(params)

                # Train model
                self.model.fit(X_train_selected, y_train)

                # Evaluate
                pred = self.model.predict(X_val_selected)
                accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(y_val, axis=1))

                return accuracy
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 0.0

        # Run optimization
        result = self.optimizer.optimize(objective, self.param_space, self.optimization_budget)

        # Apply best params
        if result.best_params:
            self._apply_params(result.best_params)
            self.best_params = result.best_params.copy()

        self.optimization_history.append(result)

        logger.info(f"Optimization complete: best_score={result.best_score:.4f}")

        return result

    def _apply_params(self, params: Dict):
        """Apply parameters to model"""
        for name, value in params.items():
            if hasattr(self.model, name):
                setattr(self.model, name, value)
            elif hasattr(self.model, 'config') and hasattr(self.model.config, name):
                setattr(self.model.config, name, value)

    def update_online(self, performance: float):
        """Online parameter update"""
        self.n_samples += 1
        self.online_tuner.update(performance)

        # Periodic full optimization
        if self.n_samples % self.retrain_frequency == 0:
            logger.info("Triggering periodic re-optimization")
            # Would need stored data to re-optimize

    def get_current_params(self) -> Dict[str, Any]:
        """Get current best parameters"""
        return self.online_tuner.get_params()
