"""
Model Store for ML Model Persistence and Versioning

This module handles:
- Saving and loading ML models
- Model versioning and history
- Model performance tracking
- A/B testing support
- Model rollback capabilities
"""

import logging
import os
import json
import hashlib
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model version"""
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for a saved model"""

    # Identification
    model_id: str
    model_type: str
    version: str
    name: str

    # Performance metrics
    accuracy: float
    loss: float
    validation_accuracy: float
    validation_loss: float

    # Training info
    samples_trained: int
    epochs_trained: int
    training_time_ms: int
    training_date: datetime

    # Configuration
    config: Dict[str, Any]
    feature_size: int
    output_size: int

    # Status
    status: ModelStatus
    is_production: bool

    # File paths
    weights_file: str
    metadata_file: str

    # Checksums for integrity
    weights_checksum: str

    # Additional info
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None

    # Performance tracking
    inference_count: int = 0
    avg_inference_time_ms: float = 0.0
    prediction_accuracy_live: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'version': self.version,
            'name': self.name,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'validation_accuracy': self.validation_accuracy,
            'validation_loss': self.validation_loss,
            'samples_trained': self.samples_trained,
            'epochs_trained': self.epochs_trained,
            'training_time_ms': self.training_time_ms,
            'training_date': self.training_date.isoformat(),
            'config': self.config,
            'feature_size': self.feature_size,
            'output_size': self.output_size,
            'status': self.status.value,
            'is_production': self.is_production,
            'weights_file': self.weights_file,
            'metadata_file': self.metadata_file,
            'weights_checksum': self.weights_checksum,
            'description': self.description,
            'tags': self.tags,
            'parent_version': self.parent_version,
            'inference_count': self.inference_count,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'prediction_accuracy_live': self.prediction_accuracy_live,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(
            model_id=data['model_id'],
            model_type=data['model_type'],
            version=data['version'],
            name=data['name'],
            accuracy=data['accuracy'],
            loss=data['loss'],
            validation_accuracy=data['validation_accuracy'],
            validation_loss=data['validation_loss'],
            samples_trained=data['samples_trained'],
            epochs_trained=data['epochs_trained'],
            training_time_ms=data['training_time_ms'],
            training_date=datetime.fromisoformat(data['training_date']),
            config=data['config'],
            feature_size=data['feature_size'],
            output_size=data['output_size'],
            status=ModelStatus(data['status']),
            is_production=data['is_production'],
            weights_file=data['weights_file'],
            metadata_file=data['metadata_file'],
            weights_checksum=data['weights_checksum'],
            description=data.get('description', ''),
            tags=data.get('tags', []),
            parent_version=data.get('parent_version'),
            inference_count=data.get('inference_count', 0),
            avg_inference_time_ms=data.get('avg_inference_time_ms', 0.0),
            prediction_accuracy_live=data.get('prediction_accuracy_live', 0.0),
        )


@dataclass
class ModelComparison:
    """Comparison between two model versions"""
    model_a: ModelMetadata
    model_b: ModelMetadata
    accuracy_diff: float
    loss_diff: float
    val_accuracy_diff: float
    recommendation: str  # "use_a", "use_b", "no_significant_difference"


class ModelStore:
    """
    Central store for ML model persistence, versioning, and management.

    Features:
    - Automatic versioning
    - Model checkpointing
    - Performance tracking
    - A/B testing support
    - Rollback capability
    """

    def __init__(
        self,
        base_path: str = "data/models",
        max_versions_per_model: int = 10,
        auto_cleanup: bool = True
    ):
        """
        Initialize the model store.

        Args:
            base_path: Base directory for storing models
            max_versions_per_model: Maximum versions to keep per model type
            auto_cleanup: Automatically remove old versions
        """
        self.base_path = Path(base_path)
        self.max_versions = max_versions_per_model
        self.auto_cleanup = auto_cleanup

        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "production").mkdir(exist_ok=True)
        (self.base_path / "archive").mkdir(exist_ok=True)
        (self.base_path / "checkpoints").mkdir(exist_ok=True)

        # Model registry
        self.registry_file = self.base_path / "registry.json"
        self.registry: Dict[str, List[ModelMetadata]] = {}

        # Load existing registry
        self._load_registry()

        # Production model cache
        self.production_models: Dict[str, ModelMetadata] = {}

        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"ModelStore initialized at {self.base_path}")

    def _load_registry(self):
        """Load model registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)

                for model_type, versions in data.items():
                    self.registry[model_type] = [
                        ModelMetadata.from_dict(v) for v in versions
                    ]

                logger.info(f"Loaded {len(self.registry)} model types from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.registry = {}
        else:
            self.registry = {}

    def _save_registry(self):
        """Save model registry to disk"""
        data = {
            model_type: [m.to_dict() for m in versions]
            for model_type, versions in self.registry.items()
        }

        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_version(self, model_type: str) -> str:
        """Generate a new version string"""
        existing = self.registry.get(model_type, [])

        if not existing:
            return "1.0.0"

        # Get latest version
        latest = existing[-1].version
        parts = latest.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment patch version
        return f"{major}.{minor}.{patch + 1}"

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def save_model(
        self,
        model_type: str,
        weights_data: Dict[str, Any],
        accuracy: float,
        loss: float,
        validation_accuracy: float,
        validation_loss: float,
        samples_trained: int,
        epochs_trained: int,
        training_time_ms: int,
        config: Dict[str, Any],
        feature_size: int,
        output_size: int,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        set_as_production: bool = False
    ) -> ModelMetadata:
        """
        Save a trained model.

        Args:
            model_type: Type of model (e.g., "error_classifier")
            weights_data: Model weights and parameters
            accuracy: Training accuracy
            loss: Training loss
            validation_accuracy: Validation accuracy
            validation_loss: Validation loss
            samples_trained: Number of samples trained on
            epochs_trained: Number of epochs trained
            training_time_ms: Training time in milliseconds
            config: Model configuration
            feature_size: Input feature size
            output_size: Output size
            name: Optional model name
            description: Model description
            tags: Optional tags for categorization
            set_as_production: Whether to set as production model

        Returns:
            ModelMetadata for the saved model
        """
        version = self._generate_version(model_type)
        model_id = f"{model_type}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        name = name or f"{model_type} v{version}"

        # Create model directory
        model_dir = self.base_path / model_type / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights_file = model_dir / "weights.json"
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f)

        checksum = self._calculate_checksum(str(weights_file))

        # Get parent version
        existing = self.registry.get(model_type, [])
        parent_version = existing[-1].version if existing else None

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            name=name,
            accuracy=accuracy,
            loss=loss,
            validation_accuracy=validation_accuracy,
            validation_loss=validation_loss,
            samples_trained=samples_trained,
            epochs_trained=epochs_trained,
            training_time_ms=training_time_ms,
            training_date=datetime.now(),
            config=config,
            feature_size=feature_size,
            output_size=output_size,
            status=ModelStatus.ACTIVE,
            is_production=set_as_production,
            weights_file=str(weights_file),
            metadata_file=str(model_dir / "metadata.json"),
            weights_checksum=checksum,
            description=description,
            tags=tags or [],
            parent_version=parent_version
        )

        # Save metadata
        with open(metadata.metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update registry
        if model_type not in self.registry:
            self.registry[model_type] = []
        self.registry[model_type].append(metadata)

        # Set as production if requested
        if set_as_production:
            self._set_production_model(metadata)

        # Save registry
        self._save_registry()

        # Cleanup old versions
        if self.auto_cleanup:
            self._cleanup_old_versions(model_type)

        logger.info(f"Saved model {model_id} (version {version})")

        return metadata

    def load_model(
        self,
        model_type: str,
        version: Optional[str] = None,
        load_production: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ModelMetadata]]:
        """
        Load a model.

        Args:
            model_type: Type of model to load
            version: Specific version to load (None for latest/production)
            load_production: If True and version is None, load production model

        Returns:
            Tuple of (weights_data, metadata) or (None, None) if not found
        """
        if model_type not in self.registry:
            logger.warning(f"No models found for type {model_type}")
            return None, None

        versions = self.registry[model_type]

        if version:
            # Load specific version
            metadata = next((m for m in versions if m.version == version), None)
        elif load_production:
            # Load production model
            metadata = next((m for m in versions if m.is_production), None)
            if not metadata:
                # Fall back to latest
                metadata = versions[-1] if versions else None
        else:
            # Load latest
            metadata = versions[-1] if versions else None

        if not metadata:
            logger.warning(f"No model found for {model_type} version {version}")
            return None, None

        # Load weights
        try:
            with open(metadata.weights_file, 'r') as f:
                weights_data = json.load(f)

            # Verify checksum
            current_checksum = self._calculate_checksum(metadata.weights_file)
            if current_checksum != metadata.weights_checksum:
                logger.warning(f"Checksum mismatch for {metadata.model_id}")

            logger.info(f"Loaded model {metadata.model_id}")
            return weights_data, metadata

        except Exception as e:
            logger.error(f"Failed to load model {metadata.model_id}: {e}")
            return None, None

    def _set_production_model(self, metadata: ModelMetadata):
        """Set a model as the production model"""
        # Unset previous production model
        if metadata.model_type in self.registry:
            for m in self.registry[metadata.model_type]:
                if m.is_production:
                    m.is_production = False
                    # Update metadata file
                    with open(m.metadata_file, 'w') as f:
                        json.dump(m.to_dict(), f, indent=2)

        # Set new production model
        metadata.is_production = True
        self.production_models[metadata.model_type] = metadata

        # Copy to production folder
        prod_path = self.base_path / "production" / metadata.model_type
        prod_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(metadata.weights_file, prod_path / "weights.json")
        shutil.copy(metadata.metadata_file, prod_path / "metadata.json")

        logger.info(f"Set {metadata.model_id} as production model")

    def promote_to_production(self, model_type: str, version: str) -> bool:
        """
        Promote a model version to production.

        Args:
            model_type: Model type
            version: Version to promote

        Returns:
            True if successful
        """
        if model_type not in self.registry:
            return False

        metadata = next(
            (m for m in self.registry[model_type] if m.version == version),
            None
        )

        if not metadata:
            return False

        self._set_production_model(metadata)
        self._save_registry()

        return True

    def rollback(self, model_type: str, to_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous model version.

        Args:
            model_type: Model type
            to_version: Version to rollback to (None for previous)

        Returns:
            True if successful
        """
        if model_type not in self.registry:
            return False

        versions = self.registry[model_type]

        if to_version:
            target = next((m for m in versions if m.version == to_version), None)
        else:
            # Get previous production model
            current_prod = next((m for m in versions if m.is_production), None)
            if current_prod and current_prod.parent_version:
                target = next(
                    (m for m in versions if m.version == current_prod.parent_version),
                    None
                )
            else:
                # Get second-to-last
                target = versions[-2] if len(versions) >= 2 else None

        if not target:
            logger.warning(f"No version to rollback to for {model_type}")
            return False

        self._set_production_model(target)
        self._save_registry()

        logger.info(f"Rolled back {model_type} to version {target.version}")
        return True

    def _cleanup_old_versions(self, model_type: str):
        """Remove old model versions beyond the limit"""
        if model_type not in self.registry:
            return

        versions = self.registry[model_type]

        if len(versions) <= self.max_versions:
            return

        # Keep production model and most recent versions
        to_remove = []
        for m in versions[:-self.max_versions]:
            if not m.is_production:
                to_remove.append(m)

        for m in to_remove:
            # Archive instead of delete
            archive_dir = self.base_path / "archive" / m.model_type / m.version
            source_dir = Path(m.weights_file).parent

            if source_dir.exists():
                shutil.move(str(source_dir), str(archive_dir))

            versions.remove(m)
            logger.info(f"Archived old model {m.model_id}")

        self._save_registry()

    def create_checkpoint(
        self,
        model_type: str,
        weights_data: Dict[str, Any],
        checkpoint_name: str
    ) -> str:
        """
        Create a checkpoint for a model during training.

        Args:
            model_type: Model type
            weights_data: Current weights
            checkpoint_name: Name for the checkpoint

        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = self.base_path / "checkpoints" / model_type
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}_{timestamp}.json"

        with open(checkpoint_file, 'w') as f:
            json.dump({
                'weights': weights_data,
                'timestamp': timestamp,
                'name': checkpoint_name
            }, f)

        logger.debug(f"Created checkpoint: {checkpoint_file}")
        return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint"""
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return data['weights']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def compare_models(
        self,
        model_type: str,
        version_a: str,
        version_b: str
    ) -> Optional[ModelComparison]:
        """
        Compare two model versions.

        Args:
            model_type: Model type
            version_a: First version
            version_b: Second version

        Returns:
            ModelComparison or None if models not found
        """
        if model_type not in self.registry:
            return None

        versions = self.registry[model_type]
        model_a = next((m for m in versions if m.version == version_a), None)
        model_b = next((m for m in versions if m.version == version_b), None)

        if not model_a or not model_b:
            return None

        accuracy_diff = model_b.accuracy - model_a.accuracy
        loss_diff = model_a.loss - model_b.loss  # Lower loss is better
        val_accuracy_diff = model_b.validation_accuracy - model_a.validation_accuracy

        # Determine recommendation
        score_a = model_a.validation_accuracy - model_a.validation_loss * 0.1
        score_b = model_b.validation_accuracy - model_b.validation_loss * 0.1

        if abs(score_a - score_b) < 0.01:
            recommendation = "no_significant_difference"
        elif score_b > score_a:
            recommendation = "use_b"
        else:
            recommendation = "use_a"

        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            accuracy_diff=accuracy_diff,
            loss_diff=loss_diff,
            val_accuracy_diff=val_accuracy_diff,
            recommendation=recommendation
        )

    def record_inference(
        self,
        model_type: str,
        inference_time_ms: float,
        prediction_correct: Optional[bool] = None
    ):
        """
        Record inference metrics for the production model.

        Args:
            model_type: Model type
            inference_time_ms: Time taken for inference
            prediction_correct: Whether prediction was correct (if known)
        """
        if model_type not in self.registry:
            return

        # Find production model
        prod_model = next(
            (m for m in self.registry[model_type] if m.is_production),
            None
        )

        if not prod_model:
            return

        # Update running averages
        n = prod_model.inference_count + 1
        old_avg = prod_model.avg_inference_time_ms

        prod_model.inference_count = n
        prod_model.avg_inference_time_ms = old_avg + (inference_time_ms - old_avg) / n

        if prediction_correct is not None:
            old_acc = prod_model.prediction_accuracy_live
            if old_acc == 0:
                prod_model.prediction_accuracy_live = 1.0 if prediction_correct else 0.0
            else:
                prod_model.prediction_accuracy_live = old_acc + (
                    (1.0 if prediction_correct else 0.0) - old_acc
                ) / n

    def get_model_history(self, model_type: str) -> List[ModelMetadata]:
        """Get version history for a model type"""
        return self.registry.get(model_type, [])

    def get_production_models(self) -> Dict[str, ModelMetadata]:
        """Get all production models"""
        result = {}
        for model_type, versions in self.registry.items():
            prod = next((m for m in versions if m.is_production), None)
            if prod:
                result[model_type] = prod
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        stats = {
            'total_model_types': len(self.registry),
            'total_versions': sum(len(v) for v in self.registry.values()),
            'production_models': len(self.get_production_models()),
            'model_types': {}
        }

        for model_type, versions in self.registry.items():
            prod = next((m for m in versions if m.is_production), None)
            stats['model_types'][model_type] = {
                'versions': len(versions),
                'production_version': prod.version if prod else None,
                'production_accuracy': prod.validation_accuracy if prod else None,
                'latest_version': versions[-1].version if versions else None
            }

        return stats

    def export_model(
        self,
        model_type: str,
        version: str,
        export_path: str
    ) -> bool:
        """
        Export a model to a specified path.

        Args:
            model_type: Model type
            version: Version to export
            export_path: Destination path

        Returns:
            True if successful
        """
        weights, metadata = self.load_model(model_type, version)

        if not weights or not metadata:
            return False

        export_data = {
            'weights': weights,
            'metadata': metadata.to_dict(),
            'exported_at': datetime.now().isoformat()
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {model_type} v{version} to {export_path}")
        return True

    def import_model(
        self,
        import_path: str,
        set_as_production: bool = False
    ) -> Optional[ModelMetadata]:
        """
        Import a model from a file.

        Args:
            import_path: Path to import file
            set_as_production: Set as production after import

        Returns:
            ModelMetadata if successful
        """
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)

            weights = data['weights']
            old_metadata = data['metadata']

            # Save as new model
            return self.save_model(
                model_type=old_metadata['model_type'],
                weights_data=weights,
                accuracy=old_metadata['accuracy'],
                loss=old_metadata['loss'],
                validation_accuracy=old_metadata['validation_accuracy'],
                validation_loss=old_metadata['validation_loss'],
                samples_trained=old_metadata['samples_trained'],
                epochs_trained=old_metadata['epochs_trained'],
                training_time_ms=old_metadata['training_time_ms'],
                config=old_metadata['config'],
                feature_size=old_metadata['feature_size'],
                output_size=old_metadata['output_size'],
                name=f"Imported: {old_metadata['name']}",
                description=f"Imported from {import_path}",
                tags=['imported'],
                set_as_production=set_as_production
            )

        except Exception as e:
            logger.error(f"Failed to import model: {e}")
            return None
