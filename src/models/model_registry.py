"""
Model Registry Module - Model versioning and tracking.

Provides model versioning, storage, and retrieval with:
- Local file system storage
- S3 integration
- MLflow integration
- Model metadata tracking
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import hashlib
import joblib

from .base_model import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Production model registry for versioning and tracking.
    
    Features:
    - Model versioning
    - Metadata tracking
    - Local and S3 storage
    - Model promotion (staging -> production)
    
    Example:
        >>> registry = ModelRegistry(base_path="models")
        >>> registry.register_model(model, metrics={"auc": 0.85})
        >>> best_model = registry.load_model("credit_risk_lgbm", stage="production")
    """
    
    STAGES = ["development", "staging", "production", "archived"]
    
    def __init__(
        self,
        base_path: str = "models",
        s3_bucket: Optional[str] = None,
    ):
        """
        Initialize ModelRegistry.
        
        Args:
            base_path: Local base path for model storage
            s3_bucket: Optional S3 bucket for cloud storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket
        self.registry_path = self.base_path / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(
        self,
        model: BaseModel,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: str = "development",
        description: str = "",
    ) -> str:
        """
        Register a trained model.
        
        Args:
            model: Trained model instance
            metrics: Model performance metrics
            tags: Additional metadata tags
            stage: Model stage (development, staging, production)
            description: Model description
        
        Returns:
            Model version ID
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage. Must be one of: {self.STAGES}")
        
        if not model.is_fitted:
            raise RuntimeError("Cannot register unfitted model")
        
        # Generate version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        model_id = f"{model.name}_{version}"
        
        # Create model directory
        model_dir = self.base_path / model.name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        model.save(str(model_path))
        
        # Prepare metadata
        metadata = {
            "model_id": model_id,
            "name": model.name,
            "version": version,
            "stage": stage,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "algorithm": model.metadata.algorithm if model.metadata else "unknown",
            "parameters": model.params,
            "metrics": metrics or {},
            "tags": tags or {},
            "feature_names": model.feature_names,
            "feature_importance": model.get_feature_importance(),
            "model_path": str(model_path),
        }
        
        # Save metadata
        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        if model.name not in self.registry["models"]:
            self.registry["models"][model.name] = {
                "versions": [],
                "production": None,
                "staging": None,
            }
        
        self.registry["models"][model.name]["versions"].append({
            "version": version,
            "created_at": metadata["created_at"],
            "stage": stage,
            "metrics": metrics or {},
        })
        
        # Update stage reference
        if stage in ["production", "staging"]:
            self.registry["models"][model.name][stage] = version
        
        self._save_registry()
        
        logger.info(f"Registered model: {model_id} (stage: {stage})")
        
        return model_id
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> BaseModel:
        """
        Load a registered model.
        
        Args:
            name: Model name
            version: Specific version (None for latest)
            stage: Load from specific stage (production, staging)
        
        Returns:
            Loaded model instance
        """
        if name not in self.registry["models"]:
            raise ValueError(f"Model not found: {name}")
        
        # Determine version
        if stage:
            version = self.registry["models"][name].get(stage)
            if not version:
                raise ValueError(f"No {stage} model for: {name}")
        elif version is None:
            # Get latest version
            versions = self.registry["models"][name]["versions"]
            if not versions:
                raise ValueError(f"No versions found for: {name}")
            version = versions[-1]["version"]
        
        model_path = self.base_path / name / version / "model.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        logger.info(f"Loaded model: {name} {version}")
        
        return model
    
    def promote_model(
        self,
        name: str,
        version: str,
        to_stage: str,
    ) -> None:
        """
        Promote model to a new stage.
        
        Args:
            name: Model name
            version: Model version
            to_stage: Target stage
        """
        if to_stage not in self.STAGES:
            raise ValueError(f"Invalid stage. Must be one of: {self.STAGES}")
        
        if name not in self.registry["models"]:
            raise ValueError(f"Model not found: {name}")
        
        # Verify version exists
        versions = [v["version"] for v in self.registry["models"][name]["versions"]]
        if version not in versions:
            raise ValueError(f"Version not found: {version}")
        
        # Archive current production model if promoting to production
        if to_stage == "production":
            current_prod = self.registry["models"][name].get("production")
            if current_prod:
                # Update stage of previous production model
                for v in self.registry["models"][name]["versions"]:
                    if v["version"] == current_prod:
                        v["stage"] = "archived"
        
        # Update stage
        self.registry["models"][name][to_stage] = version
        
        for v in self.registry["models"][name]["versions"]:
            if v["version"] == version:
                v["stage"] = to_stage
        
        self._save_registry()
        
        logger.info(f"Promoted {name} {version} to {to_stage}")
    
    def get_model_metadata(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get metadata for a model."""
        if version is None:
            versions = self.registry["models"][name]["versions"]
            version = versions[-1]["version"] if versions else None
        
        if not version:
            raise ValueError(f"No versions found for: {name}")
        
        meta_path = self.base_path / name / version / "metadata.json"
        
        with open(meta_path, "r") as f:
            return json.load(f)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []
        
        for name, info in self.registry["models"].items():
            models.append({
                "name": name,
                "versions_count": len(info.get("versions", [])),
                "production_version": info.get("production"),
                "staging_version": info.get("staging"),
            })
        
        return models
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        if name not in self.registry["models"]:
            raise ValueError(f"Model not found: {name}")
        
        return self.registry["models"][name]["versions"]
    
    def delete_version(self, name: str, version: str) -> None:
        """Delete a model version."""
        import shutil
        
        model_dir = self.base_path / name / version
        
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Update registry
        if name in self.registry["models"]:
            self.registry["models"][name]["versions"] = [
                v for v in self.registry["models"][name]["versions"]
                if v["version"] != version
            ]
            
            # Clear stage reference if needed
            for stage in ["production", "staging"]:
                if self.registry["models"][name].get(stage) == version:
                    self.registry["models"][name][stage] = None
        
        self._save_registry()
        
        logger.info(f"Deleted model version: {name} {version}")
    
    def compare_models(
        self,
        name: str,
        versions: List[str],
    ) -> pd.DataFrame:
        """
        Compare multiple model versions.
        
        Args:
            name: Model name
            versions: List of versions to compare
        
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        comparisons = []
        
        for version in versions:
            meta = self.get_model_metadata(name, version)
            comparisons.append({
                "version": version,
                "created_at": meta["created_at"],
                "stage": meta.get("stage", "unknown"),
                **meta.get("metrics", {}),
            })
        
        return pd.DataFrame(comparisons)
    
    def sync_to_s3(self, name: str, version: Optional[str] = None) -> None:
        """Sync model to S3."""
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3 sync")
        
        if version is None:
            version = self.registry["models"][name].get("production")
            if not version:
                raise ValueError(f"No production model for: {name}")
        
        model_dir = self.base_path / name / version
        s3_prefix = f"models/{name}/{version}/"
        
        s3 = boto3.client("s3")
        
        for file_path in model_dir.glob("*"):
            s3_key = f"{s3_prefix}{file_path.name}"
            s3.upload_file(str(file_path), self.s3_bucket, s3_key)
        
        logger.info(f"Synced {name} {version} to s3://{self.s3_bucket}/{s3_prefix}")


__all__ = ["ModelRegistry"]
