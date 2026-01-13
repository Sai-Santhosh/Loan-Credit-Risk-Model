"""
Base Model Module - Abstract base class for ML models.

Provides a consistent interface for all model implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for trained model."""
    name: str
    version: str
    algorithm: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "training_samples": self.training_samples,
            "tags": self.tags,
        }


class BaseModel(ABC):
    """
    Abstract base class for Credit Risk models.
    
    Provides consistent interface for:
    - Model training
    - Prediction
    - Model persistence
    - Metadata tracking
    
    All model implementations should inherit from this class.
    """
    
    def __init__(
        self,
        name: str = "credit_risk_model",
        version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize BaseModel.
        
        Args:
            name: Model name
            version: Model version
            params: Model hyperparameters
        """
        self.name = name
        self.version = version
        self.params = params or {}
        
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.metadata: Optional[ModelMetadata] = None
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Fit model to training data.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Optional validation set
            **kwargs: Additional training arguments
        
        Returns:
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Output path
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model": self.model,
            "name": self.name,
            "version": self.version,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }
        
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
        
        Returns:
            Loaded model
        """
        state = joblib.load(path)
        
        instance = cls(
            name=state["name"],
            version=state["version"],
            params=state["params"],
        )
        
        instance.model = state["model"]
        instance.is_fitted = state["is_fitted"]
        instance.feature_names = state["feature_names"]
        
        if state["metadata"]:
            instance.metadata = ModelMetadata(**state["metadata"])
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate input features match training features."""
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()
    
    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        self.params.update(params)
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, version={self.version}, fitted={self.is_fitted})"


__all__ = ["BaseModel", "ModelMetadata"]
