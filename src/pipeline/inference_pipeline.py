"""
Inference Pipeline Module - Production inference workflow.

Provides:
- Batch inference
- Real-time inference
- Prediction explanation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ..data.data_transformer import DataTransformer
from ..models.base_model import BaseModel
from ..models.model_registry import ModelRegistry
from ..evaluation.shap_analyzer import SHAPAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray
    probabilities: np.ndarray
    threshold: float
    timestamp: str
    model_version: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "prediction": self.predictions,
            "probability": self.probabilities[:, 1],
            "default_risk": ["High" if p == 0 else "Low" for p in self.predictions],
        })


class InferencePipeline:
    """
    Production inference pipeline for Credit Risk predictions.
    
    Features:
    - Batch prediction support
    - Real-time single prediction
    - Prediction explanations with SHAP
    - Model version management
    
    Example:
        >>> pipeline = InferencePipeline.from_registry(model_name="credit_risk_lgbm")
        >>> results = pipeline.predict(df)
        >>> explanation = pipeline.explain(df.iloc[0])
    """
    
    def __init__(
        self,
        model: BaseModel,
        transformer: DataTransformer,
        model_version: str = "unknown",
        threshold: float = 0.5,
    ):
        """
        Initialize InferencePipeline.
        
        Args:
            model: Trained model
            transformer: Fitted data transformer
            model_version: Model version string
            threshold: Classification threshold
        """
        self.model = model
        self.transformer = transformer
        self.model_version = model_version
        self.threshold = threshold
        
        self.shap_analyzer: Optional[SHAPAnalyzer] = None
    
    @classmethod
    def from_registry(
        cls,
        model_name: str,
        model_version: Optional[str] = None,
        stage: Optional[str] = "production",
        registry_path: str = "artifacts/models",
        transformer_path: str = "artifacts/transformer.joblib",
    ) -> "InferencePipeline":
        """
        Create pipeline from model registry.
        
        Args:
            model_name: Name of registered model
            model_version: Specific version (None for latest)
            stage: Model stage to load from
            registry_path: Path to model registry
            transformer_path: Path to saved transformer
        
        Returns:
            Configured InferencePipeline
        """
        # Load model from registry
        registry = ModelRegistry(base_path=registry_path)
        model = registry.load_model(model_name, version=model_version, stage=stage)
        
        # Determine version
        version = model_version or registry.registry["models"][model_name].get(stage)
        
        # Load transformer
        transformer = DataTransformer.load(transformer_path)
        
        logger.info(f"Loaded model {model_name} v{version} from registry")
        
        return cls(model=model, transformer=transformer, model_version=version)
    
    @classmethod
    def from_artifacts(
        cls,
        model_path: str,
        transformer_path: str,
        model_version: str = "unknown",
    ) -> "InferencePipeline":
        """
        Create pipeline from saved artifacts.
        
        Args:
            model_path: Path to saved model
            transformer_path: Path to saved transformer
            model_version: Model version string
        
        Returns:
            Configured InferencePipeline
        """
        import joblib
        
        # Load model
        model = joblib.load(model_path)
        
        # Load transformer
        transformer = DataTransformer.load(transformer_path)
        
        logger.info(f"Loaded model from {model_path}")
        
        return cls(model=model, transformer=transformer, model_version=model_version)
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_proba: bool = True,
    ) -> PredictionResult:
        """
        Generate predictions for input data.
        
        Args:
            data: Input data (DataFrame or dict for single sample)
            return_proba: Include probabilities in result
        
        Returns:
            PredictionResult with predictions
        """
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        logger.info(f"Generating predictions for {len(data)} samples")
        
        # Transform data
        data_transformed = self.transformer.transform(data, include_target=False)
        
        # Prepare categorical features for LightGBM
        for col in self.model.categorical_features if hasattr(self.model, 'categorical_features') else []:
            if col in data_transformed.columns:
                data_transformed[col] = data_transformed[col].astype("category")
        
        # Generate predictions
        probabilities = self.model.predict_proba(data_transformed)
        predictions = (probabilities[:, 1] >= self.threshold).astype(int)
        
        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            threshold=self.threshold,
            timestamp=datetime.now().isoformat(),
            model_version=self.model_version,
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return result
    
    def predict_single(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate prediction for single sample.
        
        Args:
            data: Input data as dictionary
        
        Returns:
            Dictionary with prediction details
        """
        result = self.predict(data)
        
        return {
            "prediction": int(result.predictions[0]),
            "probability": float(result.probabilities[0, 1]),
            "default_risk": "Low" if result.predictions[0] == 1 else "High",
            "threshold": result.threshold,
            "model_version": result.model_version,
            "timestamp": result.timestamp,
        }
    
    def explain(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        sample_background: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate prediction explanation with SHAP.
        
        Args:
            data: Input data to explain
            sample_background: Background data for SHAP (optional)
        
        Returns:
            Dictionary with explanation
        """
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Transform data
        data_transformed = self.transformer.transform(data, include_target=False)
        
        # Initialize SHAP analyzer if needed
        if self.shap_analyzer is None:
            self.shap_analyzer = SHAPAnalyzer(self.model)
            
            # Use sample background or the input data
            background = sample_background if sample_background is not None else data_transformed
            self.shap_analyzer.fit(background)
        
        # Get explanation
        explanation = self.shap_analyzer.explain_prediction(
            data_transformed.iloc[0],
            threshold=self.threshold
        )
        
        return explanation
    
    def predict_batch(
        self,
        data: pd.DataFrame,
        batch_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Generate predictions for large batch.
        
        Args:
            data: Input data
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with predictions
        """
        all_results = []
        
        n_batches = (len(data) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(data)} samples in {n_batches} batches")
        
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            result = self.predict(batch)
            all_results.append(result.to_dataframe())
            
            logger.info(f"Processed batch {i // batch_size + 1}/{n_batches}")
        
        return pd.concat(all_results, ignore_index=True)
    
    def set_threshold(self, threshold: float) -> None:
        """Update classification threshold."""
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model.name,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "feature_names": self.model.feature_names,
            "n_features": len(self.model.feature_names),
        }


__all__ = ["InferencePipeline", "PredictionResult"]
