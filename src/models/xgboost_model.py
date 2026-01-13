"""
XGBoost Model Module - Production XGBoost implementation.

Provides a complete XGBoost model wrapper with:
- GPU support
- Class weight handling
- Early stopping
- Feature importance
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    Production-grade XGBoost classifier for credit risk.
    
    Features:
    - GPU acceleration support
    - Automatic class weight calculation
    - Early stopping support
    - Multiple feature importance methods
    
    Example:
        >>> model = XGBoostModel(params={"max_depth": 5})
        >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
    """
    
    # Default parameters optimized for credit risk
    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 722,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.6,
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 0,
        "random_state": 42,
        "tree_method": "hist",  # Use 'gpu_hist' for GPU
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    # Categorical columns for credit risk
    CATEGORICAL_COLUMNS = [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "pub_rec",
        "initial_list_status", "pub_rec_bankruptcies"
    ]
    
    def __init__(
        self,
        name: str = "credit_risk_xgboost",
        version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize XGBoostModel.
        
        Args:
            name: Model name
            version: Model version
            params: XGBoost hyperparameters
            use_gpu: Enable GPU acceleration
        """
        # Merge default params with provided params
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)
        
        # Enable GPU if requested
        if use_gpu:
            merged_params["tree_method"] = "gpu_hist"
            merged_params["gpu_id"] = 0
        
        super().__init__(name=name, version=version, params=merged_params)
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        use_class_weights: bool = True,
        **kwargs
    ) -> "XGBoostModel":
        """
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Patience for early stopping
            use_class_weights: Calculate and use class weights
            **kwargs: Additional arguments
        
        Returns:
            Fitted model
        """
        logger.info(f"Training XGBoost model with {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Encode categorical features
        X = self._encode_categoricals(X, fit=True)
        
        # Calculate class weights
        params = self.params.copy()
        if use_class_weights:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            params["scale_pos_weight"] = neg_count / pos_count
            logger.info(f"Using scale_pos_weight: {params['scale_pos_weight']:.4f}")
        
        # Initialize model
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare fit arguments
        fit_kwargs = {}
        fit_kwargs.update(kwargs)
        
        # Handle early stopping
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._encode_categoricals(X_val, fit=False)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["verbose"] = 100
        
        # Fit model
        self.model.fit(X, y, **fit_kwargs)
        
        self.is_fitted = True
        
        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            version=self.version,
            algorithm="XGBoost",
            parameters=params,
            feature_names=self.feature_names,
            feature_importance=self.get_feature_importance(),
            training_samples=len(X),
        )
        
        best_iter = getattr(self.model, 'best_iteration', params.get('n_estimators', 0))
        logger.info(f"Model training complete. Best iteration: {best_iter}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of predicted classes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        self._validate_features(X)
        X = self._encode_categoricals(X, fit=False)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        self._validate_features(X)
        X = self._encode_categoricals(X, fit=False)
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
        
        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importances_
        
        return dict(zip(self.feature_names, importance))
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as sorted DataFrame."""
        importance = self.get_feature_importance()
        
        return pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ]).sort_values("importance", ascending=False)
    
    def _encode_categoricals(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.
        
        Args:
            X: DataFrame to encode
            fit: Whether to fit encoders
        
        Returns:
            Encoded DataFrame
        """
        X = X.copy()
        
        for col in self.CATEGORICAL_COLUMNS:
            if col not in X.columns:
                continue
            
            X[col] = X[col].astype(str)
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                if col in self.label_encoders:
                    # Handle unknown categories
                    known_classes = set(self.label_encoders[col].classes_)
                    X[col] = X[col].apply(
                        lambda x: x if x in known_classes else "Unknown"
                    )
                    
                    # Refit if needed
                    if "Unknown" not in known_classes:
                        new_classes = list(known_classes) + ["Unknown"]
                        self.label_encoders[col].fit(new_classes)
                    
                    X[col] = self.label_encoders[col].transform(X[col])
        
        return X
    
    def get_best_iteration(self) -> int:
        """Get best iteration from training."""
        if not self.is_fitted:
            return 0
        return getattr(self.model, 'best_iteration', 0)


__all__ = ["XGBoostModel"]
