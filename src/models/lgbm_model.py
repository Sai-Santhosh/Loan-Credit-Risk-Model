"""
LightGBM Model Module - Production LightGBM implementation.

Provides a complete LightGBM model wrapper with:
- Native categorical feature support
- Class weight handling
- Early stopping
- Feature importance
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb

from .base_model import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """
    Production-grade LightGBM classifier for credit risk.
    
    Features:
    - Native categorical feature handling
    - Automatic class weight calculation
    - Early stopping support
    - Comprehensive feature importance
    
    Example:
        >>> model = LightGBMModel(params={"n_estimators": 1000})
        >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """
    
    # Default parameters optimized for credit risk
    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "max_depth": 20,
        "num_leaves": 80,
        "learning_rate": 0.07,
        "subsample": 0.95,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 0.8,
        "min_child_samples": 25,
        "min_split_gain": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    # Categorical columns for credit risk
    CATEGORICAL_COLUMNS = [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "pub_rec",
        "initial_list_status", "pub_rec_bankruptcies"
    ]
    
    def __init__(
        self,
        name: str = "credit_risk_lgbm",
        version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        """
        Initialize LightGBMModel.
        
        Args:
            name: Model name
            version: Model version
            params: LightGBM hyperparameters
            categorical_features: List of categorical column names
        """
        # Merge default params with provided params
        merged_params = self.DEFAULT_PARAMS.copy()
        if params:
            merged_params.update(params)
        
        super().__init__(name=name, version=version, params=merged_params)
        
        self.categorical_features = categorical_features or self.CATEGORICAL_COLUMNS
        self.model: Optional[LGBMClassifier] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
        use_class_weights: bool = True,
        **kwargs
    ) -> "LightGBMModel":
        """
        Fit LightGBM model.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Optional (X_val, y_val) for early stopping
            early_stopping_rounds: Patience for early stopping
            use_class_weights: Calculate and use class weights
            **kwargs: Additional arguments for LGBMClassifier.fit()
        
        Returns:
            Fitted model
        """
        logger.info(f"Training LightGBM model with {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare categorical features
        X = self._prepare_categoricals(X)
        cat_features = [
            col for col in self.categorical_features 
            if col in X.columns
        ]
        
        # Calculate class weights
        params = self.params.copy()
        if use_class_weights:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            params["scale_pos_weight"] = neg_count / pos_count
            logger.info(f"Using scale_pos_weight: {params['scale_pos_weight']:.4f}")
        
        # Initialize model
        self.model = LGBMClassifier(**params)
        
        # Prepare fit arguments
        fit_kwargs = {"categorical_feature": cat_features}
        fit_kwargs.update(kwargs)
        
        # Handle early stopping
        callbacks = []
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._prepare_categoricals(X_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_names"] = ["valid"]
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=100))
        
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        
        # Fit model
        self.model.fit(X, y, **fit_kwargs)
        
        self.is_fitted = True
        
        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            version=self.version,
            algorithm="LightGBM",
            parameters=params,
            feature_names=self.feature_names,
            feature_importance=self.get_feature_importance(),
            training_samples=len(X),
        )
        
        logger.info(f"Model training complete. Best iteration: {self.model.best_iteration_}")
        
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
        X = self._prepare_categoricals(X)
        
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
        X = self._prepare_categoricals(X)
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
        
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
    
    def _prepare_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to category dtype."""
        X = X.copy()
        
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
        
        return X
    
    def get_best_iteration(self) -> int:
        """Get best iteration from training."""
        if not self.is_fitted:
            return 0
        return self.model.best_iteration_


__all__ = ["LightGBMModel"]
