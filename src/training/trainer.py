"""
Trainer Module - Production model training with MLflow integration.

Provides:
- Standardized training workflow
- MLflow/DagsHub experiment tracking
- Cross-validation support
- Model checkpointing
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from ..models.base_model import BaseModel
from ..models.lgbm_model import LightGBMModel
from ..models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Production model trainer with MLflow integration.
    
    Features:
    - MLflow/DagsHub experiment tracking
    - Automatic metric logging
    - Cross-validation training
    - Model checkpointing
    
    Example:
        >>> trainer = Trainer(experiment_name="credit-risk")
        >>> model, metrics = trainer.train(
        ...     model=LightGBMModel(),
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_val=X_val,
        ...     y_val=y_val
        ... )
    """
    
    def __init__(
        self,
        experiment_name: str = "credit-risk-prediction",
        tracking_uri: Optional[str] = None,
        artifact_path: str = "models",
        use_mlflow: bool = True,
        dagshub_repo: Optional[str] = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (DagsHub or local)
            artifact_path: Path for model artifacts
            use_mlflow: Enable MLflow tracking
            dagshub_repo: DagsHub repository (owner/repo)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_path = Path(artifact_path)
        self.use_mlflow = use_mlflow
        self.dagshub_repo = dagshub_repo
        
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        if self.use_mlflow:
            self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        try:
            import mlflow
            
            # Setup DagsHub if provided
            if self.dagshub_repo:
                try:
                    import dagshub
                    owner, repo = self.dagshub_repo.split("/")
                    dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
                    logger.info(f"Initialized DagsHub tracking: {self.dagshub_repo}")
                except ImportError:
                    logger.warning("DagsHub not installed. Using default MLflow tracking.")
            
            # Set tracking URI
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            self.mlflow = mlflow
            logger.info(f"MLflow initialized: {self.experiment_name}")
            
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            self.use_mlflow = False
    
    def train(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True,
        **train_kwargs
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train model with experiment tracking.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            run_name: MLflow run name
            tags: Additional MLflow tags
            log_model: Whether to log model artifact
            **train_kwargs: Additional training arguments
        
        Returns:
            Tuple of (trained model, metrics dict)
        """
        run_name = run_name or f"{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting training: {run_name}")
        logger.info(f"Training samples: {len(X_train)}, Features: {len(X_train.columns)}")
        
        metrics = {}
        
        if self.use_mlflow:
            with self.mlflow.start_run(run_name=run_name):
                # Log tags
                if tags:
                    self.mlflow.set_tags(tags)
                
                # Log parameters
                self.mlflow.log_params(model.params)
                self.mlflow.log_param("model_name", model.name)
                self.mlflow.log_param("n_train_samples", len(X_train))
                self.mlflow.log_param("n_features", len(X_train.columns))
                
                # Train model
                eval_set = (X_val, y_val) if X_val is not None else None
                model.fit(X_train, y_train, eval_set=eval_set, **train_kwargs)
                
                # Evaluate
                metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
                
                # Log metrics
                self.mlflow.log_metrics(metrics)
                
                # Log feature importance
                importance = model.get_feature_importance()
                for feat, imp in list(importance.items())[:20]:
                    self.mlflow.log_metric(f"importance_{feat}", imp)
                
                # Log model
                if log_model:
                    model_path = self.artifact_path / f"{run_name}.joblib"
                    model.save(str(model_path))
                    self.mlflow.log_artifact(str(model_path))
                
                logger.info(f"Training complete. Metrics: {metrics}")
        else:
            # Train without MLflow
            eval_set = (X_val, y_val) if X_val is not None else None
            model.fit(X_train, y_train, eval_set=eval_set, **train_kwargs)
            metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
            logger.info(f"Training complete. Metrics: {metrics}")
        
        return model, metrics
    
    def train_cv(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        run_name: Optional[str] = None,
        **train_kwargs
    ) -> Tuple[List[BaseModel], Dict[str, float]]:
        """
        Train model with cross-validation.
        
        Args:
            model_class: Model class to instantiate
            model_params: Model parameters
            X: Features
            y: Target
            n_folds: Number of CV folds
            run_name: MLflow run name
            **train_kwargs: Additional training arguments
        
        Returns:
            Tuple of (list of trained models, aggregated metrics)
        """
        run_name = run_name or f"cv_{n_folds}fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Create new model instance
            model = model_class(params=model_params)
            
            # Train fold
            model, metrics = self.train(
                model=model,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                run_name=f"{run_name}_fold{fold + 1}",
                log_model=False,
                **train_kwargs
            )
            
            fold_metrics.append(metrics)
            models.append(model)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for metric_name in fold_metrics[0].keys():
            values = [m[metric_name] for m in fold_metrics]
            aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregated_metrics[f"{metric_name}_std"] = np.std(values)
        
        logger.info(f"CV complete. Mean AUC: {aggregated_metrics.get('val_auc_mean', 'N/A'):.4f}")
        
        return models, aggregated_metrics
    
    def _evaluate_model(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, log_loss
        )
        
        metrics = {}
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["train_precision"] = precision_score(y_train, y_train_pred, zero_division=0)
        metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)
        metrics["train_auc"] = roc_auc_score(y_train, y_train_proba)
        metrics["train_logloss"] = log_loss(y_train, y_train_proba)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
            metrics["val_precision"] = precision_score(y_val, y_val_pred, zero_division=0)
            metrics["val_recall"] = recall_score(y_val, y_val_pred, zero_division=0)
            metrics["val_f1"] = f1_score(y_val, y_val_pred, zero_division=0)
            metrics["val_auc"] = roc_auc_score(y_val, y_val_proba)
            metrics["val_logloss"] = log_loss(y_val, y_val_proba)
        
        return metrics


__all__ = ["Trainer"]
