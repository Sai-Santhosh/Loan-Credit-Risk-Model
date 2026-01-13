"""
Hyperparameter Tuner Module - Automated hyperparameter optimization.

Provides:
- Optuna-based hyperparameter search
- RandomizedSearchCV fallback
- MLflow integration for experiment tracking
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ..models.base_model import BaseModel
from ..models.lgbm_model import LightGBMModel
from ..models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Production hyperparameter tuner with Optuna.
    
    Features:
    - Optuna optimization with pruning
    - MLflow experiment tracking
    - Support for LightGBM and XGBoost
    - Custom search spaces
    
    Example:
        >>> tuner = HyperparameterTuner(model_type="lightgbm")
        >>> best_params, best_score = tuner.tune(X_train, y_train, n_trials=100)
        >>> model = tuner.get_best_model()
    """
    
    # Default search spaces
    LIGHTGBM_SEARCH_SPACE = {
        "n_estimators": {"type": "int", "low": 100, "high": 2000},
        "max_depth": {"type": "int", "low": 3, "high": 30},
        "num_leaves": {"type": "int", "low": 20, "high": 150},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
    }
    
    XGBOOST_SEARCH_SPACE = {
        "n_estimators": {"type": "int", "low": 100, "high": 2000},
        "max_depth": {"type": "int", "low": 3, "high": 15},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
    }
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        search_space: Optional[Dict[str, Any]] = None,
        scoring: str = "roc_auc",
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize HyperparameterTuner.
        
        Args:
            model_type: Model type ('lightgbm' or 'xgboost')
            search_space: Custom search space (uses defaults if None)
            scoring: Scoring metric for optimization
            cv_folds: Number of CV folds
            random_state: Random seed
        """
        self.model_type = model_type
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Set search space
        if search_space:
            self.search_space = search_space
        elif model_type == "lightgbm":
            self.search_space = self.LIGHTGBM_SEARCH_SPACE
        elif model_type == "xgboost":
            self.search_space = self.XGBOOST_SEARCH_SPACE
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Set model class
        self.model_class = LightGBMModel if model_type == "lightgbm" else XGBoostModel
        
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.best_model: Optional[BaseModel] = None
        self.study = None
    
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training target
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs
            show_progress: Show progress bar
        
        Returns:
            Tuple of (best parameters, best score)
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            return self._tune_optuna(
                X, y, n_trials, timeout, n_jobs, show_progress
            )
        except ImportError:
            logger.warning("Optuna not installed. Using RandomizedSearchCV.")
            return self._tune_sklearn(X, y, n_trials)
    
    def _tune_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int,
        timeout: Optional[int],
        n_jobs: int,
        show_progress: bool,
    ) -> Tuple[Dict[str, Any], float]:
        """Run Optuna optimization."""
        import optuna
        from optuna.samplers import TPESampler
        
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        
        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)
            
            # Create model
            model = self.model_class(params=params)
            
            # Prepare data for categorical features
            X_prepared = X.copy()
            if self.model_type == "lightgbm":
                for col in model.categorical_features:
                    if col in X_prepared.columns:
                        X_prepared[col] = X_prepared[col].astype("category")
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            scores = []
            for train_idx, val_idx in cv.split(X_prepared, y):
                X_train = X_prepared.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X_prepared.iloc[val_idx]
                y_val = y.iloc[val_idx]
                
                model = self.model_class(params=params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(np.mean(scores), len(scores) - 1)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization complete. Best AUC: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def _tune_sklearn(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Fallback to sklearn RandomizedSearchCV."""
        from sklearn.model_selection import RandomizedSearchCV
        
        logger.info(f"Starting RandomizedSearchCV with {n_iter} iterations")
        
        # Convert search space to sklearn format
        param_distributions = {}
        for param, config in self.search_space.items():
            if config["type"] == "int":
                param_distributions[param] = list(range(config["low"], config["high"] + 1))
            elif config["type"] == "float":
                param_distributions[param] = np.linspace(config["low"], config["high"], 50)
        
        # Create base estimator
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            estimator = LGBMClassifier(random_state=self.random_state, verbose=-1)
        else:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(random_state=self.random_state, verbosity=0)
        
        # Run search
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv_folds,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )
        
        search.fit(X, y)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        logger.info(f"Search complete. Best AUC: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def _sample_params(self, trial) -> Dict[str, Any]:
        """Sample parameters from search space using Optuna trial."""
        params = {}
        
        for param, config in self.search_space.items():
            if config["type"] == "int":
                params[param] = trial.suggest_int(param, config["low"], config["high"])
            elif config["type"] == "float":
                if config.get("log", False):
                    params[param] = trial.suggest_float(
                        param, config["low"], config["high"], log=True
                    )
                else:
                    params[param] = trial.suggest_float(
                        param, config["low"], config["high"]
                    )
            elif config["type"] == "categorical":
                params[param] = trial.suggest_categorical(param, config["choices"])
        
        return params
    
    def get_best_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> BaseModel:
        """
        Train model with best parameters.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Trained model with best parameters
        """
        if self.best_params is None:
            raise RuntimeError("Must run tune() before getting best model")
        
        self.best_model = self.model_class(params=self.best_params)
        
        eval_set = (X_val, y_val) if X_val is not None else None
        self.best_model.fit(X, y, eval_set=eval_set)
        
        return self.best_model
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                **trial.params,
            }
            trials_data.append(trial_data)
        
        return pd.DataFrame(trials_data)
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import optuna.visualization as viz
            
            if self.study is None:
                logger.warning("No study available to plot")
                return
            
            # Plot optimization history
            fig = viz.plot_optimization_history(self.study)
            fig.show()
            
            # Plot parameter importances
            fig = viz.plot_param_importances(self.study)
            fig.show()
            
        except ImportError:
            logger.warning("Optuna visualization not available")


__all__ = ["HyperparameterTuner"]
