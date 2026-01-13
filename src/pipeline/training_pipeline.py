"""
Training Pipeline Module - End-to-end training orchestration.

Provides:
- Complete training workflow
- Data loading to model deployment
- Experiment tracking integration
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from ..data.data_loader import DataLoader, S3DataLoader
from ..data.data_transformer import DataTransformer
from ..data.data_validator import DataValidator
from ..features.feature_engineering import FeatureEngineer
from ..models.lgbm_model import LightGBMModel
from ..models.xgboost_model import XGBoostModel
from ..models.model_registry import ModelRegistry
from ..training.trainer import Trainer
from ..training.hyperparameter_tuner import HyperparameterTuner
from ..evaluation.evaluator import Evaluator
from ..evaluation.shap_analyzer import SHAPAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""
    data_path: str = "data/raw/train.csv"
    model_type: str = "lightgbm"  # 'lightgbm' or 'xgboost'
    test_size: float = 0.2
    use_hyperparameter_tuning: bool = True
    n_tuning_trials: int = 50
    experiment_name: str = "credit-risk-prediction"
    dagshub_repo: Optional[str] = None
    output_path: str = "artifacts"


class TrainingPipeline:
    """
    Production training pipeline for Credit Risk model.
    
    Orchestrates the complete training workflow:
    1. Data loading and validation
    2. Data preprocessing
    3. Feature engineering
    4. Model training (with optional hyperparameter tuning)
    5. Model evaluation
    6. SHAP analysis
    7. Model registration
    
    Example:
        >>> pipeline = TrainingPipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize TrainingPipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.output_path = Path(self.config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.trainer = Trainer(
            experiment_name=self.config.experiment_name,
            dagshub_repo=self.config.dagshub_repo,
        )
        self.evaluator = Evaluator()
        self.registry = ModelRegistry(base_path=str(self.output_path / "models"))
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def run(
        self,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Args:
            data_path: Override data path from config
        
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STARTING CREDIT RISK TRAINING PIPELINE")
        logger.info("=" * 60)
        
        data_path = data_path or self.config.data_path
        
        try:
            # Step 1: Load Data
            logger.info("\n[STEP 1/7] Loading Data...")
            df = self._load_data(data_path)
            
            # Step 2: Validate Data
            logger.info("\n[STEP 2/7] Validating Data...")
            validation_result = self._validate_data(df)
            
            # Step 3: Preprocess Data
            logger.info("\n[STEP 3/7] Preprocessing Data...")
            df_processed = self._preprocess_data(df)
            
            # Step 4: Prepare Train/Test Split
            logger.info("\n[STEP 4/7] Preparing Train/Test Split...")
            X_train, X_test, y_train, y_test = self._prepare_splits(df_processed)
            
            # Step 5: Train Model
            logger.info("\n[STEP 5/7] Training Model...")
            model, train_metrics = self._train_model(X_train, y_train, X_test, y_test)
            
            # Step 6: Evaluate Model
            logger.info("\n[STEP 6/7] Evaluating Model...")
            eval_result = self._evaluate_model(model, X_test, y_test)
            
            # Step 7: SHAP Analysis
            logger.info("\n[STEP 7/7] Generating SHAP Analysis...")
            shap_importance = self._generate_shap_analysis(model, X_test)
            
            # Register Model
            logger.info("\n[FINAL] Registering Model...")
            model_id = self._register_model(model, eval_result.metrics)
            
            # Compile results
            self.results = {
                "status": "success",
                "model_id": model_id,
                "train_metrics": train_metrics,
                "eval_metrics": eval_result.metrics,
                "optimal_threshold": eval_result.optimal_threshold,
                "feature_importance": shap_importance,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Model ID: {model_id}")
            logger.info(f"Test AUC: {eval_result.metrics['auc']:.4f}")
            logger.info(f"Duration: {self.results['duration_seconds']:.1f} seconds")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.results = {
                "status": "failed",
                "error": str(e),
            }
            raise
        
        return self.results
    
    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from source."""
        df = self.data_loader.load(path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate data quality."""
        result = self.validator.validate(df)
        
        if not result.is_valid:
            logger.warning(f"Data validation issues: {result.errors}")
        
        return result
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        self.transformer.fit(df)
        df_processed = self.transformer.transform(df)
        
        # Save transformer
        self.transformer.save(str(self.output_path / "transformer.joblib"))
        
        logger.info(f"Processed data: {len(df_processed)} rows")
        return df_processed
    
    def _prepare_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test splits."""
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        target_col = "loan_status"
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target
        y = y.map({"Fully Paid": 1, "Charged Off": 0})
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train model with optional hyperparameter tuning."""
        
        # Select model class
        if self.config.model_type == "lightgbm":
            model_class = LightGBMModel
        else:
            model_class = XGBoostModel
        
        # Hyperparameter tuning
        if self.config.use_hyperparameter_tuning:
            logger.info("Running hyperparameter tuning...")
            tuner = HyperparameterTuner(model_type=self.config.model_type)
            best_params, best_score = tuner.tune(
                X_train, y_train,
                n_trials=self.config.n_tuning_trials
            )
            model = model_class(params=best_params)
        else:
            model = model_class()
        
        # Train model
        model, metrics = self.trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        
        return model, metrics
    
    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        result = self.evaluator.evaluate(model, X_test, y_test)
        
        # Save evaluation plots
        self.evaluator.plot_roc_curve(
            result,
            save_path=str(self.output_path / "roc_curve.png")
        )
        self.evaluator.plot_confusion_matrix(
            result,
            save_path=str(self.output_path / "confusion_matrix.png")
        )
        
        return result
    
    def _generate_shap_analysis(
        self, model, X_test: pd.DataFrame
    ) -> Dict[str, float]:
        """Generate SHAP analysis."""
        analyzer = SHAPAnalyzer(model)
        analyzer.fit(X_test, sample_size=1000)
        
        # Save plots
        analyzer.plot_summary(
            save_path=str(self.output_path / "shap_summary.png")
        )
        analyzer.plot_bar(
            save_path=str(self.output_path / "shap_importance.png")
        )
        
        # Save SHAP values
        analyzer.save_shap_values(str(self.output_path / "shap_values.joblib"))
        
        importance_df = analyzer.get_feature_importance()
        return dict(zip(importance_df["feature"], importance_df["importance"]))
    
    def _register_model(
        self, model, metrics: Dict[str, float]
    ) -> str:
        """Register trained model."""
        model_id = self.registry.register_model(
            model=model,
            metrics=metrics,
            stage="staging",
            description="Credit risk prediction model trained via pipeline",
        )
        return model_id


__all__ = ["TrainingPipeline", "PipelineConfig"]
