#!/usr/bin/env python
"""
Credit Risk Model Training Script

Production training script with MLflow integration and DagsHub tracking.
Supports both LightGBM and XGBoost models with hyperparameter tuning.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --model lightgbm --tune
    python scripts/train.py --dagshub-repo username/credit-risk
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader
from src.data.data_transformer import DataTransformer
from src.data.data_validator import DataValidator
from src.models.lgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.models.model_registry import ModelRegistry
from src.training.trainer import Trainer
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.evaluation.evaluator import Evaluator
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Credit Risk Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data (overrides config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lightgbm", "xgboost"],
        default="lightgbm",
        help="Model type to train"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of tuning trials"
    )
    parser.add_argument(
        "--dagshub-repo",
        type=str,
        help="DagsHub repository (owner/repo) for MLflow tracking"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="credit-risk-prediction",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CREDIT RISK MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Hyperparameter Tuning: {args.tune}")
    logger.info(f"DagsHub Repo: {args.dagshub_repo or 'Not configured'}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else {}
    
    # ==================== Data Loading ====================
    logger.info("\n[1/6] Loading Data...")
    
    data_path = args.data_path or config.get("data", {}).get("raw", {}).get("train_path", "data/raw/train.csv")
    
    loader = DataLoader()
    df = loader.load(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # ==================== Data Validation ====================
    logger.info("\n[2/6] Validating Data...")
    
    validator = DataValidator()
    validation_result = validator.validate(df)
    
    if not validation_result.is_valid:
        logger.warning(f"Validation warnings: {validation_result.warnings}")
    
    # ==================== Data Preprocessing ====================
    logger.info("\n[3/6] Preprocessing Data...")
    
    transformer = DataTransformer()
    transformer.fit(df)
    df_processed = transformer.transform(df)
    
    # Save transformer
    transformer.save(str(output_dir / "transformer.joblib"))
    logger.info(f"Processed data: {len(df_processed)} rows")
    
    # ==================== Train/Test Split ====================
    logger.info("\n[4/6] Creating Train/Test Split...")
    
    target_col = "loan_status"
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col].map({"Fully Paid": 1, "Charged Off": 0})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ==================== Model Training ====================
    logger.info("\n[5/6] Training Model...")
    
    # Initialize trainer
    trainer = Trainer(
        experiment_name=args.experiment_name,
        dagshub_repo=args.dagshub_repo,
        artifact_path=str(output_dir / "models")
    )
    
    # Select model class
    model_class = LightGBMModel if args.model == "lightgbm" else XGBoostModel
    
    # Hyperparameter tuning
    if args.tune:
        logger.info(f"Running hyperparameter tuning ({args.n_trials} trials)...")
        tuner = HyperparameterTuner(model_type=args.model)
        best_params, best_score = tuner.tune(
            X_train, y_train,
            n_trials=args.n_trials
        )
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        model = model_class(params=best_params)
    else:
        model = model_class()
    
    # Train model
    model, metrics = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # ==================== Model Evaluation ====================
    logger.info("\n[6/6] Evaluating Model...")
    
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(model, X_test, y_test)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    for metric, value in eval_result.metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"  Optimal Threshold: {eval_result.optimal_threshold:.3f}")
    
    # Save plots
    evaluator.plot_roc_curve(eval_result, save_path=str(output_dir / "roc_curve.png"))
    evaluator.plot_confusion_matrix(eval_result, save_path=str(output_dir / "confusion_matrix.png"))
    
    # ==================== Model Registration ====================
    logger.info("\nRegistering Model...")
    
    registry = ModelRegistry(base_path=str(output_dir / "models"))
    model_id = registry.register_model(
        model=model,
        metrics=eval_result.metrics,
        stage="staging",
        description=f"Trained with {args.model}, tune={args.tune}"
    )
    
    logger.info(f"\nModel registered: {model_id}")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return eval_result.metrics


if __name__ == "__main__":
    main()
