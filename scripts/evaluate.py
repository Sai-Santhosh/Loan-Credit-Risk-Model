#!/usr/bin/env python
"""
Credit Risk Model Evaluation Script

Evaluate trained models with comprehensive metrics and visualizations.

Usage:
    python scripts/evaluate.py --model-path artifacts/models/model.joblib
    python scripts/evaluate.py --model-name credit_risk_lgbm --stage production
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_transformer import DataTransformer
from src.models.model_registry import ModelRegistry
from src.evaluation.evaluator import Evaluator
from src.evaluation.shap_analyzer import SHAPAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Credit Risk Model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Registered model name"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        help="Model version"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["development", "staging", "production"],
        help="Model stage to load"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/test.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default="artifacts/transformer.joblib",
        help="Path to data transformer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP analysis"
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=1000,
        help="Number of samples for SHAP analysis"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CREDIT RISK MODEL EVALUATION")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== Load Model ====================
    logger.info("\n[1/4] Loading Model...")
    
    if args.model_path:
        import joblib
        model = joblib.load(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    elif args.model_name:
        registry = ModelRegistry(base_path="artifacts/models")
        model = registry.load_model(
            args.model_name,
            version=args.model_version,
            stage=args.stage
        )
        logger.info(f"Loaded {args.model_name} from registry")
    else:
        raise ValueError("Must provide --model-path or --model-name")
    
    # ==================== Load Data ====================
    logger.info("\n[2/4] Loading Test Data...")
    
    loader = DataLoader()
    df = loader.load(args.data_path)
    logger.info(f"Loaded {len(df)} test samples")
    
    # Transform data
    transformer = DataTransformer.load(args.transformer_path)
    df_processed = transformer.transform(df)
    
    # Prepare features and target
    target_col = "loan_status"
    X_test = df_processed.drop(columns=[target_col])
    y_test = df_processed[target_col].map({"Fully Paid": 1, "Charged Off": 0})
    
    # ==================== Evaluate Model ====================
    logger.info("\n[3/4] Evaluating Model...")
    
    evaluator = Evaluator()
    result = evaluator.evaluate(model, X_test, y_test)
    
    # Print metrics
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 50)
    for metric, value in result.metrics.items():
        logger.info(f"  {metric:20s}: {value:.4f}")
    logger.info(f"  {'Optimal Threshold':20s}: {result.optimal_threshold:.3f}")
    
    # Save plots
    evaluator.plot_roc_curve(result, save_path=str(output_dir / "roc_curve.png"))
    evaluator.plot_precision_recall_curve(result, save_path=str(output_dir / "pr_curve.png"))
    evaluator.plot_confusion_matrix(result, save_path=str(output_dir / "confusion_matrix.png"))
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")
    
    # ==================== SHAP Analysis ====================
    if args.shap:
        logger.info("\n[4/4] Generating SHAP Analysis...")
        
        analyzer = SHAPAnalyzer(model)
        analyzer.fit(X_test, sample_size=args.shap_samples)
        
        # Summary plots
        analyzer.plot_summary(save_path=str(output_dir / "shap_summary.png"))
        analyzer.plot_bar(save_path=str(output_dir / "shap_importance.png"))
        
        # Feature importance
        importance_df = analyzer.get_feature_importance()
        importance_df.to_csv(output_dir / "shap_importance.csv", index=False)
        
        # Save SHAP values
        analyzer.save_shap_values(str(output_dir / "shap_values.joblib"))
        
        logger.info("SHAP analysis complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
