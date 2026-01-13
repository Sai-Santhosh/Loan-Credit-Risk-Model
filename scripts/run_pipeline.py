#!/usr/bin/env python
"""
Credit Risk ML Pipeline Runner

Execute the complete ML pipeline from data ingestion to model deployment.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/config.yaml
    python scripts/run_pipeline.py --steps data,train,evaluate
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.training_pipeline import TrainingPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Credit Risk ML Pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lightgbm", "xgboost"],
        default="lightgbm",
        help="Model type"
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
        help="DagsHub repository for MLflow tracking"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts",
        help="Output path for artifacts"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("CREDIT RISK ML PIPELINE")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Create pipeline configuration
    config = PipelineConfig(
        data_path=args.data_path or "data/raw/train.csv",
        model_type=args.model_type,
        use_hyperparameter_tuning=args.tune,
        n_tuning_trials=args.n_trials,
        dagshub_repo=args.dagshub_repo,
        output_path=args.output_path,
    )
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Status: {results['status']}")
    
    if results['status'] == 'success':
        logger.info(f"Model ID: {results['model_id']}")
        logger.info(f"Test AUC: {results['eval_metrics']['auc']:.4f}")
        logger.info(f"Test F1: {results['eval_metrics']['f1']:.4f}")
        logger.info(f"Duration: {results['duration_seconds']:.1f}s")
    
    return results


if __name__ == "__main__":
    main()
