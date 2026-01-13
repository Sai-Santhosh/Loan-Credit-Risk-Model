"""
Credit Risk Prediction - Core ML Pipeline Package.

A production-grade machine learning pipeline for credit risk assessment,
featuring AWS integration, MLflow experiment tracking, and enterprise-ready
model deployment capabilities.

Modules:
    - data: Data loading, validation, and transformation
    - features: Feature engineering and feature store
    - models: Model definitions and registry
    - training: Model training and hyperparameter tuning
    - evaluation: Model evaluation and SHAP analysis
    - pipeline: End-to-end training and inference pipelines
    - utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from . import data
from . import features
from . import models
from . import training
from . import evaluation
from . import pipeline
from . import utils

__all__ = [
    "data",
    "features", 
    "models",
    "training",
    "evaluation",
    "pipeline",
    "utils",
    "__version__",
]
