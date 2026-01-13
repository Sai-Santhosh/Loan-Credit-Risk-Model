"""
Configuration Module for Credit Risk Prediction Pipeline.

This module provides configuration management utilities for the ML pipeline,
including YAML config loading, environment variable handling, and validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from dataclasses import dataclass, field


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str = "us-east-1"
    s3_bucket: str = "credit-risk-ml-pipeline"
    redshift_host: Optional[str] = None
    sagemaker_role: Optional[str] = None


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "credit-risk-prediction"
    log_models: bool = True
    log_artifacts: bool = True


@dataclass
class ModelConfig:
    """Model training configuration."""
    name: str = "lightgbm"
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


@dataclass 
class Config:
    """Main configuration container."""
    aws: AWSConfig = field(default_factory=AWSConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Expand environment variables
        config_dict = cls._expand_env_vars(config_dict)
        
        aws_config = AWSConfig(**config_dict.get('aws', {})) if 'aws' in config_dict else AWSConfig()
        mlflow_config = MLflowConfig(**config_dict.get('mlflow', {})) if 'mlflow' in config_dict else MLflowConfig()
        model_config = ModelConfig(**config_dict.get('model', {})) if 'model' in config_dict else ModelConfig()
        
        return cls(aws=aws_config, mlflow=mlflow_config, model=model_config)
    
    @staticmethod
    def _expand_env_vars(config: Any) -> Any:
        """Recursively expand environment variables in config values."""
        if isinstance(config, dict):
            return {k: Config._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Config._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = Config._expand_env_vars(config)
    
    return config


# Export configuration
__all__ = ['Config', 'AWSConfig', 'MLflowConfig', 'ModelConfig', 'load_config']
