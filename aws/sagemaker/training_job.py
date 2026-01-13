"""
AWS SageMaker Training Job - Credit Risk Model Training

Production training job configuration and execution using SageMaker.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn import SKLearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Configuration
REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "credit-risk-ml-pipeline")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")


class SageMakerTrainingJob:
    """
    SageMaker training job manager for credit risk model.
    
    Handles:
    - Training job configuration
    - Hyperparameter tuning jobs
    - Model artifact management
    """
    
    def __init__(
        self,
        role_arn: Optional[str] = None,
        bucket: Optional[str] = None,
        region: str = REGION
    ):
        """
        Initialize SageMaker training job manager.
        
        Args:
            role_arn: SageMaker execution role ARN
            bucket: S3 bucket for artifacts
            region: AWS region
        """
        self.region = region
        self.bucket = bucket or BUCKET_NAME
        self.role = role_arn or ROLE_ARN
        
        # Initialize SageMaker session
        self.session = sagemaker.Session(
            boto_session=boto3.Session(region_name=region)
        )
        
        if not self.role:
            self.role = sagemaker.get_execution_role()
    
    def create_training_job(
        self,
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        hyperparameters: Optional[Dict[str, Any]] = None,
        train_data_path: Optional[str] = None,
        validation_data_path: Optional[str] = None,
        max_runtime_seconds: int = 86400,
    ) -> str:
        """
        Create and run a SageMaker training job.
        
        Args:
            job_name: Training job name
            instance_type: EC2 instance type
            instance_count: Number of instances
            hyperparameters: Model hyperparameters
            train_data_path: S3 path to training data
            validation_data_path: S3 path to validation data
            max_runtime_seconds: Maximum training time
        
        Returns:
            Training job name
        """
        # Generate job name if not provided
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"credit-risk-training-{timestamp}"
        
        # Default paths
        train_data_path = train_data_path or f"s3://{self.bucket}/data/processed/train/"
        validation_data_path = validation_data_path or f"s3://{self.bucket}/data/processed/validation/"
        
        # Default hyperparameters
        default_hyperparameters = {
            "model_type": "lightgbm",
            "n_estimators": "1000",
            "max_depth": "20",
            "learning_rate": "0.07",
            "num_leaves": "80",
            "subsample": "0.95",
            "colsample_bytree": "0.8",
            "reg_alpha": "0.3",
            "reg_lambda": "0.8",
        }
        
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        # Create estimator
        estimator = SKLearn(
            entry_point="train.py",
            source_dir="src",
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version="1.2-1",
            py_version="py3",
            hyperparameters=default_hyperparameters,
            output_path=f"s3://{self.bucket}/models/",
            base_job_name="credit-risk",
            sagemaker_session=self.session,
            max_run=max_runtime_seconds,
            metric_definitions=[
                {"Name": "train:auc", "Regex": "train_auc=([0-9.]+)"},
                {"Name": "validation:auc", "Regex": "val_auc=([0-9.]+)"},
                {"Name": "train:loss", "Regex": "train_loss=([0-9.]+)"},
            ],
        )
        
        # Configure training inputs
        train_input = TrainingInput(
            train_data_path,
            content_type="application/x-parquet"
        )
        
        validation_input = TrainingInput(
            validation_data_path,
            content_type="application/x-parquet"
        )
        
        # Start training
        logger.info(f"Starting training job: {job_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Training data: {train_data_path}")
        
        estimator.fit(
            inputs={
                "train": train_input,
                "validation": validation_input
            },
            job_name=job_name,
            wait=False
        )
        
        logger.info(f"Training job {job_name} started")
        
        return job_name
    
    def create_hyperparameter_tuning_job(
        self,
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        max_jobs: int = 20,
        max_parallel_jobs: int = 4,
    ) -> str:
        """
        Create a hyperparameter tuning job.
        
        Args:
            job_name: Tuning job name
            instance_type: EC2 instance type
            max_jobs: Maximum number of training jobs
            max_parallel_jobs: Maximum parallel jobs
        
        Returns:
            Tuning job name
        """
        from sagemaker.tuner import (
            HyperparameterTuner,
            IntegerParameter,
            ContinuousParameter,
            CategoricalParameter
        )
        
        # Generate job name
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"credit-risk-tuning-{timestamp}"
        
        # Create base estimator
        estimator = SKLearn(
            entry_point="train.py",
            source_dir="src",
            role=self.role,
            instance_type=instance_type,
            framework_version="1.2-1",
            py_version="py3",
            output_path=f"s3://{self.bucket}/models/",
            sagemaker_session=self.session,
        )
        
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            "n_estimators": IntegerParameter(100, 2000),
            "max_depth": IntegerParameter(3, 30),
            "learning_rate": ContinuousParameter(0.01, 0.3),
            "num_leaves": IntegerParameter(20, 150),
            "subsample": ContinuousParameter(0.5, 1.0),
            "colsample_bytree": ContinuousParameter(0.5, 1.0),
            "reg_alpha": ContinuousParameter(0.0, 10.0),
            "reg_lambda": ContinuousParameter(0.0, 10.0),
        }
        
        # Create tuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name="validation:auc",
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            strategy="Bayesian",
            objective_type="Maximize",
        )
        
        # Start tuning
        train_data_path = f"s3://{self.bucket}/data/processed/train/"
        validation_data_path = f"s3://{self.bucket}/data/processed/validation/"
        
        logger.info(f"Starting hyperparameter tuning job: {job_name}")
        
        tuner.fit(
            inputs={
                "train": train_data_path,
                "validation": validation_data_path
            },
            job_name=job_name,
            wait=False
        )
        
        return job_name
    
    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a training job."""
        sm_client = boto3.client("sagemaker", region_name=self.region)
        
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        
        return {
            "job_name": job_name,
            "status": response["TrainingJobStatus"],
            "secondary_status": response.get("SecondaryStatus"),
            "creation_time": str(response["CreationTime"]),
            "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            "billable_seconds": response.get("BillableTimeInSeconds"),
        }
    
    def download_model_artifacts(
        self,
        job_name: str,
        local_path: str = "models/"
    ) -> str:
        """Download model artifacts from completed training job."""
        import tarfile
        
        # Get job details
        job_status = self.get_training_job_status(job_name)
        
        if job_status["status"] != "Completed":
            raise RuntimeError(f"Training job not completed: {job_status['status']}")
        
        model_artifacts_path = job_status["model_artifacts"]
        
        # Download from S3
        s3_client = boto3.client("s3")
        
        # Parse S3 path
        bucket, key = model_artifacts_path.replace("s3://", "").split("/", 1)
        
        local_tar_path = os.path.join(local_path, "model.tar.gz")
        os.makedirs(local_path, exist_ok=True)
        
        logger.info(f"Downloading model artifacts from {model_artifacts_path}")
        s3_client.download_file(bucket, key, local_tar_path)
        
        # Extract
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=local_path)
        
        logger.info(f"Model artifacts extracted to {local_path}")
        
        return local_path


def main():
    """Main function for running training job."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker Training Job")
    parser.add_argument("--job-name", type=str, help="Training job name")
    parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    
    args = parser.parse_args()
    
    trainer = SageMakerTrainingJob()
    
    if args.tune:
        job_name = trainer.create_hyperparameter_tuning_job(
            job_name=args.job_name,
            instance_type=args.instance_type
        )
        print(f"Started hyperparameter tuning job: {job_name}")
    else:
        job_name = trainer.create_training_job(
            job_name=args.job_name,
            instance_type=args.instance_type
        )
        print(f"Started training job: {job_name}")


if __name__ == "__main__":
    main()
