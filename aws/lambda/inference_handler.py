"""
AWS Lambda Handler - Model Inference

Real-time credit risk prediction endpoint.
Designed for low-latency inference via API Gateway.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import boto3
import joblib
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client("s3")

# Environment variables
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "credit-risk-ml-pipeline")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/production/model.joblib")
TRANSFORMER_KEY = os.environ.get("TRANSFORMER_KEY", "models/production/transformer.joblib")
THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))

# Global variables for model caching
_model = None
_transformer = None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for model inference.
    
    Supports:
    - Single prediction (real-time)
    - Batch prediction
    - Model health check
    
    Args:
        event: Lambda event object
        context: Lambda context object
    
    Returns:
        Response dictionary
    """
    logger.info("Received inference request")
    
    try:
        # Handle different HTTP methods
        http_method = event.get("httpMethod", "POST")
        path = event.get("path", "/predict")
        
        if path == "/health" or event.get("action") == "health":
            return health_check()
        
        if http_method == "GET":
            return get_model_info()
        
        # Parse request body
        body = parse_request_body(event)
        
        if body is None:
            return error_response(400, "Invalid request body")
        
        # Determine prediction type
        if "batch" in body or isinstance(body.get("data"), list):
            return batch_predict(body)
        else:
            return single_predict(body)
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return error_response(500, str(e))


def single_predict(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate single prediction."""
    logger.info("Processing single prediction request")
    
    # Load model and transformer
    model, transformer = load_model_artifacts()
    
    # Prepare input data
    input_data = body.get("data", body)
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Transform data
    df_transformed = transform_data(df, transformer)
    
    # Generate prediction
    proba = model.predict_proba(df_transformed)[0]
    prediction = int(proba[1] >= THRESHOLD)
    
    response = {
        "prediction": prediction,
        "probability": float(proba[1]),
        "default_risk": "Low" if prediction == 1 else "High",
        "confidence": float(max(proba)),
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Prediction: {prediction}, Probability: {proba[1]:.4f}")
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(response)
    }


def batch_predict(body: Dict[str, Any]) -> Dict[str, Any]:
    """Generate batch predictions."""
    logger.info("Processing batch prediction request")
    
    # Load model and transformer
    model, transformer = load_model_artifacts()
    
    # Get batch data
    batch_data = body.get("batch", body.get("data", []))
    
    if not isinstance(batch_data, list):
        return error_response(400, "Batch data must be a list")
    
    logger.info(f"Batch size: {len(batch_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(batch_data)
    
    # Transform data
    df_transformed = transform_data(df, transformer)
    
    # Generate predictions
    probas = model.predict_proba(df_transformed)
    predictions = (probas[:, 1] >= THRESHOLD).astype(int)
    
    # Build results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probas)):
        results.append({
            "index": i,
            "prediction": int(pred),
            "probability": float(prob[1]),
            "default_risk": "Low" if pred == 1 else "High"
        })
    
    response = {
        "predictions": results,
        "batch_size": len(results),
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(response)
    }


def load_model_artifacts():
    """Load model and transformer from S3 (with caching)."""
    global _model, _transformer
    
    if _model is None:
        logger.info("Loading model from S3...")
        
        # Download model
        model_path = "/tmp/model.joblib"
        s3_client.download_file(BUCKET_NAME, MODEL_KEY, model_path)
        _model = joblib.load(model_path)
        
        logger.info("Model loaded successfully")
    
    if _transformer is None:
        logger.info("Loading transformer from S3...")
        
        # Download transformer
        transformer_path = "/tmp/transformer.joblib"
        s3_client.download_file(BUCKET_NAME, TRANSFORMER_KEY, transformer_path)
        _transformer = joblib.load(transformer_path)
        
        logger.info("Transformer loaded successfully")
    
    return _model, _transformer


def transform_data(df: pd.DataFrame, transformer) -> pd.DataFrame:
    """Transform input data for prediction."""
    # Apply transformer
    df_transformed = transformer.transform(df, include_target=False)
    
    # Handle categorical columns for LightGBM
    categorical_cols = [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "pub_rec",
        "initial_list_status", "pub_rec_bankruptcies"
    ]
    
    for col in categorical_cols:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].astype("category")
    
    return df_transformed


def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Try loading model
        model, transformer = load_model_artifacts()
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "healthy",
                "model_loaded": model is not None,
                "transformer_loaded": transformer is not None,
                "timestamp": datetime.now().isoformat()
            })
        }
    except Exception as e:
        return {
            "statusCode": 503,
            "body": json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        }


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    model, _ = load_model_artifacts()
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "model_name": getattr(model, "name", "unknown"),
            "model_version": getattr(model, "version", "unknown"),
            "threshold": THRESHOLD,
            "feature_count": len(getattr(model, "feature_names", [])),
            "timestamp": datetime.now().isoformat()
        })
    }


def parse_request_body(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse request body from event."""
    body = event.get("body")
    
    if body is None:
        # Check if data is directly in event
        if "data" in event:
            return event
        return None
    
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
    
    return body


def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "error": message,
            "timestamp": datetime.now().isoformat()
        })
    }
