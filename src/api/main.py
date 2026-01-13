"""
FastAPI Application - Credit Risk Prediction API

Production REST API for real-time credit risk predictions.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production ML API for credit risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model = None
_transformer = None


# ==================== Pydantic Models ====================

class LoanApplication(BaseModel):
    """Input schema for loan application."""
    loan_amnt: float = Field(..., description="Loan amount", ge=0)
    term: str = Field(..., description="Loan term (36 months or 60 months)")
    int_rate: float = Field(..., description="Interest rate", ge=0, le=100)
    installment: float = Field(..., description="Monthly installment", ge=0)
    grade: str = Field(..., description="Loan grade (A-G)")
    emp_length: Optional[str] = Field(None, description="Employment length")
    home_ownership: str = Field(..., description="Home ownership status")
    annual_inc: float = Field(..., description="Annual income", ge=0)
    verification_status: str = Field(..., description="Verification status")
    purpose: str = Field(..., description="Loan purpose")
    dti: float = Field(..., description="Debt-to-income ratio", ge=0)
    open_acc: float = Field(..., description="Open credit accounts", ge=0)
    pub_rec: float = Field(0, description="Public records", ge=0)
    revol_bal: float = Field(0, description="Revolving balance", ge=0)
    revol_util: Optional[float] = Field(None, description="Revolving utilization")
    total_acc: float = Field(..., description="Total accounts", ge=0)
    initial_list_status: str = Field("f", description="Initial list status")
    mort_acc: Optional[float] = Field(None, description="Mortgage accounts")
    pub_rec_bankruptcies: Optional[float] = Field(None, description="Bankruptcies")
    earliest_cr_line: Optional[str] = Field(None, description="Earliest credit line date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 10000,
                "term": " 36 months",
                "int_rate": 12.5,
                "installment": 335.47,
                "grade": "B",
                "emp_length": "5 years",
                "home_ownership": "RENT",
                "annual_inc": 60000,
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "dti": 15.5,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 5000,
                "revol_util": 45.0,
                "total_acc": 15,
                "initial_list_status": "f",
                "mort_acc": 0,
                "pub_rec_bankruptcies": 0,
                "earliest_cr_line": "Jan-2010"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction."""
    prediction: int = Field(..., description="0=Default, 1=Fully Paid")
    probability: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., description="Model confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    applications: List[LoanApplication]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    batch_size: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


# ==================== Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    global _model
    
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication):
    """
    Generate credit risk prediction for a single loan application.
    
    Returns probability of default and risk classification.
    """
    try:
        # Load model if not cached
        model, transformer = await load_model_artifacts()
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame([application.dict()])
        
        # Transform data
        df_transformed = transformer.transform(df, include_target=False)
        
        # Prepare categorical features
        categorical_cols = [
            "term", "grade", "emp_length", "home_ownership",
            "verification_status", "purpose", "pub_rec",
            "initial_list_status", "pub_rec_bankruptcies"
        ]
        for col in categorical_cols:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].astype("category")
        
        # Generate prediction
        proba = model.predict_proba(df_transformed)[0]
        prediction = int(proba[1] >= 0.5)
        
        # Determine risk level
        default_prob = 1 - proba[1]
        if default_prob < 0.2:
            risk_level = "Low"
        elif default_prob < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(default_prob),
            risk_level=risk_level,
            confidence=float(max(proba)),
            timestamp=datetime.now().isoformat(),
            model_version=getattr(model, "version", "1.0.0")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate predictions for multiple loan applications.
    """
    predictions = []
    
    for application in request.applications:
        pred = await predict(application)
        predictions.append(pred)
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions),
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    model, _ = await load_model_artifacts()
    
    return {
        "name": getattr(model, "name", "credit_risk_model"),
        "version": getattr(model, "version", "1.0.0"),
        "algorithm": getattr(model, "metadata", {}).algorithm if hasattr(model, "metadata") and model.metadata else "LightGBM",
        "features": len(getattr(model, "feature_names", [])),
        "feature_names": getattr(model, "feature_names", [])[:10]  # First 10
    }


# ==================== Helper Functions ====================

async def load_model_artifacts():
    """Load model and transformer (with caching)."""
    global _model, _transformer
    
    if _model is None:
        logger.info("Loading model artifacts...")
        
        import joblib
        
        model_path = Path("artifacts/models/model.joblib")
        transformer_path = Path("artifacts/transformer.joblib")
        
        if model_path.exists():
            _model = joblib.load(model_path)
        else:
            # Create dummy model for testing
            logger.warning("Model not found, using dummy model")
            from src.models.lgbm_model import LightGBMModel
            _model = LightGBMModel()
        
        if transformer_path.exists():
            from src.data.data_transformer import DataTransformer
            _transformer = DataTransformer.load(str(transformer_path))
        else:
            logger.warning("Transformer not found, using new instance")
            from src.data.data_transformer import DataTransformer
            _transformer = DataTransformer()
    
    return _model, _transformer


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Credit Risk Prediction API...")
    
    # Pre-load model
    try:
        await load_model_artifacts()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Credit Risk Prediction API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
