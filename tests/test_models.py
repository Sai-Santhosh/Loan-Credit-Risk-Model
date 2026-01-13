"""
Tests for Model Module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel


class TestLightGBMModel:
    """Test cases for LightGBMModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LightGBMModel()
        
        assert model.name == "credit_risk_lgbm"
        assert model.is_fitted is False
        assert model.model is None
    
    def test_initialization_with_params(self):
        """Test model initialization with custom parameters."""
        params = {"n_estimators": 100, "max_depth": 5}
        model = LightGBMModel(params=params)
        
        assert model.params["n_estimators"] == 100
        assert model.params["max_depth"] == 5
    
    def test_fit(self, sample_features_and_target):
        """Test model fitting."""
        X, y = sample_features_and_target
        
        model = LightGBMModel(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        
        assert model.is_fitted is True
        assert model.model is not None
        assert len(model.feature_names) == len(X.columns)
    
    def test_predict(self, trained_lightgbm_model, sample_features_and_target):
        """Test model prediction."""
        X, _ = sample_features_and_target
        
        predictions = trained_lightgbm_model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, trained_lightgbm_model, sample_features_and_target):
        """Test probability prediction."""
        X, _ = sample_features_and_target
        
        probas = trained_lightgbm_model.predict_proba(X)
        
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 2)
        assert all(0 <= p <= 1 for p in probas.flatten())
    
    def test_feature_importance(self, trained_lightgbm_model):
        """Test feature importance."""
        importance = trained_lightgbm_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_save_load(self, trained_lightgbm_model, temp_dir, sample_features_and_target):
        """Test model save and load."""
        X, _ = sample_features_and_target
        
        # Save model
        model_path = temp_dir / "model.joblib"
        trained_lightgbm_model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = LightGBMModel.load(str(model_path))
        
        assert loaded_model.is_fitted is True
        
        # Compare predictions
        original_preds = trained_lightgbm_model.predict(X)
        loaded_preds = loaded_model.predict(X)
        
        np.testing.assert_array_equal(original_preds, loaded_preds)
    
    def test_predict_without_fit_raises_error(self, sample_features_and_target):
        """Test prediction without fitting raises error."""
        X, _ = sample_features_and_target
        model = LightGBMModel()
        
        with pytest.raises(RuntimeError):
            model.predict(X)


class TestXGBoostModel:
    """Test cases for XGBoostModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = XGBoostModel()
        
        assert model.name == "credit_risk_xgboost"
        assert model.is_fitted is False
    
    def test_fit(self, sample_features_and_target):
        """Test model fitting."""
        X, y = sample_features_and_target
        
        model = XGBoostModel(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        
        assert model.is_fitted is True
        assert model.model is not None
    
    def test_predict(self, sample_features_and_target):
        """Test model prediction."""
        X, y = sample_features_and_target
        
        model = XGBoostModel(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
