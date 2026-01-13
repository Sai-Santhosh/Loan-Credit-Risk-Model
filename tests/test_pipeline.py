"""
Tests for Pipeline Module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_transformer import DataTransformer
from src.pipeline.inference_pipeline import InferencePipeline, PredictionResult


class TestDataTransformer:
    """Test cases for DataTransformer class."""
    
    def test_initialization(self):
        """Test transformer initialization."""
        transformer = DataTransformer()
        
        assert transformer.is_fitted is False
    
    def test_fit(self, sample_raw_data):
        """Test transformer fitting."""
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        assert transformer.is_fitted is True
    
    def test_transform(self, sample_raw_data):
        """Test data transformation."""
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        transformed = transformer.transform(sample_raw_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) > 0
        # Check no nulls in final output
        assert transformed.isnull().sum().sum() == 0
    
    def test_save_load(self, sample_raw_data, temp_dir):
        """Test transformer save and load."""
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        # Save transformer
        save_path = temp_dir / "transformer.joblib"
        transformer.save(str(save_path))
        
        assert save_path.exists()
        
        # Load transformer
        loaded = DataTransformer.load(str(save_path))
        
        assert loaded.is_fitted is True
        
        # Compare transformations
        original = transformer.transform(sample_raw_data)
        loaded_result = loaded.transform(sample_raw_data)
        
        pd.testing.assert_frame_equal(original, loaded_result)


class TestInferencePipeline:
    """Test cases for InferencePipeline class."""
    
    def test_predict(self, trained_lightgbm_model, sample_raw_data, temp_dir):
        """Test inference pipeline prediction."""
        # Setup transformer
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        # Create pipeline
        pipeline = InferencePipeline(
            model=trained_lightgbm_model,
            transformer=transformer,
            model_version="test_v1"
        )
        
        # Make prediction
        result = pipeline.predict(sample_raw_data.head(10))
        
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == 10
        assert all(p in [0, 1] for p in result.predictions)
    
    def test_predict_single(self, trained_lightgbm_model, sample_raw_data):
        """Test single sample prediction."""
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        pipeline = InferencePipeline(
            model=trained_lightgbm_model,
            transformer=transformer
        )
        
        # Single prediction
        single_sample = sample_raw_data.iloc[0].to_dict()
        result = pipeline.predict_single(single_sample)
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "probability" in result
        assert "default_risk" in result
    
    def test_model_info(self, trained_lightgbm_model, sample_raw_data):
        """Test getting model info."""
        transformer = DataTransformer()
        transformer.fit(sample_raw_data)
        
        pipeline = InferencePipeline(
            model=trained_lightgbm_model,
            transformer=transformer,
            model_version="test_v1"
        )
        
        info = pipeline.get_model_info()
        
        assert "model_name" in info
        assert "model_version" in info
        assert info["model_version"] == "test_v1"
