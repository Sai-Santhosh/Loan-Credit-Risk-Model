"""
Tests for Data Loader Module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_load_csv(self, temp_dir, sample_raw_data):
        """Test loading CSV file."""
        # Save sample data
        csv_path = temp_dir / "test_data.csv"
        sample_raw_data.to_csv(csv_path, index=False)
        
        # Load data
        loader = DataLoader()
        df = loader.load(str(csv_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_data)
        assert list(df.columns) == list(sample_raw_data.columns)
    
    def test_load_parquet(self, temp_dir, sample_raw_data):
        """Test loading Parquet file."""
        # Save sample data
        parquet_path = temp_dir / "test_data.parquet"
        sample_raw_data.to_parquet(parquet_path, index=False)
        
        # Load data
        loader = DataLoader()
        df = loader.load(str(parquet_path))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_data)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.csv")
    
    def test_save_csv(self, temp_dir, sample_raw_data):
        """Test saving CSV file."""
        csv_path = temp_dir / "output.csv"
        
        loader = DataLoader()
        loader.save(sample_raw_data, str(csv_path))
        
        assert csv_path.exists()
        
        # Verify content
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(sample_raw_data)
    
    def test_save_parquet(self, temp_dir, sample_raw_data):
        """Test saving Parquet file."""
        parquet_path = temp_dir / "output.parquet"
        
        loader = DataLoader()
        loader.save(sample_raw_data, str(parquet_path))
        
        assert parquet_path.exists()
        
        # Verify content
        loaded = pd.read_parquet(parquet_path)
        assert len(loaded) == len(sample_raw_data)
