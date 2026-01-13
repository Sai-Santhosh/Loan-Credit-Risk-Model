"""
Pytest Configuration and Fixtures.

Provides shared fixtures for test cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_raw_data():
    """Create sample raw loan data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "loan_amnt": np.random.uniform(1000, 40000, n_samples),
        "term": np.random.choice([" 36 months", " 60 months"], n_samples),
        "int_rate": np.random.uniform(5, 30, n_samples),
        "installment": np.random.uniform(50, 1500, n_samples),
        "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
        "sub_grade": np.random.choice(["A1", "B2", "C3", "D4"], n_samples),
        "emp_title": np.random.choice(["Engineer", "Manager", "Teacher", None], n_samples),
        "emp_length": np.random.choice(["10+ years", "5 years", "< 1 year", None], n_samples),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n_samples),
        "annual_inc": np.random.uniform(30000, 200000, n_samples),
        "verification_status": np.random.choice(
            ["Verified", "Source Verified", "Not Verified"], n_samples
        ),
        "loan_status": np.random.choice(["Fully Paid", "Charged Off"], n_samples, p=[0.8, 0.2]),
        "purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement"], n_samples
        ),
        "title": np.random.choice(["Debt consolidation", None], n_samples),
        "dti": np.random.uniform(0, 40, n_samples),
        "earliest_cr_line": pd.to_datetime("2000-01-01") + pd.to_timedelta(
            np.random.randint(0, 7000, n_samples), unit="D"
        ),
        "open_acc": np.random.randint(1, 30, n_samples).astype(float),
        "pub_rec": np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]).astype(float),
        "revol_bal": np.random.uniform(0, 50000, n_samples),
        "revol_util": np.random.uniform(0, 100, n_samples),
        "total_acc": np.random.randint(5, 50, n_samples).astype(float),
        "initial_list_status": np.random.choice(["f", "w"], n_samples),
        "application_type": np.random.choice(["Individual", "Joint App"], n_samples, p=[0.95, 0.05]),
        "mort_acc": np.random.choice([0, 1, 2, 3, np.nan], n_samples),
        "pub_rec_bankruptcies": np.random.choice([0, 1, np.nan], n_samples, p=[0.9, 0.05, 0.05]),
        "address": np.random.choice(["123 Main St", "456 Oak Ave"], n_samples),
        "issue_d": pd.to_datetime("2020-01-01") + pd.to_timedelta(
            np.random.randint(0, 365, n_samples), unit="D"
        ),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data(sample_raw_data):
    """Create sample processed data for testing."""
    from src.data.data_transformer import DataTransformer
    
    transformer = DataTransformer()
    transformer.fit(sample_raw_data)
    return transformer.transform(sample_raw_data)


@pytest.fixture
def sample_features_and_target(sample_processed_data):
    """Create sample features and target for testing."""
    X = sample_processed_data.drop(columns=["loan_status"])
    y = sample_processed_data["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
    return X, y


@pytest.fixture
def trained_lightgbm_model(sample_features_and_target):
    """Create a trained LightGBM model for testing."""
    from src.models.lgbm_model import LightGBMModel
    
    X, y = sample_features_and_target
    
    model = LightGBMModel(params={
        "n_estimators": 10,
        "max_depth": 3,
    })
    model.fit(X, y)
    
    return model


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path
