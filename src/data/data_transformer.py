"""
Data Transformer Module - Production-grade data preprocessing.

Provides modular, reproducible data transformation pipeline with:
- Missing value imputation
- Feature type conversion
- Encoding strategies
- Scaling and normalization
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransformationConfig:
    """Configuration for data transformations."""
    
    # Columns to drop
    drop_columns: List[str] = field(default_factory=lambda: [
        "Unnamed: 0", "emp_title", "title", "address", 
        "issue_d", "sub_grade", "application_type", "revol_bal"
    ])
    
    # Target column
    target_column: str = "loan_status"
    
    # Categorical columns
    categorical_columns: List[str] = field(default_factory=lambda: [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "pub_rec",
        "initial_list_status", "pub_rec_bankruptcies"
    ])
    
    # Columns to impute with median
    median_impute_columns: List[str] = field(default_factory=lambda: [
        "mort_acc", "revol_util", "pub_rec_bankruptcies"
    ])
    
    # Employment length mapping
    emp_length_mapping: Dict[str, int] = field(default_factory=lambda: {
        "10+ years": 10,
        "< 1 year": 0,
    })


class BaseTransformer(ABC):
    """Abstract base class for transformers."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Fit transformer to data."""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        pass
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(df).transform(df)


class DataTransformer:
    """
    Production-grade data transformer for Credit Risk data.
    
    Implements the full preprocessing pipeline including:
    - Column dropping
    - Missing value imputation
    - Feature engineering
    - Categorical encoding
    - Scaling
    
    Example:
        >>> transformer = DataTransformer()
        >>> transformer.fit(train_df)
        >>> train_processed = transformer.transform(train_df)
        >>> test_processed = transformer.transform(test_df)
        >>> transformer.save("artifacts/transformer.joblib")
    """
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        """
        Initialize DataTransformer.
        
        Args:
            config: Transformation configuration
        """
        self.config = config or TransformationConfig()
        self.is_fitted = False
        
        # Fitted parameters
        self.mort_acc_medians: Optional[pd.Series] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_statistics: Dict[str, Any] = {}
        
    def fit(self, df: pd.DataFrame) -> "DataTransformer":
        """
        Fit transformer to training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            Fitted transformer
        """
        logger.info("Fitting DataTransformer...")
        
        # Store mort_acc medians for imputation
        if "mort_acc" in df.columns and "total_acc" in df.columns:
            self.mort_acc_medians = df.groupby("total_acc")["mort_acc"].median()
            self.feature_statistics["mort_acc_global_median"] = df["mort_acc"].median()
        
        # Store medians for other columns
        for col in self.config.median_impute_columns:
            if col in df.columns:
                self.feature_statistics[f"{col}_median"] = df[col].median()
        
        # Fit label encoders for categorical columns
        df_copy = self._apply_initial_transforms(df.copy())
        
        for col in self.config.categorical_columns:
            if col in df_copy.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle unknown categories by fitting on string values
                self.label_encoders[col].fit(df_copy[col].astype(str))
        
        # Store target classes
        if self.config.target_column in df_copy.columns:
            self.label_encoders[self.config.target_column] = LabelEncoder()
            self.label_encoders[self.config.target_column].fit(
                df_copy[self.config.target_column].astype(str)
            )
        
        self.is_fitted = True
        logger.info("DataTransformer fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            df: DataFrame to transform
            include_target: Whether to include target column
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")
        
        logger.info(f"Transforming DataFrame with {len(df)} rows")
        
        df = df.copy()
        
        # Step 1: Drop unwanted columns
        df = self._drop_columns(df)
        
        # Step 2: Handle missing values
        df = self._impute_missing_values(df)
        
        # Step 3: Apply initial transforms (emp_length, pub_rec, etc.)
        df = self._apply_initial_transforms(df)
        
        # Step 4: Feature engineering
        df = self._engineer_features(df)
        
        # Step 5: Drop remaining nulls
        df = self._drop_remaining_nulls(df)
        
        # Step 6: Select final features
        df = self._select_features(df, include_target)
        
        logger.info(f"Transformation complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unwanted columns."""
        cols_to_drop = [c for c in self.config.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.debug(f"Dropped columns: {cols_to_drop}")
        return df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values."""
        
        # Impute mort_acc using total_acc correlation
        if "mort_acc" in df.columns and self.mort_acc_medians is not None:
            df["mort_acc"] = df.apply(self._fill_mort_acc, axis=1)
        
        # Impute other columns with median
        for col in self.config.median_impute_columns:
            if col in df.columns and col != "mort_acc":
                median_key = f"{col}_median"
                if median_key in self.feature_statistics:
                    df[col] = df[col].fillna(self.feature_statistics[median_key])
        
        return df
    
    def _fill_mort_acc(self, row: pd.Series) -> float:
        """Fill mort_acc using total_acc median lookup."""
        if pd.isna(row["mort_acc"]):
            return self.mort_acc_medians.get(
                row["total_acc"],
                self.feature_statistics.get("mort_acc_global_median", 0)
            )
        return row["mort_acc"]
    
    def _apply_initial_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply initial feature transformations."""
        
        # Convert employment length
        if "emp_length" in df.columns:
            df["emp_length"] = df["emp_length"].apply(self._convert_emp_length)
        
        # Bin pub_rec
        if "pub_rec" in df.columns:
            df["pub_rec"] = df["pub_rec"].apply(
                lambda x: "0" if x == 0 else ("1" if x == 1 else "2+")
            )
        
        # Bin pub_rec_bankruptcies
        if "pub_rec_bankruptcies" in df.columns:
            df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].apply(
                lambda x: "0" if pd.isna(x) or x == 0 else ("1" if x == 1 else "2+")
            )
        
        # Convert earliest_cr_line to datetime
        if "earliest_cr_line" in df.columns:
            df["earliest_cr_line"] = pd.to_datetime(
                df["earliest_cr_line"], errors="coerce"
            )
        
        return df
    
    def _convert_emp_length(self, emp: Any) -> str:
        """Convert employment length to standardized format."""
        if pd.isnull(emp):
            return "Unknown"
        
        emp = str(emp).strip()
        
        if emp == "10+ years":
            return "10"
        elif emp == "< 1 year":
            return "0"
        else:
            try:
                return str(int(emp.split()[0]))
            except (ValueError, IndexError):
                return "Unknown"
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        
        # Filter zero income
        if "annual_inc" in df.columns:
            df = df[df["annual_inc"] > 0].copy()
        
        # Loan to income ratio
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
        
        # Total interest owed
        if "loan_amnt" in df.columns and "int_rate" in df.columns:
            df["total_interest_owed"] = df["loan_amnt"] * (df["int_rate"] / 100)
        
        # Installment to income ratio
        if "installment" in df.columns and "annual_inc" in df.columns:
            df["installment_to_income_ratio"] = df["installment"] / (df["annual_inc"] / 12)
        
        # Active credit percentage
        if "open_acc" in df.columns and "total_acc" in df.columns:
            df["active_credit_pct"] = df.apply(
                lambda row: row["open_acc"] / row["total_acc"] 
                if row["total_acc"] > 0 else np.nan,
                axis=1
            )
        
        # Credit age
        if "earliest_cr_line" in df.columns:
            current_year = pd.Timestamp.now().year
            df["credit_age"] = current_year - df["earliest_cr_line"].dt.year
            df = df.drop(columns=["earliest_cr_line"])
        
        return df
    
    def _drop_remaining_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with remaining null values."""
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with null values")
        
        return df
    
    def _select_features(
        self, df: pd.DataFrame, include_target: bool
    ) -> pd.DataFrame:
        """Select final features for modeling."""
        
        feature_columns = [
            "term", "int_rate", "grade", "emp_length",
            "home_ownership", "annual_inc", "verification_status",
            "purpose", "dti", "pub_rec", "revol_util",
            "initial_list_status", "mort_acc", "pub_rec_bankruptcies",
            "loan_to_income", "total_interest_owed",
            "installment_to_income_ratio", "active_credit_pct", "credit_age"
        ]
        
        if include_target and self.config.target_column in df.columns:
            feature_columns.append(self.config.target_column)
        
        # Only select columns that exist
        available_columns = [c for c in feature_columns if c in df.columns]
        
        return df[available_columns]
    
    def encode_categoricals(
        self, 
        df: pd.DataFrame, 
        method: str = "label"
    ) -> pd.DataFrame:
        """
        Encode categorical columns.
        
        Args:
            df: DataFrame to encode
            method: Encoding method ('label' or 'category')
        
        Returns:
            Encoded DataFrame
        """
        df = df.copy()
        
        for col in self.config.categorical_columns:
            if col not in df.columns:
                continue
            
            if method == "label":
                if col in self.label_encoders:
                    # Handle unknown categories
                    df[col] = df[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else "Unknown"
                    )
                    # Refit if Unknown not in classes
                    if "Unknown" not in known_classes:
                        self.label_encoders[col].fit(
                            list(known_classes) + ["Unknown"]
                        )
                    df[col] = self.label_encoders[col].transform(df[col])
            elif method == "category":
                df[col] = df[col].astype("category")
        
        return df
    
    def encode_target(self, y: pd.Series) -> pd.Series:
        """Encode target variable."""
        if self.config.target_column in self.label_encoders:
            return pd.Series(
                self.label_encoders[self.config.target_column].transform(
                    y.astype(str)
                ),
                index=y.index
            )
        return y
    
    def save(self, path: str) -> None:
        """
        Save fitted transformer to disk.
        
        Args:
            path: Output path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config,
            "is_fitted": self.is_fitted,
            "mort_acc_medians": self.mort_acc_medians,
            "label_encoders": self.label_encoders,
            "scalers": self.scalers,
            "feature_statistics": self.feature_statistics,
        }
        
        joblib.dump(state, path)
        logger.info(f"Transformer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DataTransformer":
        """
        Load fitted transformer from disk.
        
        Args:
            path: Path to saved transformer
        
        Returns:
            Loaded transformer
        """
        state = joblib.load(path)
        
        transformer = cls(config=state["config"])
        transformer.is_fitted = state["is_fitted"]
        transformer.mort_acc_medians = state["mort_acc_medians"]
        transformer.label_encoders = state["label_encoders"]
        transformer.scalers = state["scalers"]
        transformer.feature_statistics = state["feature_statistics"]
        
        logger.info(f"Transformer loaded from {path}")
        return transformer


__all__ = ["DataTransformer", "TransformationConfig"]
