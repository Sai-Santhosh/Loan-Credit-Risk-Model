"""
Feature Engineering Module - Production-grade feature creation.

Provides modular feature engineering with versioning and reproducibility.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of an engineered feature."""
    name: str
    description: str
    formula: Callable[[pd.DataFrame], pd.Series]
    dependencies: List[str]
    version: str = "1.0.0"


class FeatureEngineer:
    """
    Production-grade feature engineering system.
    
    Provides:
    - Modular feature definitions
    - Automatic dependency resolution
    - Feature importance analysis
    - Feature versioning
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> df = engineer.create_features(raw_df)
        >>> importance = engineer.get_feature_importance(df, y)
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with default features."""
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self._register_default_features()
    
    def _register_default_features(self) -> None:
        """Register default credit risk features."""
        
        # Loan to income ratio
        self.register_feature(FeatureDefinition(
            name="loan_to_income",
            description="Ratio of loan amount to annual income",
            formula=lambda df: df["loan_amnt"] / df["annual_inc"],
            dependencies=["loan_amnt", "annual_inc"],
            version="1.0.0"
        ))
        
        # Total interest owed
        self.register_feature(FeatureDefinition(
            name="total_interest_owed",
            description="Total interest based on loan amount and rate",
            formula=lambda df: df["loan_amnt"] * (df["int_rate"] / 100),
            dependencies=["loan_amnt", "int_rate"],
            version="1.0.0"
        ))
        
        # Installment to income ratio
        self.register_feature(FeatureDefinition(
            name="installment_to_income_ratio",
            description="Monthly installment as ratio of monthly income",
            formula=lambda df: df["installment"] / (df["annual_inc"] / 12),
            dependencies=["installment", "annual_inc"],
            version="1.0.0"
        ))
        
        # Active credit percentage
        self.register_feature(FeatureDefinition(
            name="active_credit_pct",
            description="Percentage of open accounts to total accounts",
            formula=lambda df: df["open_acc"] / df["total_acc"].replace(0, np.nan),
            dependencies=["open_acc", "total_acc"],
            version="1.0.0"
        ))
        
        # Credit age
        self.register_feature(FeatureDefinition(
            name="credit_age",
            description="Years since earliest credit line",
            formula=lambda df: pd.Timestamp.now().year - pd.to_datetime(
                df["earliest_cr_line"], errors="coerce"
            ).dt.year,
            dependencies=["earliest_cr_line"],
            version="1.0.0"
        ))
        
        # DTI bucket
        self.register_feature(FeatureDefinition(
            name="dti_bucket",
            description="DTI categorized into risk buckets",
            formula=lambda df: pd.cut(
                df["dti"],
                bins=[0, 10, 20, 30, 40, 100],
                labels=["very_low", "low", "medium", "high", "very_high"]
            ),
            dependencies=["dti"],
            version="1.0.0"
        ))
        
        # Income bucket
        self.register_feature(FeatureDefinition(
            name="income_bucket",
            description="Annual income categorized into brackets",
            formula=lambda df: pd.cut(
                df["annual_inc"],
                bins=[0, 30000, 50000, 75000, 100000, 150000, np.inf],
                labels=["very_low", "low", "medium", "high", "very_high", "top"]
            ),
            dependencies=["annual_inc"],
            version="1.0.0"
        ))
        
        # Debt burden score
        self.register_feature(FeatureDefinition(
            name="debt_burden_score",
            description="Combined debt burden indicator",
            formula=lambda df: (
                df["dti"] * 0.3 + 
                df["revol_util"].fillna(0) * 0.3 + 
                (df["loan_amnt"] / df["annual_inc"]) * 100 * 0.4
            ),
            dependencies=["dti", "revol_util", "loan_amnt", "annual_inc"],
            version="1.0.0"
        ))
        
        # Payment capacity
        self.register_feature(FeatureDefinition(
            name="payment_capacity",
            description="Estimated payment capacity score",
            formula=lambda df: (
                df["annual_inc"] / 12 - df["installment"]
            ) / (df["annual_inc"] / 12 + 1),
            dependencies=["annual_inc", "installment"],
            version="1.0.0"
        ))
    
    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """
        Register a new feature definition.
        
        Args:
            feature_def: Feature definition object
        """
        self.feature_definitions[feature_def.name] = feature_def
        logger.debug(f"Registered feature: {feature_def.name}")
    
    def create_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Input DataFrame
            features: List of features to create (None for all)
            inplace: Modify DataFrame in place
        
        Returns:
            DataFrame with new features
        """
        if not inplace:
            df = df.copy()
        
        features_to_create = features or list(self.feature_definitions.keys())
        
        logger.info(f"Creating {len(features_to_create)} engineered features")
        
        for feature_name in features_to_create:
            if feature_name not in self.feature_definitions:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            feature_def = self.feature_definitions[feature_name]
            
            # Check dependencies
            missing_deps = [
                dep for dep in feature_def.dependencies 
                if dep not in df.columns
            ]
            
            if missing_deps:
                logger.warning(
                    f"Skipping {feature_name}: missing dependencies {missing_deps}"
                )
                continue
            
            try:
                df[feature_name] = feature_def.formula(df)
                logger.debug(f"Created feature: {feature_name}")
            except Exception as e:
                logger.error(f"Error creating {feature_name}: {e}")
        
        return df
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate feature importance scores.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Importance method ('mutual_info', 'correlation')
            top_k: Return only top k features
        
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating feature importance using {method}")
        
        if method == "mutual_info":
            # Handle categorical columns
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
            
            importance = mutual_info_classif(
                X_encoded.fillna(0),
                y,
                random_state=42
            )
            
        elif method == "correlation":
            X_numeric = X.select_dtypes(include=[np.number])
            importance = X_numeric.corrwith(y).abs().values
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create importance DataFrame
        result = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        if top_k:
            result = result.head(top_k)
        
        return result
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 15,
        method: str = "mutual_info"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Selection method
        
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        importance_df = self.get_feature_importance(X, y, method, top_k=k)
        selected_features = importance_df["feature"].tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        
        return X[selected_features], selected_features
    
    def get_feature_correlations(
        self,
        df: pd.DataFrame,
        threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Find highly correlated feature pairs.
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold
        
        Returns:
            DataFrame with correlated pairs
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        
        # Get upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold
        correlated_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if upper_tri.loc[idx, col] >= threshold:
                    correlated_pairs.append({
                        "feature_1": idx,
                        "feature_2": col,
                        "correlation": upper_tri.loc[idx, col]
                    })
        
        return pd.DataFrame(correlated_pairs).sort_values(
            "correlation", ascending=False
        )
    
    def list_features(self) -> pd.DataFrame:
        """List all registered features."""
        return pd.DataFrame([
            {
                "name": f.name,
                "description": f.description,
                "dependencies": ", ".join(f.dependencies),
                "version": f.version
            }
            for f in self.feature_definitions.values()
        ])


__all__ = ["FeatureEngineer", "FeatureDefinition"]
