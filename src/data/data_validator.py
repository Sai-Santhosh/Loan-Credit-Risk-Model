"""
Data Validator Module - Schema validation and data quality checks.

Uses Pandera for schema validation and Great Expectations patterns
for comprehensive data quality validation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        result = f"Validation Result: {status}\n"
        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"
        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"
        return result


class CreditRiskDataSchema:
    """
    Schema definition for Credit Risk dataset.
    
    Defines expected columns, data types, and constraints.
    """
    
    # Required columns with their expected types
    REQUIRED_COLUMNS = {
        "loan_amnt": "float64",
        "term": "object",
        "int_rate": "float64",
        "installment": "float64",
        "grade": "object",
        "emp_length": "object",
        "home_ownership": "object",
        "annual_inc": "float64",
        "verification_status": "object",
        "loan_status": "object",
        "purpose": "object",
        "dti": "float64",
        "open_acc": "float64",
        "pub_rec": "float64",
        "revol_bal": "float64",
        "revol_util": "float64",
        "total_acc": "float64",
        "initial_list_status": "object",
        "mort_acc": "float64",
        "pub_rec_bankruptcies": "float64",
    }
    
    # Value constraints
    CONSTRAINTS = {
        "loan_amnt": {"min": 0, "max": 1000000},
        "int_rate": {"min": 0, "max": 100},
        "annual_inc": {"min": 0, "max": 50000000},
        "dti": {"min": 0, "max": 100},
        "revol_util": {"min": 0, "max": 200},
        "open_acc": {"min": 0, "max": 100},
        "total_acc": {"min": 0, "max": 200},
    }
    
    # Categorical value sets
    CATEGORICAL_VALUES = {
        "term": {" 36 months", " 60 months"},
        "grade": {"A", "B", "C", "D", "E", "F", "G"},
        "home_ownership": {"RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"},
        "verification_status": {"Verified", "Source Verified", "Not Verified"},
        "loan_status": {"Fully Paid", "Charged Off"},
        "initial_list_status": {"f", "w"},
    }
    
    # Maximum allowed null percentage per column
    MAX_NULL_PERCENTAGE = {
        "loan_amnt": 0,
        "int_rate": 0,
        "annual_inc": 0,
        "loan_status": 0,
        "emp_length": 10,
        "mort_acc": 15,
        "revol_util": 1,
        "pub_rec_bankruptcies": 1,
    }


class DataValidator:
    """
    Production-grade data validator for Credit Risk data.
    
    Performs comprehensive validation including:
    - Schema validation (columns, types)
    - Null value checks
    - Range constraints
    - Categorical value validation
    - Statistical anomaly detection
    
    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate(df)
        >>> if not result.is_valid:
        ...     print(result.errors)
    """
    
    def __init__(
        self,
        schema: type = CreditRiskDataSchema,
        strict: bool = False,
    ):
        """
        Initialize DataValidator.
        
        Args:
            schema: Schema class with validation rules
            strict: If True, treat warnings as errors
        """
        self.schema = schema
        self.strict = strict
    
    def validate(
        self,
        df: pd.DataFrame,
        check_schema: bool = True,
        check_nulls: bool = True,
        check_ranges: bool = True,
        check_categoricals: bool = True,
        check_statistics: bool = True,
    ) -> ValidationResult:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            check_schema: Validate column names and types
            check_nulls: Check null value percentages
            check_ranges: Validate numeric ranges
            check_categoricals: Validate categorical values
            check_statistics: Check for statistical anomalies
        
        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        errors = []
        warnings = []
        statistics = {}
        
        logger.info(f"Validating DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        # Schema validation
        if check_schema:
            schema_errors, schema_warnings = self._validate_schema(df)
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)
        
        # Null value validation
        if check_nulls:
            null_errors, null_warnings, null_stats = self._validate_nulls(df)
            errors.extend(null_errors)
            warnings.extend(null_warnings)
            statistics["null_percentages"] = null_stats
        
        # Range validation
        if check_ranges:
            range_errors, range_warnings = self._validate_ranges(df)
            errors.extend(range_errors)
            warnings.extend(range_warnings)
        
        # Categorical validation
        if check_categoricals:
            cat_errors, cat_warnings = self._validate_categoricals(df)
            errors.extend(cat_errors)
            warnings.extend(cat_warnings)
        
        # Statistical validation
        if check_statistics:
            stat_warnings, stat_stats = self._validate_statistics(df)
            warnings.extend(stat_warnings)
            statistics["summary"] = stat_stats
        
        # Determine validity
        is_valid = len(errors) == 0
        if self.strict:
            is_valid = is_valid and len(warnings) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
        )
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed with {len(errors)} errors")
        
        return result
    
    def _validate_schema(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Validate DataFrame schema against expected columns and types."""
        errors = []
        warnings = []
        
        # Check for missing required columns
        missing_cols = set(self.schema.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for extra columns
        extra_cols = set(df.columns) - set(self.schema.REQUIRED_COLUMNS.keys())
        if extra_cols:
            warnings.append(f"Extra columns found: {extra_cols}")
        
        # Check data types for existing columns
        for col, expected_type in self.schema.REQUIRED_COLUMNS.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type == "float64" and actual_type not in ["float64", "float32", "int64", "int32"]:
                    warnings.append(f"Column '{col}' has type {actual_type}, expected numeric")
                elif expected_type == "object" and actual_type != "object":
                    if actual_type != "category":
                        warnings.append(f"Column '{col}' has type {actual_type}, expected object/category")
        
        return errors, warnings
    
    def _validate_nulls(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate null value percentages."""
        errors = []
        warnings = []
        null_stats = {}
        
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            null_stats[col] = round(null_pct, 2)
            
            max_null = self.schema.MAX_NULL_PERCENTAGE.get(col, 20)
            
            if null_pct > max_null:
                errors.append(
                    f"Column '{col}' has {null_pct:.2f}% nulls (max allowed: {max_null}%)"
                )
            elif null_pct > max_null * 0.8:
                warnings.append(
                    f"Column '{col}' has {null_pct:.2f}% nulls, approaching limit"
                )
        
        return errors, warnings, null_stats
    
    def _validate_ranges(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Validate numeric column ranges."""
        errors = []
        warnings = []
        
        for col, constraints in self.schema.CONSTRAINTS.items():
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            col_data = df[col].dropna()
            
            if "min" in constraints:
                below_min = (col_data < constraints["min"]).sum()
                if below_min > 0:
                    errors.append(
                        f"Column '{col}' has {below_min} values below minimum {constraints['min']}"
                    )
            
            if "max" in constraints:
                above_max = (col_data > constraints["max"]).sum()
                if above_max > 0:
                    warnings.append(
                        f"Column '{col}' has {above_max} values above maximum {constraints['max']}"
                    )
        
        return errors, warnings
    
    def _validate_categoricals(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Validate categorical column values."""
        errors = []
        warnings = []
        
        for col, valid_values in self.schema.CATEGORICAL_VALUES.items():
            if col not in df.columns:
                continue
            
            unique_values = set(df[col].dropna().unique())
            invalid_values = unique_values - valid_values
            
            if invalid_values:
                warnings.append(
                    f"Column '{col}' has unexpected values: {invalid_values}"
                )
        
        return errors, warnings
    
    def _validate_statistics(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Check for statistical anomalies."""
        warnings = []
        stats = {}
        
        # Calculate basic statistics
        stats["row_count"] = len(df)
        stats["column_count"] = len(df.columns)
        stats["memory_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100
        stats["duplicate_rows"] = dup_count
        
        if dup_pct > 5:
            warnings.append(f"High duplicate rate: {dup_pct:.2f}%")
        
        # Check target distribution if present
        if "loan_status" in df.columns:
            target_dist = df["loan_status"].value_counts(normalize=True)
            stats["target_distribution"] = target_dist.to_dict()
            
            # Check for severe imbalance
            min_class_pct = target_dist.min() * 100
            if min_class_pct < 5:
                warnings.append(
                    f"Severe class imbalance: minority class is {min_class_pct:.2f}%"
                )
        
        return warnings, stats
    
    def get_null_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed null value report.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            DataFrame with null statistics per column
        """
        null_counts = df.isnull().sum()
        null_pcts = (null_counts / len(df)) * 100
        
        report = pd.DataFrame({
            "null_count": null_counts,
            "null_percentage": null_pcts.round(2),
            "non_null_count": len(df) - null_counts,
        })
        
        return report.sort_values("null_percentage", ascending=False)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            DataFrame with summary statistics
        """
        return df.describe(include="all").T


__all__ = ["DataValidator", "CreditRiskDataSchema", "ValidationResult"]
