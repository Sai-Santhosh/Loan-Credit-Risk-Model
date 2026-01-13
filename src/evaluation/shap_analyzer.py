"""
SHAP Analyzer Module - Model interpretability with SHAP values.

Provides:
- SHAP value calculation
- Summary plots
- Dependence plots
- Force plots
- Feature importance analysis
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based model interpretability analyzer.
    
    Features:
    - TreeExplainer for tree-based models
    - Multiple visualization options
    - Feature importance extraction
    - Individual prediction explanations
    
    Example:
        >>> analyzer = SHAPAnalyzer(model)
        >>> analyzer.fit(X_train)
        >>> analyzer.plot_summary(X_test)
        >>> analyzer.explain_prediction(X_test.iloc[0])
    """
    
    def __init__(
        self,
        model: BaseModel,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize SHAPAnalyzer.
        
        Args:
            model: Trained model
            feature_names: Feature names (inferred from model if None)
        """
        self.model = model
        self.feature_names = feature_names or model.feature_names
        
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        # Import SHAP
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
    
    def fit(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = None,
    ) -> "SHAPAnalyzer":
        """
        Create SHAP explainer and calculate values.
        
        Args:
            X: Data for SHAP calculation
            sample_size: Sample size for large datasets
        
        Returns:
            Self
        """
        logger.info(f"Creating SHAP explainer for {len(X)} samples")
        
        # Sample if needed
        if sample_size and len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} samples for SHAP analysis")
        
        # Prepare data
        X_prepared = self._prepare_data(X)
        
        # Create explainer
        self.explainer = self.shap.TreeExplainer(self.model.model)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_prepared)
        self.expected_value = self.explainer.expected_value
        
        # Store data for plotting
        self._X_display = X_prepared
        
        logger.info("SHAP explainer fitted successfully")
        
        return self
    
    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for SHAP analysis."""
        X = X.copy()
        
        # Convert categoricals for LightGBM
        for col in X.columns:
            if X[col].dtype.name == "category":
                X[col] = X[col].cat.codes
        
        return X
    
    def get_shap_values(
        self,
        class_idx: int = 1
    ) -> np.ndarray:
        """
        Get SHAP values for specified class.
        
        Args:
            class_idx: Class index (1 for positive class in binary)
        
        Returns:
            SHAP values array
        """
        if self.shap_values is None:
            raise RuntimeError("Must call fit() before getting SHAP values")
        
        # Handle different SHAP output formats
        if isinstance(self.shap_values, list):
            return self.shap_values[class_idx]
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get mean absolute SHAP values as feature importance.
        
        Returns:
            DataFrame with feature importances
        """
        shap_vals = self.get_shap_values()
        
        importance = np.abs(shap_vals).mean(axis=0)
        
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
    
    def plot_summary(
        self,
        X: Optional[pd.DataFrame] = None,
        plot_type: str = "dot",
        max_display: int = 20,
        title: str = "SHAP Summary Plot",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data for plotting (uses fitted data if None)
            plot_type: Plot type ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        shap_vals = self.get_shap_values()
        X_display = X if X is not None else self._X_display
        X_display = self._prepare_data(X_display) if X is not None else X_display
        
        fig = plt.figure(figsize=figsize)
        
        self.shap.summary_plot(
            shap_vals,
            X_display,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        return fig
    
    def plot_bar(
        self,
        max_display: int = 20,
        title: str = "SHAP Feature Importance",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create SHAP bar plot (mean |SHAP|).
        
        Args:
            max_display: Maximum features to display
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        return self.plot_summary(
            plot_type="bar",
            max_display=max_display,
            title=title,
            figsize=figsize,
            save_path=save_path,
        )
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = "auto",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create SHAP dependence plot.
        
        Args:
            feature: Feature to plot
            interaction_feature: Interaction feature ('auto' for automatic)
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        shap_vals = self.get_shap_values()
        
        fig = plt.figure(figsize=figsize)
        
        self.shap.dependence_plot(
            feature,
            shap_vals,
            self._X_display,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP dependence plot saved to {save_path}")
        
        return fig
    
    def plot_force(
        self,
        idx: int,
        figsize: Tuple[int, int] = (20, 3),
        save_path: Optional[str] = None,
    ):
        """
        Create SHAP force plot for single prediction.
        
        Args:
            idx: Sample index
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            SHAP force plot
        """
        shap_vals = self.get_shap_values()
        
        # Get expected value
        expected_val = (
            self.expected_value[1] 
            if isinstance(self.expected_value, list) 
            else self.expected_value
        )
        
        force_plot = self.shap.force_plot(
            expected_val,
            shap_vals[idx],
            self._X_display.iloc[idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP force plot saved to {save_path}")
        
        return force_plot
    
    def plot_waterfall(
        self,
        idx: int,
        max_display: int = 15,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create SHAP waterfall plot for single prediction.
        
        Args:
            idx: Sample index
            max_display: Maximum features to display
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        shap_vals = self.get_shap_values()
        
        # Get expected value
        expected_val = (
            self.expected_value[1] 
            if isinstance(self.expected_value, list) 
            else self.expected_value
        )
        
        fig = plt.figure(figsize=figsize)
        
        # Create Explanation object
        explanation = self.shap.Explanation(
            values=shap_vals[idx],
            base_values=expected_val,
            data=self._X_display.iloc[idx].values,
            feature_names=self.feature_names,
        )
        
        self.shap.plots.waterfall(explanation, max_display=max_display, show=False)
        
        if title:
            plt.title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        return fig
    
    def explain_prediction(
        self,
        X_single: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single sample features
            threshold: Classification threshold
        
        Returns:
            Dictionary with explanation
        """
        # Prepare data
        X_df = pd.DataFrame([X_single])
        X_prepared = self._prepare_data(X_df)
        
        # Get prediction
        proba = self.model.predict_proba(X_df)[0, 1]
        prediction = int(proba >= threshold)
        
        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(X_prepared)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        shap_vals = shap_vals[0]
        
        # Get expected value
        expected_val = (
            self.expected_value[1] 
            if isinstance(self.expected_value, list) 
            else self.expected_value
        )
        
        # Create feature contributions
        contributions = dict(zip(self.feature_names, shap_vals))
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return {
            "prediction": prediction,
            "probability": proba,
            "base_value": expected_val,
            "contributions": sorted_contributions,
            "top_positive": {
                k: v for k, v in list(sorted_contributions.items())[:5] if v > 0
            },
            "top_negative": {
                k: v for k, v in list(sorted_contributions.items())[:5] if v < 0
            },
        }
    
    def save_shap_values(self, path: str) -> None:
        """Save SHAP values to disk."""
        import joblib
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "shap_values": self.shap_values,
            "expected_value": self.expected_value,
            "feature_names": self.feature_names,
        }, path)
        
        logger.info(f"SHAP values saved to {path}")
    
    @classmethod
    def load_shap_values(cls, path: str, model: BaseModel) -> "SHAPAnalyzer":
        """Load SHAP values from disk."""
        import joblib
        
        data = joblib.load(path)
        
        analyzer = cls(model, feature_names=data["feature_names"])
        analyzer.shap_values = data["shap_values"]
        analyzer.expected_value = data["expected_value"]
        
        logger.info(f"SHAP values loaded from {path}")
        
        return analyzer


__all__ = ["SHAPAnalyzer"]
