"""
Evaluator Module - Comprehensive model evaluation.

Provides:
- Multiple classification metrics
- Threshold optimization
- ROC/PR curve analysis
- Classification reports
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pr_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    optimal_threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "optimal_threshold": self.optimal_threshold,
        }


class Evaluator:
    """
    Production model evaluator for credit risk models.
    
    Features:
    - Comprehensive metric calculation
    - Threshold optimization
    - ROC/PR curve analysis
    - Visualization utilities
    
    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate(model, X_test, y_test)
        >>> print(result.metrics)
        >>> evaluator.plot_roc_curve(result)
    """
    
    def __init__(
        self,
        default_threshold: float = 0.5,
        pos_label: int = 1,
    ):
        """
        Initialize Evaluator.
        
        Args:
            default_threshold: Default classification threshold
            pos_label: Positive class label
        """
        self.default_threshold = default_threshold
        self.pos_label = pos_label
    
    def evaluate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None,
        optimize_threshold: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            threshold: Classification threshold
            optimize_threshold: Find optimal threshold
        
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating model on {len(X)} samples")
        
        # Get predictions
        y_proba = model.predict_proba(X)[:, 1]
        
        # Optimize threshold if requested
        if optimize_threshold:
            threshold = self._find_optimal_threshold(y, y_proba)
        else:
            threshold = threshold or self.default_threshold
        
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Get curves
        fpr, tpr, roc_thresholds = roc_curve(y, y_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y, y_proba)
        
        # Create result
        result = EvaluationResult(
            metrics=metrics,
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred),
            roc_curve=(fpr, tpr, roc_thresholds),
            pr_curve=(precision, recall, pr_thresholds),
            optimal_threshold=threshold,
        )
        
        logger.info(f"Evaluation complete. AUC: {metrics['auc']:.4f}")
        
        return result
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "log_loss": log_loss(y_true, y_proba),
            "specificity": self._specificity(y_true, y_pred),
            "balanced_accuracy": self._balanced_accuracy(y_true, y_pred),
        }
    
    def _specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _balanced_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._specificity(y_true, y_pred)
        return (recall + specificity) / 2
    
    def _find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        metric: str = "f1"
    ) -> float:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'youden', 'balanced')
        
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "youden":
                # Youden's J statistic
                recall = recall_score(y_true, y_pred, zero_division=0)
                specificity = self._specificity(y_true, y_pred)
                score = recall + specificity - 1
            elif metric == "balanced":
                score = self._balanced_accuracy(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} ({metric}: {best_score:.4f})")
        
        return best_threshold
    
    def plot_roc_curve(
        self,
        result: EvaluationResult,
        title: str = "ROC Curve",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            result: Evaluation result
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = result.roc_curve
        auc = result.metrics["auc"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        result: EvaluationResult,
        title: str = "Precision-Recall Curve",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            result: Evaluation result
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = result.pr_curve
        pr_auc = result.metrics["pr_auc"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, color="blue", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
        
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        result: EvaluationResult,
        labels: List[str] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            result: Evaluation result
            labels: Class labels
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        import seaborn as sns
        
        labels = labels or ["Charged Off", "Fully Paid"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            result.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def compare_models(
        self,
        models: Dict[str, BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model name to model
            X: Test features
            y: True labels
        
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for name, model in models.items():
            eval_result = self.evaluate(model, X, y)
            results.append({
                "model": name,
                **eval_result.metrics,
                "optimal_threshold": eval_result.optimal_threshold,
            })
        
        return pd.DataFrame(results).sort_values("auc", ascending=False)
    
    def get_metrics_summary(self, result: EvaluationResult) -> str:
        """Get formatted metrics summary."""
        summary = "=" * 50 + "\n"
        summary += "MODEL EVALUATION SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        for metric, value in result.metrics.items():
            summary += f"{metric:20s}: {value:.4f}\n"
        
        summary += f"\n{'Optimal Threshold':20s}: {result.optimal_threshold:.3f}\n"
        summary += "\n" + "=" * 50 + "\n"
        summary += "CLASSIFICATION REPORT\n"
        summary += "=" * 50 + "\n"
        summary += result.classification_report
        
        return summary


__all__ = ["Evaluator", "EvaluationResult"]
