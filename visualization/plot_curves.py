"""PR/ROC curve comparison plots for network inference methods."""
from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


# Distinct colors and line styles for up to 8 methods
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def _flatten_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle values (excluding diagonal) from a square matrix."""
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def plot_pr_roc_curves(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot Precision-Recall and ROC curves for multiple network inference methods.

    Parameters
    ----------
    results_dict : dict
        Mapping of method name to a dict containing:
        - ``true_adj``: binary ground-truth adjacency matrix (P x P)
        - ``pred_scores``: continuous predicted score matrix (P x P)
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_pr = axes[0]
    ax_roc = axes[1]

    for i, (method_name, data) in enumerate(results_dict.items()):
        true_adj: np.ndarray = data["true_adj"]
        pred_scores: np.ndarray = data["pred_scores"]

        # Flatten upper-triangle entries to get per-edge labels and scores
        y_true = _flatten_upper_triangle(true_adj).astype(int)
        y_scores = _flatten_upper_triangle(pred_scores)

        color = _COLORS[i % len(_COLORS)]
        style = _STYLES[i % len(_STYLES)]

        # --- Precision-Recall curve ---
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(pr_recall, pr_precision)
        ax_pr.plot(
            pr_recall,
            pr_precision,
            color=color,
            linestyle=style,
            linewidth=2,
            label=f"{method_name} (AUPRC={pr_auc:.3f})",
        )

        # --- ROC curve ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(
            fpr,
            tpr,
            color=color,
            linestyle=style,
            linewidth=2,
            label=f"{method_name} (AUROC={roc_auc:.3f})",
        )

    # --- PR panel formatting ---
    ax_pr.set_xlabel("Recall", fontsize=12)
    ax_pr.set_ylabel("Precision", fontsize=12)
    ax_pr.set_title("Precision-Recall Curve", fontsize=14)
    ax_pr.tick_params(axis="both", labelsize=10)
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(fontsize=9, loc="lower left")

    # --- ROC panel formatting ---
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.set_title("ROC Curve", fontsize=14)
    ax_roc.tick_params(axis="both", labelsize=10)
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(fontsize=9, loc="lower right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig
