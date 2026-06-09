"""Parameter sensitivity and ablation study visualisations."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def plot_sensitivity_heatmap(
    sensitivity_results: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "f1",
    save_path: Optional[str] = None,
) -> Figure:
    """Heatmap showing a metric as a function of two hyper-parameters.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        Must contain at least the columns *param_x*, *param_y*, and *metric*.
        If a ``method`` column is present, the first method is used (or filter
        before calling).
    param_x : str
        Column name for the x-axis parameter.
    param_y : str
        Column name for the y-axis parameter.
    metric : str
        Column name of the metric to display.
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    df = sensitivity_results.copy()

    # If multiple methods exist, pick the first one
    if "method" in df.columns:
        first_method = df["method"].iloc[0]
        df = df[df["method"] == first_method]

    # Pivot: rows = param_y, columns = param_x
    pivot = df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8 + 2), max(5, len(pivot) * 0.7 + 1.5)))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric.upper() if metric.islower() else metric},
    )

    ax.set_xlabel(param_x.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(param_y.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"Sensitivity: {metric.upper() if metric.islower() else metric} "
        f"vs {param_x} / {param_y}",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig


# Default ablation configurations
_ABLATION_CONFIGS = ["Full", "No-VAE", "No-MI", "No-DM", "No-Prop", "Uniform-weights"]
_RADAR_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
]
_RADAR_STYLES = ["-", "--", "-.", ":", "-", "--"]


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> Figure:
    """Radar / spider chart comparing ablation configurations across metrics.

    Parameters
    ----------
    ablation_results : dict
        Mapping of configuration name (e.g. ``'Full'``, ``'No-VAE'``) to a
        dict of ``{metric_name: value}``.  All configurations should share the
        same set of metric keys.
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Collect metric names from the first configuration
    all_metrics: list[str] = []
    for metrics in ablation_results.values():
        for key in metrics:
            if key not in all_metrics:
                all_metrics.append(key)

    n_metrics = len(all_metrics)
    if n_metrics < 3:
        # Radar charts need at least 3 axes; pad if necessary
        while len(all_metrics) < 3:
            all_metrics.append("_pad_")
        n_metrics = len(all_metrics)

    # Compute angles for each metric axis (evenly spaced around the circle)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    configs = list(ablation_results.keys())

    for c_idx, config in enumerate(configs):
        metrics = ablation_results[config]
        values = [metrics.get(m, 0.0) for m in all_metrics]
        values += values[:1]  # close polygon

        color = _RADAR_COLORS[c_idx % len(_RADAR_COLORS)]
        style = _RADAR_STYLES[c_idx % len(_RADAR_STYLES)]

        ax.plot(angles, values, linewidth=2, linestyle=style, color=color, label=config)
        ax.fill(angles, values, alpha=0.08, color=color)

    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [m.replace("_", " ").upper() if not m.startswith("_") else "" for m in all_metrics],
        fontsize=10,
    )
    ax.set_title("Ablation Study — Multi-Metric Radar", fontsize=14, pad=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig
