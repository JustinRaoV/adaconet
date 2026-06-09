"""Computational performance and metric comparison plots."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# Distinct colours and markers for methods
_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def _should_use_log_scale(values: np.ndarray) -> bool:
    """Return True if the range of values spans more than 100x."""
    positive = values[values > 0]
    if len(positive) == 0:
        return False
    return (positive.max() / positive.min()) > 100


def plot_scaling_results(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot wall-time and memory scaling with respect to dataset dimensions.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ``method``, ``n_samples``, ``n_taxa``,
        ``wall_time_sec``, ``peak_memory_mb``.  Multiple repeats per
        configuration are expected; mean ± std will be plotted.
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = results_df["method"].unique().tolist()

    # Determine a representative fixed dimension for each axis
    taxa_values = sorted(results_df["n_taxa"].unique())
    sample_values = sorted(results_df["n_samples"].unique())
    fixed_taxa = taxa_values[len(taxa_values) // 2] if taxa_values else None
    fixed_samples = sample_values[len(sample_values) // 2] if sample_values else None

    # Collect all time and memory values for log-scale decision
    all_times = results_df["wall_time_sec"].values
    all_mem = results_df["peak_memory_mb"].values

    use_log_time = _should_use_log_scale(all_times)
    use_log_mem = _should_use_log_scale(all_mem)

    panels = [
        (axes[0, 0], "n_samples", fixed_taxa, "n_taxa", "Wall Time (s)", "wall_time_sec", use_log_time),
        (axes[0, 1], "n_taxa", fixed_samples, "n_samples", "Wall Time (s)", "wall_time_sec", use_log_time),
        (axes[1, 0], "n_samples", fixed_taxa, "n_taxa", "Peak Memory (MB)", "peak_memory_mb", use_log_mem),
        (axes[1, 1], "n_taxa", fixed_samples, "n_samples", "Peak Memory (MB)", "peak_memory_mb", use_log_mem),
    ]

    for ax, x_col, fixed_val, fixed_col, ylabel, y_col, log_y in panels:
        for m_idx, method in enumerate(methods):
            color = _COLORS[m_idx % len(_COLORS)]
            marker = _MARKERS[m_idx % len(_MARKERS)]

            if fixed_val is not None:
                subset = results_df[
                    (results_df["method"] == method)
                    & (results_df[fixed_col] == fixed_val)
                ]
            else:
                subset = results_df[results_df["method"] == method]

            if subset.empty:
                continue

            grouped = subset.groupby(x_col)[y_col]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            x_vals = mean.index.values.astype(float)
            y_mean = mean.values
            y_std = std.values

            ax.errorbar(
                x_vals, y_mean, yerr=y_std,
                color=color, marker=marker, markersize=6,
                linewidth=2, capsize=3, elinewidth=1.2,
                label=method,
            )

        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        fixed_label = f" ({fixed_col}={fixed_val})" if fixed_val is not None else ""
        ax.set_title(f"{ylabel} vs {x_col.replace('_', ' ')}{fixed_label}", fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = "f1",
    save_path: Optional[str] = None,
) -> Figure:
    """Grouped bar chart comparing methods on a chosen metric across dataset sizes.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ``method``, ``n_samples``, ``n_taxa``, and the
        requested *metric* column.
    metric : str
        Name of the metric column to plot (e.g. ``'f1'``, ``'auprc'``, ``'auroc'``).
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    # Build configuration labels
    df = results_df.copy()
    df["config"] = "N=" + df["n_samples"].astype(str) + ",P=" + df["n_taxa"].astype(str)

    configs = df["config"].unique().tolist()
    methods = df["method"].unique().tolist()

    # Aggregate: mean across repeats
    pivot = df.groupby(["config", "method"])[metric].mean().unstack(fill_value=0)

    # Ensure consistent ordering
    configs = [c for c in configs if c in pivot.index]
    methods = [m for m in methods if m in pivot.columns]

    n_configs = len(configs)
    n_methods = len(methods)
    bar_width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 2.5), 6))
    x = np.arange(n_configs)

    for m_idx, method in enumerate(methods):
        offset = (m_idx - n_methods / 2 + 0.5) * bar_width
        vals = [pivot.loc[c, method] if method in pivot.columns and c in pivot.index else 0 for c in configs]
        color = _COLORS[m_idx % len(_COLORS)]
        ax.bar(x + offset, vals, width=bar_width, color=color, label=method, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Dataset Configuration", fontsize=12)
    ax.set_ylabel(metric.upper() if metric.islower() else metric, fontsize=12)
    ax.set_title(f"Method Comparison — {metric.upper() if metric.islower() else metric}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig
