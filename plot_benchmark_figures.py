#!/usr/bin/env python3
"""Generate publication-quality benchmark figures (Nature style).

Produces a multi-panel figure comparing AdaCoNet vs baselines on:
  - Simulated accuracy (AUROC, AUPRC, F1)
  - Computational speed (wall time)
  - Real-data network topology

Follows nature-figure skill guidelines:
  - svg.fonttype='none' for editable text
  - pdf.fonttype=42 for TrueType
  - apply_publication_style(font_size=8) for dense multi-panel
  - NMI pastel palette for unified method families
  - SVG primary, PDF + TIFF secondary
"""
import json
import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================================
# Nature figure setup
# ============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ============================================================================
# PALETTE — NMI Pastel for unified method families
# ============================================================================
PALETTE_NMI = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny":  "#E4E4F0",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "neutral_light": "#D8D8D8",
    "neutral_mid":   "#A8A8A8",
    "neutral_dark":  "#606060",
    "delta_up":   "#2E9E44",
    "delta_down": "#E53935",
}

# Method colors: AdaCoNet is highlighted, others are baseline family
METHOD_COLORS = {
    "AdaCoNet":        "#E53935",     # strong red — hero method
    "SparCC":          "#484878",     # dark blue-purple
    "Spearman CLR":    "#7884B4",     # mid blue
    "Graphical Lasso": "#B4C0E4",     # soft blue
    "SPIEC-EASI":      "#A8A8A8",     # neutral mid
    "Proportionality": "#D8D8D8",     # neutral light
}

METHOD_ORDER = [
    "AdaCoNet", "SparCC", "Spearman CLR",
    "Graphical Lasso", "SPIEC-EASI", "Proportionality",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=10,
                    color="black", fontweight="bold"):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            color=color, ha="left", va="bottom")


def save_pub(fig, filename, dpi=600):
    base = os.path.join(FIG_DIR, filename)
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.tiff", dpi=dpi, bbox_inches="tight")
    print(f"Saved {base}.{{svg,pdf,tiff}}")


# ============================================================================
# Data loading
# ============================================================================
def load_results():
    """Load benchmark results from JSON."""
    path = os.path.join(RESULTS_DIR, "public_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================================
# Figure 1: Main benchmark figure (3 panels)
# ============================================================================
def plot_main_benchmark(sim_metrics, real_topology):
    """
    Figure contract:
      Core conclusion: AdaCoNet achieves superior accuracy and speed
        for microbial co-occurrence network inference.
      Panel a: Simulated accuracy (AUROC, AUPRC, F1) — grouped bars
      Panel b: Computational speed (log-scale wall time)
      Panel c: Real-data network topology (modularity, max CC, degree)
      Archetype: quantitative grid
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))  # ~183mm wide

    # ------------------------------------------------------------------
    # Panel a: Simulated accuracy (N=280, P=553)
    # ------------------------------------------------------------------
    ax = axes[0]
    add_panel_label(ax, "a")

    sim_df = pd.DataFrame(sim_metrics)
    metrics = ["auroc", "auprc", "f1"]
    metric_labels = ["AUROC", "AUPRC", "F1"]
    methods = [m for m in METHOD_ORDER if m in sim_df["method"].unique()]

    n_methods = len(methods)
    n_metrics = len(metrics)
    bar_w = 0.8 / n_methods
    x = np.arange(n_metrics)

    for i, method in enumerate(methods):
        m_data = sim_df[sim_df["method"] == method]
        means = [m_data[met].mean() for met in metrics]
        stds = [m_data[met].std() for met in metrics]
        offset = (i - (n_methods - 1) / 2) * bar_w
        color = METHOD_COLORS[method]
        bars = ax.bar(
            x + offset, means, width=bar_w, yerr=stds,
            color=color, edgecolor="black", linewidth=0.5,
            label=method, capsize=2, error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="#A8A8A8", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.text(n_metrics - 0.5, 0.52, "random", fontsize=6, color="#A8A8A8",
            ha="right", va="bottom")
    ax.set_title("Simulated (N=280, P=553)", fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=5.5, ncol=2,
              labelspacing=0.3, handlelength=1.2, borderpad=0.3)

    # ------------------------------------------------------------------
    # Panel b: Speed comparison
    # ------------------------------------------------------------------
    ax = axes[1]
    add_panel_label(ax, "b")

    times = []
    for method in methods:
        m_data = sim_df[sim_df["method"] == method]
        times.append(m_data["wall_time"].mean())

    colors = [METHOD_COLORS[m] for m in methods]
    bars = ax.barh(range(len(methods)), times, color=colors,
                   edgecolor="black", linewidth=0.5, height=0.6)

    # Add time labels
    for i, (bar, t) in enumerate(zip(bars, times)):
        label = f"{t:.1f}s" if t < 10 else f"{t:.0f}s"
        ax.text(max(t * 1.05, 0.5), i, label, va="center", fontsize=6)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Wall time (s)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_title("Computational cost", fontsize=8, pad=4)
    ax.invert_yaxis()

    # ------------------------------------------------------------------
    # Panel c: Real data topology
    # ------------------------------------------------------------------
    ax = axes[2]
    add_panel_label(ax, "c")

    real_df = pd.DataFrame(real_topology)
    if len(real_df) > 0:
        # Use first dataset for topology comparison
        ds_name = real_df["dataset"].iloc[0]
        ds_df = real_df[real_df["dataset"] == ds_name]
        methods_real = [m for m in METHOD_ORDER if m in ds_df["method"].unique()]

        modularity = []
        max_cc = []
        for m in methods_real:
            m_data = ds_df[ds_df["method"] == m]
            modularity.append(m_data["modularity"].values[0] if len(m_data) > 0 else 0)
            max_cc.append(m_data["max_cc"].values[0] if len(m_data) > 0 else 0)

        x = np.arange(len(methods_real))
        w = 0.35
        colors = [METHOD_COLORS[m] for m in methods_real]

        # Modularity bars — colored by method
        ax.bar(x - w/2, modularity, width=w, color=colors,
               edgecolor="black", linewidth=0.5, label="Modularity", alpha=0.9)
        ax2 = ax.twinx()
        ax2.bar(x + w/2, max_cc, width=w, color=colors,
                edgecolor="black", linewidth=0.5, label="Max CC", alpha=0.4,
                hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods_real],
                           fontsize=6, rotation=0)
        ax.set_ylabel("Modularity", fontsize=7)
        ax2.set_ylabel("Max connected\ncomponent", fontsize=7)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", labelsize=6)
        # Focus y-axis on meaningful range (glasso/spiecas produce complete graphs)
        max_mod = max(m for m in modularity if m > 0.01) if any(m > 0.01 for m in modularity) else 1.0
        ax.set_ylim(0, min(max_mod * 1.35, 0.7))
        max_cc_val = max(max_cc)
        ax2.set_ylim(0, max_cc_val * 1.3 if max_cc_val > 0 else 100)
        # Clean dataset label
        short_name = ds_name.split("(")[0].strip().replace("_", " ").title()
        ax.set_title(f"Network topology ({short_name})", fontsize=8, pad=4)

        # Custom legend: one modularity + one max-CC entry
        from matplotlib.patches import Patch
        mod_patch = Patch(facecolor="#A8A8A8", edgecolor="black", linewidth=0.5, label="Modularity")
        cc_patch = Patch(facecolor="#A8A8A8", edgecolor="black", linewidth=0.5,
                         hatch="//", alpha=0.4, label="Max CC")
        ax.legend(handles=[mod_patch, cc_patch], loc="upper right", fontsize=6,
                  ncol=1, borderpad=0.3)

    fig.tight_layout(pad=0.8)
    save_pub(fig, "benchmark_main")
    plt.close(fig)
    return fig


# ============================================================================
# Figure 2: Speed-Accuracy tradeoff scatter
# ============================================================================
def plot_speed_accuracy(sim_metrics):
    """
    Figure contract:
      Core conclusion: AdaCoNet occupies the Pareto-optimal corner
        (high accuracy, low runtime).
      Archetype: quantitative scatter
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    sim_df = pd.DataFrame(sim_metrics)

    for method in METHOD_ORDER:
        m_data = sim_df[sim_df["method"] == method]
        if len(m_data) == 0:
            continue

        auroc = m_data["auroc"].mean()
        wall = m_data["wall_time"].mean()
        color = METHOD_COLORS[method]

        marker = "o" if method == "AdaCoNet" else "s"
        size = 80 if method == "AdaCoNet" else 50
        zorder = 10 if method == "AdaCoNet" else 5
        edge = "black" if method == "AdaCoNet" else "gray"
        lw = 1.2 if method == "AdaCoNet" else 0.5

        ax.scatter(wall, auroc, c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=lw, zorder=zorder, label=method)

        # Label
        offset_x, offset_y = 0, 0.02
        if method == "SparCC":
            offset_x = -10
            offset_y = -0.03
        ax.annotate(
            method, (wall, auroc),
            textcoords="offset points",
            xytext=(offset_x + 5, offset_y + 3),
            fontsize=6, color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("AUROC")
    ax.set_title("Speed–accuracy tradeoff", fontsize=9, pad=4)
    ax.set_ylim(0.4, 0.85)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Highlight Pareto-optimal region
    ax.axhspan(0.7, 0.85, alpha=0.05, color="green")
    ax.axvspan(0.01, 10, alpha=0.05, color="green")

    fig.tight_layout(pad=0.5)
    save_pub(fig, "benchmark_speed_accuracy")
    plt.close(fig)


# ============================================================================
# Figure 3: Real data network topology comparison (2 datasets side-by-side)
# ============================================================================
def plot_real_topology(real_topology):
    """
    Figure contract:
      Core conclusion: AdaCoNet produces biologically meaningful networks
        with comparable topology to established methods, at a fraction of
        the computational cost.
    """
    real_df = pd.DataFrame(real_topology)
    datasets = real_df["dataset"].unique()

    if len(datasets) == 0:
        return

    fig, axes = plt.subplots(1, len(datasets) + 1,
                             figsize=(3.5 * (len(datasets) + 1), 3.0))
    if len(datasets) == 1:
        axes = [axes]

    for ds_idx, ds_name in enumerate(datasets):
        ax = axes[ds_idx]
        add_panel_label(ax, chr(ord("a") + ds_idx))

        ds_df = real_df[real_df["dataset"] == ds_name]
        methods = [m for m in METHOD_ORDER if m in ds_df["method"].unique()]

        # Grouped bar: modularity + max_degree (normalized)
        modularity = []
        max_deg = []
        times = []
        for m in methods:
            m_data = ds_df[ds_df["method"] == m]
            modularity.append(m_data["modularity"].values[0] if len(m_data) > 0 else 0)
            max_deg.append(m_data["max_degree"].values[0] if len(m_data) > 0 else 0)
            times.append(m_data["wall_time_sec"].values[0] if len(m_data) > 0 else 0)

        n_methods = len(methods)
        x = np.arange(n_methods)
        colors = [METHOD_COLORS[m] for m in methods]

        # Modularity bars
        bars = ax.bar(x, modularity, color=colors,
                      edgecolor="black", linewidth=0.5, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=6)
        ax.set_ylabel("Modularity")
        ax.set_ylim(0, max(modularity) * 1.3 if max(modularity) > 0 else 1)
        ax.set_title(ds_name[:30], fontsize=8, pad=4)

        # Annotate with time
        for i, (bar, t) in enumerate(zip(bars, times)):
            t_label = f"{t:.1f}s" if t < 10 else f"{t:.0f}s"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    t_label, ha="center", va="bottom", fontsize=5, color="#606060")

    # Final panel: runtime comparison across datasets
    if len(datasets) > 0:
        ax = axes[-1]
        add_panel_label(ax, chr(ord("a") + len(datasets)))

        ds = datasets[0]
        ds_df = real_df[real_df["dataset"] == ds]
        methods = [m for m in METHOD_ORDER if m in ds_df["method"].unique()]
        times = []
        for m in methods:
            m_data = ds_df[ds_df["method"] == m]
            times.append(m_data["wall_time_sec"].values[0] if len(m_data) > 0 else 0)

        colors = [METHOD_COLORS[m] for m in methods]
        ax.barh(range(len(methods)), times, color=colors,
                edgecolor="black", linewidth=0.5, height=0.6)
        for i, t in enumerate(times):
            label = f"{t:.1f}s" if t < 10 else f"{t:.0f}s"
            ax.text(max(t * 1.05, 0.1), i, label, va="center", fontsize=6)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=6)
        ax.set_xlabel("Wall time (s)")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_title("Runtime (real data)", fontsize=8, pad=4)
        ax.invert_yaxis()

    fig.tight_layout(pad=0.8)
    save_pub(fig, "benchmark_real_topology")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    results = load_results()
    if results is None:
        print("ERROR: No results found. Run run_real_benchmark.py first.")
        sys.exit(1)

    sim_metrics = results.get("simulated_metrics", [])
    real_topology = results.get("real_topology", [])

    print(f"Simulated metrics: {len(sim_metrics)} entries")
    print(f"Real topology: {len(real_topology)} entries")

    if sim_metrics:
        print("\nGenerating main benchmark figure...")
        plot_main_benchmark(sim_metrics, real_topology)

        print("Generating speed-accuracy tradeoff figure...")
        plot_speed_accuracy(sim_metrics)

    if real_topology:
        print("Generating real data topology figure...")
        plot_real_topology(real_topology)

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
