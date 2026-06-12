#!/usr/bin/env python3
"""Regenerate paper figures (fig2, fig3, fig4) from 10-seed benchmark data."""

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Nature-style rcParams ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

# ── Colours ──
BLUE   = "#0F4D92"   # AdaCoNet
ORANGE = "#E8A735"   # Proportionality
GRAY   = "#A0A0A0"   # Others

METHOD_COLORS = {
    "AdaCoNet": BLUE, "SparCC": GRAY, "REBACCA": GRAY,
    "FastSpar": GRAY, "CCLasso": GRAY, "Spearman": GRAY,
    "Proportionality": ORANGE, "Glasso": GRAY, "SPIEC-EASI": GRAY,
}
METHOD_ORDER = [
    "AdaCoNet", "SparCC", "REBACCA", "FastSpar", "CCLasso",
    "Proportionality", "Glasso", "Spearman", "SPIEC-EASI",
]
SHORT = {
    "AdaCoNet": "AdaCoNet", "SparCC": "SparCC", "REBACCA": "REBACCA",
    "FastSpar": "FastSpar", "CCLasso": "CCLasso", "Spearman": "Spearman",
    "Proportionality": "Prop", "Glasso": "Glasso", "SPIEC-EASI": "SPIEC-EASI",
}

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"  Saved {name}.{{pdf,png}}")


# ── Load data ──
with open("results/ablation_auprc_10seeds.json") as f:
    data = json.load(f)

# ===================================================================
# Fig 2 — v4 grouped bar chart (panel b)
# ===================================================================
def plot_fig2():
    configs = ["v4 N=200,P=50", "v4 N=500,P=200", "v4 N=500,P=500",
               "v4 N=1000,P=500", "v4 N=1000,P=1000"]
    xlabels = ["N200\nP50", "N500\nP200", "N500\nP500", "N1k\nP500", "N1k\nP1k"]
    n_cfg = len(configs)
    n_methods = len(METHOD_ORDER)

    # Gather AUROC
    vals = np.zeros((n_methods, n_cfg))
    for ci, cfg in enumerate(configs):
        methods = data[cfg]["methods"]
        for mi, m in enumerate(METHOD_ORDER):
            if m in methods:
                vals[mi, ci] = methods[m]["auroc_mean"]
            else:
                vals[mi, ci] = np.nan

    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.09
    x = np.arange(n_cfg)
    for mi, m in enumerate(METHOD_ORDER):
        offset = (mi - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals[mi], width, color=METHOD_COLORS[m],
                      edgecolor="white", linewidth=0.3, label=SHORT[m],
                      zorder=3 if m == "AdaCoNet" else 1)
        # Mark missing as empty
        for ci in range(n_cfg):
            if np.isnan(vals[mi, ci]):
                pass  # nan bars just don't show

    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, ls="--", color="#999999", lw=0.6, zorder=0)
    ax.set_title("b", fontweight="bold", loc="left", fontsize=10, x=-0.02, y=1.05)

    # Legend — two rows, top right
    handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS[m], ec="none")
               for m in METHOD_ORDER]
    ax.legend(handles, [SHORT[m] for m in METHOD_ORDER],
              ncol=5, loc="upper right", fontsize=5.5,
              bbox_to_anchor=(1.0, 1.0), columnspacing=0.8, handletextpad=0.4,
              handlelength=0.8)

    fig.tight_layout()
    save(fig, "docs/figures/fig2_v4_benchmark")
    plt.close(fig)


# ===================================================================
# Fig 3 — Cross-simulator horizontal bars (panels c, d)
# ===================================================================
def plot_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0), sharex=True)

    configs_labels = [
        ("v4 N=500,P=500", "c", "v4 (N=500, P=500)"),
        ("SD2 N500",       "d", "SparseDOSSA2 (N=500)"),
    ]

    for ax, (cfg, panel, title) in zip(axes, configs_labels):
        methods = data[cfg]["methods"]
        # Sort by AUROC descending
        sorted_methods = sorted(METHOD_ORDER, key=lambda m: methods.get(m, {}).get("auroc_mean", 0))

        aurocs = []
        colors = []
        labels = []
        for m in sorted_methods:
            if m in methods:
                aurocs.append(methods[m]["auroc_mean"])
                colors.append(METHOD_COLORS[m])
                labels.append(SHORT[m])

        y = np.arange(len(labels))
        ax.barh(y, aurocs, color=colors, edgecolor="white", linewidth=0.3, height=0.7, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlim(0.4, 1.02)
        ax.axvline(0.5, ls="--", color="#999999", lw=0.6, zorder=0)
        ax.set_xlabel("AUROC")
        ax.set_title(title, fontsize=7, pad=4)
        ax.text(-0.08, 1.05, panel, fontweight="bold", fontsize=10,
                transform=ax.transAxes, va="top")

    fig.tight_layout()
    save(fig, "docs/figures/fig3_cross_simulator")
    plt.close(fig)


# ===================================================================
# Fig 4 — Speed–accuracy Pareto (panels e, f)
# ===================================================================
def plot_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.0))

    configs_labels = [
        ("v4 N=500,P=500", "e", "v4 (N=500, P=500)", (1e-3, 2e2)),
        ("SD2 N500",       "f", "SparseDOSSA2 (N=500)", (5e-4, 1e2)),
    ]

    for ax, (cfg, panel, title, xlim) in zip(axes, configs_labels):
        methods = data[cfg]["methods"]
        for m in METHOD_ORDER:
            if m not in methods:
                continue
            t = methods[m]["time_mean"]
            a = methods[m]["auroc_mean"]
            color = METHOD_COLORS[m]

            if m == "AdaCoNet":
                marker, sz, zorder = "o", 40, 5
            elif m == "Proportionality":
                marker, sz, zorder = "s", 35, 4
            else:
                marker, sz, zorder = "^", 25, 2

            ax.scatter(t, a, c=color, s=sz, marker=marker,
                       edgecolor="white", linewidth=0.3, zorder=zorder)
            # Label
            lbl = SHORT[m]
            dx, dy = 0.04, 0.02
            if m == "AdaCoNet":
                dx, dy = 0.06, -0.04
            elif m == "Proportionality":
                dx, dy = 0.06, 0.01
            ax.annotate(lbl, (t, a), xytext=(dx, dy),
                        textcoords="offset points", fontsize=5,
                        color=color if m in ("AdaCoNet", "Proportionality") else "#555555")

        # Pareto dashed line for Prop → REBACCA
        if "Proportionality" in methods and "REBACCA" in methods:
            tp = methods["Proportionality"]["time_mean"]
            ap = methods["Proportionality"]["auroc_mean"]
            tr = methods["REBACCA"]["time_mean"]
            ar = methods["REBACCA"]["auroc_mean"]
            ax.plot([tp, tr], [ap, ar], ls="--", color="#BBBBBB", lw=0.5, zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AUROC")
        ax.set_title(title, fontsize=7, pad=4)
        ax.text(-0.08, 1.05, panel, fontweight="bold", fontsize=10,
                transform=ax.transAxes, va="top")

    fig.tight_layout()
    save(fig, "docs/figures/fig4_speed_accuracy")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures from 10-seed data...")
    plot_fig2()
    plot_fig3()
    plot_fig4()
    print("Done.")
