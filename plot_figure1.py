"""Figure 1: AdaCoNet architecture diagram for Nature Communications."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
})

# Nature palette (from api.md)
C = {
    "blue_main": "#0F4D92",
    "blue_sec": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "bg_peach": "#F0E0D0",
    "bg_aqua": "#E0F0F0",
    "bg_lilac": "#E0E0F0",
    "delta_up": "#2E9E44",
}

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

def box(x, y, w, h, label, fc, ec=None, fontsize=7.5, bold=False):
    """Draw a rounded box with centered text."""
    if ec is None:
        ec = C["neutral_dark"]
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        fc=fc, ec=ec, lw=0.8, zorder=2,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, zorder=3, wrap=True,
            multialignment="center")

def arrow(x1, y1, x2, y2, style="->", lw=1.0, color=None):
    """Draw an arrow between two points."""
    if color is None:
        color = C["neutral_mid"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                connectionstyle="arc3,rad=0"), zorder=1)

# ===== Title =====
ax.text(5, 6.8, "AdaCoNet: Diversity-Aware Ensemble Architecture",
        ha="center", fontsize=10, fontweight="bold", color=C["neutral_dark"])

# ===== Input =====
box(3.5, 6.0, 3.0, 0.6, "Count Matrix $X$\n($n$ samples × $p$ taxa)",
    C["bg_peach"], fontsize=8, bold=True)
arrow(5, 6.0, 5, 5.7)

# ===== Step 0: Filter =====
box(3.2, 5.2, 3.6, 0.5, "Prevalence Filtering\n(min_prevalence = 0.05)",
    C["neutral_light"], fontsize=7)
arrow(5, 5.2, 5, 4.9)

# ===== Three parallel layers =====
layer_y = 3.4
layer_h = 1.5
layer_w = 2.8

# Layer 1: DM Foundation
x1 = 0.5
box(x1, layer_y, layer_w, layer_h, "", C["bg_aqua"], fontsize=7)
ax.text(x1 + layer_w/2, layer_y + layer_h - 0.15,
        "Layer 1: DM Foundation", ha="center", fontsize=8, fontweight="bold",
        color=C["blue_main"])
ax.text(x1 + layer_w/2, layer_y + 0.95,
        "Method-of-moments\n+ Newton-Raphson\n$|\\alpha| = (1-r)/r$",
        ha="center", fontsize=6.5, color=C["neutral_dark"])
ax.text(x1 + layer_w/2, layer_y + 0.35,
        "Posterior Correlation\n$R^{DM}$",
        ha="center", fontsize=7, fontweight="bold", color=C["blue_main"])

# Layer 2: Adaptive CLR
x2 = 3.6
box(x2, layer_y, layer_w, layer_h, "", C["bg_lilac"], fontsize=7)
ax.text(x2 + layer_w/2, layer_y + layer_h - 0.15,
        "Layer 2: Adaptive CLR", ha="center", fontsize=8, fontweight="bold",
        color=C["violet"])
ax.text(x2 + layer_w/2, layer_y + 0.95,
        "N/P > 2: Bayesian CLR\nN/P ≤ 2: Regularized\n+ Ledoit-Wolf shrinkage",
        ha="center", fontsize=6.5, color=C["neutral_dark"])
ax.text(x2 + layer_w/2, layer_y + 0.35,
        "Spearman on CLR\n$S^{Spear}$",
        ha="center", fontsize=7, fontweight="bold", color=C["violet"])

# Layer 3: Proportionality
x3 = 6.7
box(x3, layer_y, layer_w, layer_h, "", C["bg_peach"], fontsize=7)
ax.text(x3 + layer_w/2, layer_y + layer_h - 0.15,
        "Layer 3: Proportionality", ha="center", fontsize=8, fontweight="bold",
        color=C["red_strong"])
ax.text(x3 + layer_w/2, layer_y + 0.95,
        "VLR-based ratio\n$\\rho_p = 1 - \\frac{VLR}{var_j + var_k}$",
        ha="center", fontsize=6.5, color=C["neutral_dark"])
ax.text(x3 + layer_w/2, layer_y + 0.35,
        "Proportionality\n$\\rho_p$",
        ha="center", fontsize=7, fontweight="bold", color=C["red_strong"])

# Arrows from filter to each layer
for x_center in [x1 + layer_w/2, x2 + layer_w/2, x3 + layer_w/2]:
    arrow(5, 4.9, x_center, layer_y + layer_h, lw=0.8, color=C["neutral_mid"])

# ===== Ensemble =====
ens_y = 2.3
box(2.0, ens_y, 6.0, 0.9, "", "#FFFFFF", fontsize=7)
ax.text(5.0, ens_y + 0.75, "Diversity-Aware Equal-Weight Ensemble",
        ha="center", fontsize=8, fontweight="bold", color=C["neutral_dark"])
ax.text(5.0, ens_y + 0.35,
        "Min-max normalize → $W = \\frac{1}{K}\\sum_{l=1}^{K} S^*_l$  |  Diversity diagnostic: pairwise signal correlation",
        ha="center", fontsize=6.5, color=C["neutral_dark"])

# Arrows from layers to ensemble
for x_center in [x1 + layer_w/2, x2 + layer_w/2, x3 + layer_w/2]:
    arrow(x_center, layer_y, x_center, ens_y + 0.9, lw=0.8, color=C["neutral_mid"])

# ===== StARS =====
stars_y = 1.3
box(3.0, stars_y, 4.0, 0.7,
    "StARS Threshold Selection\nSubsample 80% → minimize edge instability",
    C["bg_aqua"], fontsize=7)
arrow(5, ens_y, 5, stars_y + 0.7, lw=1.0, color=C["neutral_mid"])

# ===== Output =====
out_y = 0.2
box(3.0, out_y, 4.0, 0.8,
    "Co-occurrence Network $G$\n(Adjacency matrix, signed weights)",
    C["delta_up"], fontsize=8, bold=True, ec=C["delta_up"])
arrow(5, stars_y, 5, out_y + 0.8, lw=1.0, color=C["neutral_dark"])

# ===== Layer labels on left =====
ax.text(0.2, 5.5, "A", fontsize=11, fontweight="bold", color=C["neutral_dark"])
ax.text(0.2, 4.1, "B", fontsize=11, fontweight="bold", color=C["neutral_dark"])
ax.text(0.2, 2.7, "C", fontsize=11, fontweight="bold", color=C["neutral_dark"])
ax.text(0.2, 0.6, "D", fontsize=11, fontweight="bold", color=C["neutral_dark"])

# Panel labels
ax.text(0.2, 5.85, "Input & filtering", fontsize=6.5, color=C["neutral_mid"],
        style="italic")
ax.text(0.2, 4.45, "Three parallel\nstatistical layers", fontsize=6.5,
        color=C["neutral_mid"], style="italic")
ax.text(0.2, 3.05, "Ensemble\nintegration", fontsize=6.5,
        color=C["neutral_mid"], style="italic")
ax.text(0.2, 0.95, "Edge selection\n& output", fontsize=6.5,
        color=C["neutral_mid"], style="italic")

plt.tight_layout()
fig.savefig("/Users/justin/project/school/method/adaconet/results/figures/figure1_architecture.svg",
            bbox_inches="tight")
fig.savefig("/Users/justin/project/school/method/adaconet/results/figures/figure1_architecture.pdf",
            bbox_inches="tight")
fig.savefig("/Users/justin/project/school/method/adaconet/results/figures/figure1_architecture.tiff",
            dpi=600, bbox_inches="tight")
plt.close()
print("Figure 1 saved: SVG + PDF + TIFF@600dpi")
