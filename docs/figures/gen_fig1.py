import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Style setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# Color palette
C_INPUT = '#D4E6F1'       # light blue
C_DM = '#2171B5'          # dark blue
C_ADAPT = '#2E86C1'       # medium blue
C_VLR = '#1A5276'         # navy
C_GC = '#2980B9'          # sky blue
C_SELECT = '#F39C12'      # orange
C_ENSEMBLE = '#27AE60'    # green
C_OUTPUT = '#1E8449'      # dark green
C_BOX_TEXT = 'white'
C_ARROW = '#555555'
C_BG = '#F8F9FA'
C_PATH_A = '#E8F8F5'      # light green tint for "all 4" path
C_PATH_B = '#FEF9E7'      # light yellow tint for "exclude Spearman" path

fig, ax = plt.subplots(figsize=(10.24, 7.68))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# Panel label
ax.text(0.15, 8.3, 'a', fontsize=16, fontweight='bold', va='top', ha='left')

def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=8.5, alpha=1.0, linewidth=0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='#BDC3C7', linewidth=linewidth,
                          alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=3,
            linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)

# === INPUT ===
draw_box(ax, 3.8, 7.7, 2.4, 0.45, 'Count Matrix  X', C_INPUT, text_color='#2C3E50', fontsize=10, linewidth=1.0)
draw_arrow(ax, 5.0, 7.7, 5.0, 7.35)

# === LAYER 1: DM Posterior Correlation ===
draw_box(ax, 2.5, 6.75, 5.0, 0.55,
         'Layer 1: DM Posterior Correlation\n$\\alpha$ estimation  →  posterior mean  →  Pearson',
         C_DM, fontsize=8.5)
draw_arrow(ax, 5.0, 6.75, 5.0, 6.35)

# === LAYER 2: Adaptive Spearman CLR ===
draw_box(ax, 2.0, 5.7, 6.0, 0.6,
         'Layer 2: Adaptive Spearman CLR\nN/P > 2  →  Bayesian CLR   |   else  →  raw CLR + LW',
         C_ADAPT, fontsize=8.5)
draw_arrow(ax, 5.0, 5.7, 5.0, 5.3)

# === LAYER 3: VLR Proportionality ===
draw_box(ax, 2.5, 4.7, 5.0, 0.55,
         'Layer 3: VLR Proportionality\nnon-z-scored CLR  →  $\\rho_p$',
         C_VLR, fontsize=8.5)
draw_arrow(ax, 5.0, 4.7, 5.0, 4.3)

# === LAYER 4: Gaussian Copula ===
draw_box(ax, 2.5, 3.7, 5.0, 0.55,
         'Layer 4: Gaussian Copula\nempirical CDF  →  $\\Phi^{-1}$  →  Pearson',
         C_GC, fontsize=8.5)
draw_arrow(ax, 5.0, 3.7, 5.0, 3.3)

# === Model-Based Selection ===
draw_box(ax, 2.2, 2.55, 5.6, 0.7,
         'Model-Based Selection\n$|\\alpha|/p \\geq 0.05$  →  all 4 layers    |    $|\\alpha|/p < 0.05$  →  exclude Spearman',
         C_SELECT, text_color='white', fontsize=8.5)

# Two arrows from selection to ensemble (showing two paths)
draw_arrow(ax, 3.8, 2.55, 3.5, 2.1, color='#E67E22', lw=1.0)
draw_arrow(ax, 6.2, 2.55, 6.5, 2.1, color='#E67E22', lw=1.0)

# Small path labels
ax.text(3.2, 2.25, '4 layers', fontsize=6.5, color='#E67E22', ha='center', fontstyle='italic')
ax.text(6.8, 2.25, '3 layers', fontsize=6.5, color='#E67E22', ha='center', fontstyle='italic')

# === Ensemble ===
draw_box(ax, 2.5, 1.5, 5.0, 0.55,
         'Ensemble: Min-max normalize  →  Equal weights  →  W',
         C_ENSEMBLE, fontsize=9)
draw_arrow(ax, 5.0, 1.5, 5.0, 1.1)

# === StARS + Output ===
draw_box(ax, 3.0, 0.45, 4.0, 0.6,
         'StARS threshold selection  →  Binary network',
         C_OUTPUT, fontsize=9)

# Side annotation: Layer numbering
for i, (yp, label) in enumerate([(7.02, '1'), (6.0, '2'), (4.97, '3'), (3.97, '4')]):
    ax.text(1.6, yp, label, fontsize=11, fontweight='bold', color='#AEB6BF',
            ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='#ECF0F1', edgecolor='#D5D8DC', linewidth=0.8))

plt.tight_layout(pad=0.3)
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig1_architecture.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig1_architecture.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 1 saved.")
