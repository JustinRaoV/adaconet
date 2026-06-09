import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.lines import Line2D

# Load data
with open('/Users/justin/project/school/method/adaconet/results/adaptive_ensemble_benchmark.json') as f:
    data = json.load(f)

# Style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

all_methods = ['AdaCoNet', 'SparCC', 'REBACCA', 'CCLasso', 'FastSpar', 'Spearman', 'Prop', 'Glasso', 'SPIEC-EASI']

# Markers and colors for each method
method_styles = {
    'AdaCoNet':    {'marker': 'o', 'color': '#2171B5', 'ms': 10},
    'SparCC':      {'marker': 's', 'color': '#E74C3C', 'ms': 7},
    'REBACCA':     {'marker': '^', 'color': '#27AE60', 'ms': 7},
    'CCLasso':     {'marker': 'D', 'color': '#8E44AD', 'ms': 7},
    'FastSpar':    {'marker': 'v', 'color': '#F39C12', 'ms': 7},
    'Spearman':    {'marker': 'P', 'color': '#95A5A6', 'ms': 7},
    'Prop':        {'marker': '*', 'color': '#E67E22', 'ms': 10},
    'Glasso':      {'marker': 'X', 'color': '#566573', 'ms': 7},
    'SPIEC-EASI':  {'marker': 'p', 'color': '#1ABC9C', 'ms': 7},
}

def compute_pareto_frontier(points):
    """Compute Pareto frontier for minimizing x (time) and maximizing y (auroc).
    A point is non-dominated if no other point has both lower time and higher auroc."""
    frontier = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j has <= time and >= auroc (with at least one strict)
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Panel e: v4 N=500,P=500 ---
cfg_e = 'v4 N=500,P=500'
points_e = []
methods_e = []
for m in all_methods:
    if m in data[cfg_e]:
        t = data[cfg_e][m]['time']
        a = data[cfg_e][m]['auroc']
        points_e.append((t, a))
        methods_e.append(m)

for i, m in enumerate(methods_e):
    t, a = points_e[i]
    s = method_styles[m]
    zorder = 10 if m in ['AdaCoNet', 'Prop'] else 5
    ax1.scatter(t, a, marker=s['marker'], c=s['color'], s=s['ms']**2,
                edgecolors='white', linewidths=0.5, zorder=zorder)
    # Labels
    offset_x, offset_y = 0.06, 0.015
    if m == 'SparCC':
        offset_y = -0.025
    elif m == 'Glasso':
        offset_y = 0.02
    ax1.annotate(m, (t, a), xytext=(offset_x, offset_y), textcoords='offset points',
                 fontsize=6.5, color=s['color'], fontweight='bold' if m in ['AdaCoNet', 'Prop'] else 'normal')

# Pareto frontier
frontier_idx = compute_pareto_frontier(points_e)
frontier_points = sorted([points_e[i] for i in frontier_idx], key=lambda p: p[0])
fx = [p[0] for p in frontier_points]
fy = [p[1] for p in frontier_points]
ax1.plot(fx, fy, '--', color='#2C3E50', linewidth=1.0, alpha=0.5, zorder=3)

ax1.set_xscale('log')
ax1.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
ax1.set_ylabel('AUROC', fontsize=10, fontweight='bold')
ax1.set_ylim(0.4, 0.9)
ax1.axhline(y=0.5, color='#E74C3C', linestyle=':', linewidth=0.6, alpha=0.5)
ax1.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax1.xaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('v4  N=500, P=500', fontsize=10, fontweight='bold', pad=8)
ax1.text(-0.15, 1.05, 'e', fontsize=16, fontweight='bold', transform=ax1.transAxes, va='top')

# --- Panel f: SD2 N500 ---
cfg_f = 'SD2 N500'
points_f = []
methods_f = []
for m in all_methods:
    if m in data[cfg_f]:
        t = data[cfg_f][m]['time']
        a = data[cfg_f][m]['auroc']
        points_f.append((t, a))
        methods_f.append(m)

for i, m in enumerate(methods_f):
    t, a = points_f[i]
    s = method_styles[m]
    zorder = 10 if m in ['AdaCoNet', 'Prop'] else 5
    ax2.scatter(t, a, marker=s['marker'], c=s['color'], s=s['ms']**2,
                edgecolors='white', linewidths=0.5, zorder=zorder)
    offset_x, offset_y = 0.06, 0.015
    if m == 'Prop':
        offset_y = -0.03
    elif m == 'Spearman':
        offset_y = -0.03
    elif m == 'AdaCoNet':
        offset_y = -0.03
    ax2.annotate(m, (t, a), xytext=(offset_x, offset_y), textcoords='offset points',
                 fontsize=6.5, color=s['color'], fontweight='bold' if m in ['AdaCoNet', 'Prop'] else 'normal')

# Pareto frontier
frontier_idx_f = compute_pareto_frontier(points_f)
frontier_points_f = sorted([points_f[i] for i in frontier_idx_f], key=lambda p: p[0])
fx_f = [p[0] for p in frontier_points_f]
fy_f = [p[1] for p in frontier_points_f]
ax2.plot(fx_f, fy_f, '--', color='#2C3E50', linewidth=1.0, alpha=0.5, zorder=3)

ax2.set_xscale('log')
ax2.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
ax2.set_ylabel('AUROC', fontsize=10, fontweight='bold')
ax2.set_ylim(0.38, 1.0)
ax2.axhline(y=0.5, color='#E74C3C', linestyle=':', linewidth=0.6, alpha=0.5)
ax2.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax2.xaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('SparseDOSSA2  N=500', fontsize=10, fontweight='bold', pad=8)
ax2.text(-0.15, 1.05, 'f', fontsize=16, fontweight='bold', transform=ax2.transAxes, va='top')

plt.tight_layout(w_pad=3)
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig4_speed_accuracy.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig4_speed_accuracy.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 4 saved.")
