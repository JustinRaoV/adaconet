import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Panel c: v4 N=500,P=500 ---
cfg_c = 'v4 N=500,P=500'
aurocs_c = []
for m in all_methods:
    if m in data[cfg_c]:
        aurocs_c.append(data[cfg_c][m]['auroc'])
    else:
        aurocs_c.append(0)

# Sort by AUROC
sorted_pairs_c = sorted(zip(all_methods, aurocs_c), key=lambda x: x[1])
methods_c = [p[0] for p in sorted_pairs_c]
vals_c = [p[1] for p in sorted_pairs_c]

bar_colors_c = []
for m in methods_c:
    if m == 'AdaCoNet':
        bar_colors_c.append('#2171B5')
    elif m == 'Prop':
        bar_colors_c.append('#E67E22')
    else:
        bar_colors_c.append('#BDC3C7')

y_pos = np.arange(len(methods_c))
bars1 = ax1.barh(y_pos, vals_c, 0.6, color=bar_colors_c, edgecolor='white', linewidth=0.3, zorder=2)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(methods_c, fontsize=8.5)
ax1.set_xlabel('AUROC', fontsize=10, fontweight='bold')
ax1.set_xlim(0.4, 0.9)
ax1.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

# Add value labels
for i, v in enumerate(vals_c):
    ax1.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=7, color='#2C3E50')

ax1.xaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('v4  N=500, P=500', fontsize=10, fontweight='bold', pad=8)

# Panel label 'c'
ax1.text(-0.15, 1.05, 'c', fontsize=16, fontweight='bold', transform=ax1.transAxes, va='top')

# --- Panel d: SD2 N500 ---
cfg_d = 'SD2 N500'
aurocs_d = []
for m in all_methods:
    if m in data[cfg_d]:
        aurocs_d.append(data[cfg_d][m]['auroc'])
    else:
        aurocs_d.append(0)

sorted_pairs_d = sorted(zip(all_methods, aurocs_d), key=lambda x: x[1])
methods_d = [p[0] for p in sorted_pairs_d]
vals_d = [p[1] for p in sorted_pairs_d]

bar_colors_d = []
for m in methods_d:
    if m == 'AdaCoNet':
        bar_colors_d.append('#2171B5')
    elif m == 'Prop':
        bar_colors_d.append('#E67E22')
    else:
        bar_colors_d.append('#BDC3C7')

bars2 = ax2.barh(y_pos, vals_d, 0.6, color=bar_colors_d, edgecolor='white', linewidth=0.3, zorder=2)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(methods_d, fontsize=8.5)
ax2.set_xlabel('AUROC', fontsize=10, fontweight='bold')
ax2.set_xlim(0.35, 1.0)
ax2.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)

for i, v in enumerate(vals_d):
    ax2.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=7, color='#2C3E50')

ax2.xaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('SparseDOSSA2  N=500', fontsize=10, fontweight='bold', pad=8)

# Panel label 'd'
ax2.text(-0.15, 1.05, 'd', fontsize=16, fontweight='bold', transform=ax2.transAxes, va='top')

plt.tight_layout(w_pad=3)
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig3_cross_simulator.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig3_cross_simulator.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved.")
