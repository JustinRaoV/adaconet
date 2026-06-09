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

configs = ['v4 N=200,P=50', 'v4 N=500,P=200', 'v4 N=500,P=500', 'v4 N=1000,P=500', 'v4 N=1000,P=1000']
config_labels = ['N200\nP50', 'N500\nP200', 'N500\nP500', 'N1000\nP500', 'N1000\nP1000']
all_methods = ['AdaCoNet', 'SparCC', 'REBACCA', 'CCLasso', 'FastSpar', 'Spearman', 'Prop', 'Glasso', 'SPIEC-EASI']

# Colors: AdaCoNet is blue, others are grays with slight variation
colors = {
    'AdaCoNet': '#2171B5',
    'SparCC': '#969696',
    'REBACCA': '#BDBDBD',
    'CCLasso': '#737373',
    'FastSpar': '#D9D9D9',
    'Spearman': '#A0A0A0',
    'Prop': '#C0C0C0',
    'Glasso': '#888888',
    'SPIEC-EASI': '#B0B0B0',
}

fig, ax = plt.subplots(figsize=(10, 5.5))

n_methods = len(all_methods)
n_configs = len(configs)
bar_width = 0.09
group_width = n_methods * bar_width + 0.05

x = np.arange(n_configs)

for i, method in enumerate(all_methods):
    vals = []
    for cfg in configs:
        if method in data[cfg]:
            vals.append(data[cfg][method]['auroc'])
        else:
            vals.append(np.nan)
    offset = (i - n_methods / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, vals, bar_width, color=colors[method],
                  edgecolor='white', linewidth=0.3, label=method, zorder=2)

# Dashed line at 0.5
ax.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
ax.text(4.55, 0.505, 'random', fontsize=7, color='#E74C3C', alpha=0.7, va='bottom', ha='right')

ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
ax.set_xlabel('')
ax.set_xticks(x)
ax.set_xticklabels(config_labels, fontsize=9)
ax.set_ylim(0.4, 1.02)
ax.set_xlim(-0.5, n_configs - 0.5)
ax.yaxis.set_ticks(np.arange(0.4, 1.1, 0.1))

# Light grid
ax.yaxis.grid(True, linestyle='-', linewidth=0.3, color='#D5D8DC', zorder=0)
ax.set_axisbelow(True)

# Legend at top
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                   ncol=9, frameon=False, fontsize=7.5, handlelength=1.2, columnspacing=1.0)
for leg_item in legend.get_lines():
    leg_item.set_linewidth(3)

# Panel label
ax.text(-0.48, 1.02, 'b', fontsize=16, fontweight='bold', transform=ax.transAxes, va='top')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig2_v4_benchmark.png',
            dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('/Users/justin/project/school/method/adaconet/docs/figures/fig2_v4_benchmark.pdf',
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 2 saved.")
