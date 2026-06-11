#!/usr/bin/env python3
"""Plot c_ref sensitivity analysis figure (Supplementary Fig S1)."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.linewidth'] = 0.8

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "results", "sensitivity_cref.json")) as f:
    data = json.load(f)

C_REF_VALUES = [float(x) for x in data["v4_N500P500"]["c_ref_sweep"].keys()]

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# --- Panel a: v4 N500P500 ---
ax = axes[0]
v4 = data["v4_N500P500"]
aurocs_std = [v4["c_ref_sweep"][str(c)]["auroc"] for c in C_REF_VALUES]
aurocs_nof = [v4["c_ref_sweep_ignore_zero_frac"][str(c)]["auroc"] for c in C_REF_VALUES]
included = [v4["c_ref_sweep"][str(c)]["spearman_included"] for c in C_REF_VALUES]

ax.plot(C_REF_VALUES, aurocs_std, 'o-', color='#2563eb', lw=1.5, ms=4, label='Standard (with $f_0$ guard)')
ax.plot(C_REF_VALUES, aurocs_nof, 's--', color='#9333ea', lw=1.2, ms=3, alpha=0.6, label='Ignoring $f_0$ guard')
ax.axvline(0.05, color='#dc2626', ls=':', lw=1.2, alpha=0.7, label='$c_{\\mathrm{ref}} = 0.05$')
# Shade transition zone
ax.axvspan(0.12, 0.13, alpha=0.1, color='orange', label='Transition zone')
ax.set_xlabel('$c_{\\mathrm{ref}}$')
ax.set_ylabel('AUROC')
ax.set_title('(a) v4 Simulator ($N{=}500, P{=}500$)')
ax.set_ylim(0.55, 0.78)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# --- Panel b: SD2 N500 ---
ax = axes[1]
sd2 = data["SD2_N500"]
aurocs_std = [sd2["c_ref_sweep"][str(c)]["auroc"] for c in C_REF_VALUES]
aurocs_nof = [sd2["c_ref_sweep_ignore_zero_frac"][str(c)]["auroc"] for c in C_REF_VALUES]

ax.plot(C_REF_VALUES, aurocs_std, 'o-', color='#2563eb', lw=1.5, ms=4, label='Standard (with $f_0$ guard)')
ax.plot(C_REF_VALUES, aurocs_nof, 's--', color='#9333ea', lw=1.2, ms=3, alpha=0.6, label='Ignoring $f_0$ guard')
ax.axvline(0.05, color='#dc2626', ls=':', lw=1.2, alpha=0.7, label='$c_{\\mathrm{ref}} = 0.05$')
ax.axvspan(0.06, 0.07, alpha=0.1, color='orange', label='Transition zone (no $f_0$)')
ax.set_xlabel('$c_{\\mathrm{ref}}$')
ax.set_ylabel('AUROC')
ax.set_title('(b) SparseDOSSA2 ($N{=}500$)')
ax.set_ylim(0.65, 0.95)
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
out_dir = os.path.join(BASE, "docs", "figures")
os.makedirs(out_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    plt.savefig(os.path.join(out_dir, f"figS1_sensitivity_cref.{ext}"),
                dpi=300, bbox_inches='tight')
print(f"Saved to docs/figures/figS1_sensitivity_cref.{{pdf,png}}")
