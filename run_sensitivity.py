#!/usr/bin/env python3
"""c_ref sensitivity analysis for AdaCoNet.

Sweeps c_ref values [0.01..0.15] on two datasets:
  1. v4 simulator N=500, P=500 (multinomial regime)
  2. SparseDOSSA2 Stool N=500 (copula regime)

The pipeline is run once per dataset to extract per-layer score matrices,
then for each c_ref the Spearman-inclusion criterion is re-evaluated and
the ensemble is recombined with equal weights.  AUROC is computed against
the respective ground truth.

The key question: does c_ref=0.05 sit in a "gap" between the two
simulators, or is the result very sensitive to the exact threshold?
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from adaconet import AdaCoNetPipeline

from run_benchmarks import (
    _generate_compositional_data,
    _generate_ground_truth_network,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sweep range chosen to bracket the alpha_per_taxon of both datasets:
#   SD2 ~ 0.066, v4 ~ 0.125
C_REF_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.10, 0.11, 0.12, 0.13, 0.14, 0.15]


def _normalize_score(S: np.ndarray) -> np.ndarray:
    """Min-max normalize off-diagonal entries to [0, 1]."""
    S = S.copy()
    np.fill_diagonal(S, 0.0)
    s_min, s_max = S.min(), S.max()
    if s_max - s_min < 1e-15:
        return np.zeros_like(S)
    S_norm = (S - s_min) / (s_max - s_min)
    np.fill_diagonal(S_norm, 0.0)
    return S_norm


def _compute_auroc(W: np.ndarray, truth_binary: np.ndarray) -> float:
    """AUROC from continuous score matrix vs binary ground truth."""
    idx = np.triu_indices_from(W, k=1)
    y_true = truth_binary[idx].astype(int)
    y_scores = W[idx]
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    return float(roc_auc_score(y_true, y_scores))


def _build_ensemble(layer_scores: dict, include_spearman: bool) -> np.ndarray:
    """Min-max normalise selected layers and return their equal-weight mean."""
    names = (
        ["dm", "spearman", "proportionality", "copula"]
        if include_spearman
        else ["dm", "proportionality", "copula"]
    )
    normed = [_normalize_score(layer_scores[n]) for n in names]
    return sum(normed) / len(names)


def run_sensitivity(
    counts: np.ndarray,
    truth_binary: np.ndarray,
    label: str,
) -> dict:
    """Run the pipeline once, then sweep c_ref thresholds.

    For each c_ref value the Spearman-inclusion criterion is re-evaluated:
        spearman_reliable = (alpha_per_taxon >= c_ref) AND (zero_frac < 0.5)
    The appropriate subset of layers is then combined with equal weights
    and scored against the ground truth.

    Additionally reports AUROC when Spearman is force-included and
    force-excluded, to show the actual impact of the inclusion decision.
    """
    print(f"\n{'='*60}")
    print(f"  Running AdaCoNet on {label}")
    print(f"{'='*60}")

    pipe = AdaCoNetPipeline(n_folds=3, n_subsamples_stars=10, verbose=True)
    pipe.fit(counts)

    layer_scores = pipe._layer_scores
    alpha_pt = float(pipe._alpha_per_taxon)
    zero_frac = float(pipe._zero_frac)
    kept_mask = pipe._kept_mask

    # Subset ground truth to taxa that survived filtering
    gt = truth_binary[kept_mask][:, kept_mask]

    print(f"  alpha_per_taxon = {alpha_pt:.6f}")
    print(f"  zero_frac       = {zero_frac:.4f}")
    print(f"  taxa kept       = {kept_mask.sum()} / {len(kept_mask)}")

    # --- Force-include / force-exclude comparison ---
    W_with = _build_ensemble(layer_scores, include_spearman=True)
    W_without = _build_ensemble(layer_scores, include_spearman=False)
    auroc_with = _compute_auroc(W_with, gt)
    auroc_without = _compute_auroc(W_without, gt)

    print(f"  AUROC (4 layers, Spearman included) : {auroc_with:.4f}")
    print(f"  AUROC (3 layers, Spearman excluded)  : {auroc_without:.4f}")
    print(f"  Delta (with - without)               : {auroc_with - auroc_without:+.4f}")

    # --- c_ref sweep (standard pipeline logic: alpha_pt >= c_ref AND zero_frac < 0.5) ---
    sweep_results = {}
    for c_ref in C_REF_VALUES:
        spearman_ok = bool((alpha_pt >= c_ref) and (zero_frac < 0.5))
        layer_names = (
            ["dm", "spearman", "proportionality", "copula"]
            if spearman_ok
            else ["dm", "proportionality", "copula"]
        )
        W = _build_ensemble(layer_scores, include_spearman=spearman_ok)
        auroc = _compute_auroc(W, gt)
        sweep_results[str(c_ref)] = {
            "auroc": round(auroc, 6),
            "spearman_included": spearman_ok,
            "layers": layer_names,
            "n_layers": len(layer_names),
        }

    # --- c_ref sweep IGNORING zero_frac (isolates c_ref effect) ---
    sweep_cref_only = {}
    for c_ref in C_REF_VALUES:
        spearman_ok = bool(alpha_pt >= c_ref)
        layer_names = (
            ["dm", "spearman", "proportionality", "copula"]
            if spearman_ok
            else ["dm", "proportionality", "copula"]
        )
        W = _build_ensemble(layer_scores, include_spearman=spearman_ok)
        auroc = _compute_auroc(W, gt)
        sweep_cref_only[str(c_ref)] = {
            "auroc": round(auroc, 6),
            "spearman_included": spearman_ok,
            "layers": layer_names,
            "n_layers": len(layer_names),
        }

    return {
        "alpha_per_taxon": alpha_pt,
        "zero_frac": zero_frac,
        "taxa_kept": int(kept_mask.sum()),
        "taxa_total": len(kept_mask),
        "auroc_with_spearman": round(auroc_with, 6),
        "auroc_without_spearman": round(auroc_without, 6),
        "c_ref_sweep": sweep_results,
        "c_ref_sweep_ignore_zero_frac": sweep_cref_only,
    }


def print_summary(all_results: dict) -> None:
    """Print a formatted summary table per dataset."""
    for dataset, res in all_results.items():
        print(f"\n{'='*70}")
        print(f"  {dataset}")
        print(f"  alpha_per_taxon={res['alpha_per_taxon']:.4f}  "
              f"zero_frac={res['zero_frac']:.4f}  "
              f"taxa={res['taxa_kept']}/{res['taxa_total']}")
        print(f"  AUROC with Spearman   : {res['auroc_with_spearman']:.4f}")
        print(f"  AUROC without Spearman: {res['auroc_without_spearman']:.4f}")
        delta = res['auroc_with_spearman'] - res['auroc_without_spearman']
        print(f"  Delta                 : {delta:+.4f}")
        print(f"{'='*70}")

        # Standard sweep (alpha_pt >= c_ref AND zero_frac < 0.5)
        print(f"\n  --- Standard sweep (c_ref AND zero_frac < 0.5) ---")
        header = f"  {'c_ref':>6s}  {'AUROC':>8s}  {'Spearman':>10s}  {'Layers'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for c_ref in C_REF_VALUES:
            r = res["c_ref_sweep"][str(c_ref)]
            sp = "Yes" if r["spearman_included"] else "No"
            layers = ", ".join(r["layers"])
            print(f"  {c_ref:6.2f}  {r['auroc']:8.4f}  {sp:>10s}  {layers}")

        # c_ref-only sweep (ignores zero_frac)
        print(f"\n  --- c_ref only (ignoring zero_frac guard) ---")
        header2 = f"  {'c_ref':>6s}  {'AUROC':>8s}  {'Spearman':>10s}  {'Layers'}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for c_ref in C_REF_VALUES:
            r = res["c_ref_sweep_ignore_zero_frac"][str(c_ref)]
            sp = "Yes" if r["spearman_included"] else "No"
            layers = ", ".join(r["layers"])
            print(f"  {c_ref:6.2f}  {r['auroc']:8.4f}  {sp:>10s}  {layers}")


def main() -> None:
    all_results: dict = {}

    # ---------------------------------------------------------------
    # 1. v4 simulator: N=500, P=500, seed=0
    # ---------------------------------------------------------------
    rng_gt = np.random.default_rng(0)
    true_adj = _generate_ground_truth_network(500, density=0.1, rng=rng_gt)

    rng_data = np.random.default_rng(0)
    counts_v4 = _generate_compositional_data(500, 500, true_adj, rng=rng_data)

    truth_binary_v4 = (true_adj > 0).astype(float)

    all_results["v4_N500P500"] = run_sensitivity(
        counts_v4, truth_binary_v4, "v4 Simulator (N=500, P=500)"
    )

    # ---------------------------------------------------------------
    # 2. SparseDOSSA2: N=500 (Stool)
    # ---------------------------------------------------------------
    sd2_path = os.path.join(BASE_DIR, "data", "simulated", "sparsedossa2")
    counts_df = pd.read_csv(
        os.path.join(sd2_path, "Stool332_N500_counts.csv"), index_col=0
    )
    truth_df = pd.read_csv(
        os.path.join(sd2_path, "Stool332_N500_truth.csv"), index_col=0
    )

    counts_np = counts_df.values
    truth_aligned = truth_df.loc[counts_df.columns, counts_df.columns].values
    truth_binary_sd2 = (np.abs(truth_aligned) > 0.1).astype(float)

    all_results["SD2_N500"] = run_sensitivity(
        counts_np, truth_binary_sd2, "SparseDOSSA2 Stool (N=500)"
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    print_summary(all_results)

    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sensitivity_cref.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
