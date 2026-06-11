#!/usr/bin/env python3
"""Ablation study and AUPRC computation for AdaCoNet.

Runs two sets of experiments:

1. AUPRC: Full AdaCoNet + 8 baselines with both AUROC and AUPRC
2. Ablation: 9 variants of AdaCoNet to assess each component's contribution

Usage
-----
    python run_ablation.py --seeds 3
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from run_benchmarks import (
    _generate_ground_truth_network,
    _generate_compositional_data,
    _run_adaconet,
    _run_sparcc,
    _run_cclasso,
    _run_rebacca,
    _run_fastSpar,
    _run_glasso,
    _run_spiecasi,
    _run_proportionality,
    _run_correlation,
)

from sklearn.metrics import roc_auc_score, average_precision_score


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def compute_auroc_auprc(true_adj: np.ndarray, pred_scores: np.ndarray) -> dict:
    """Compute AUROC and AUPRC from score matrix vs ground truth."""
    iu = np.triu_indices_from(true_adj, k=1)
    y_true = true_adj[iu].astype(int)
    y_scores = np.abs(pred_scores[iu])

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auroc": 0.0, "auprc": 0.0}
    if np.all(y_scores == y_scores[0]):
        return {"auroc": 0.5, "auprc": float(y_true.mean())}

    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))
    return {"auroc": auroc, "auprc": auprc}


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

def run_ablation_variants(counts: np.ndarray) -> tuple:
    """Run AdaCoNet ablation variants.

    Returns (variants, kept_mask) where variants maps name -> score matrix
    in the FILTERED taxa space, and kept_mask is a boolean array of length p
    indicating which taxa survived filtering.
    """
    from adaconet import AdaCoNetPipeline
    from adaconet.ensemble import AdaptiveEnsemble
    from adaconet.utils import normalize_scores, symmetrize

    pipe = AdaCoNetPipeline(
        n_folds=3,
        n_subsamples_stars=10,
        verbose=False,
    )
    pipe.fit(counts)
    layer_scores = pipe._layer_scores  # in filtered taxa space
    alpha_p = pipe._alpha_per_taxon
    zero_frac = pipe._zero_frac
    spearman_reliable = alpha_p >= 0.05 and zero_frac < 0.5
    kept_mask = pipe._kept_mask

    variants = {}

    # Single layers
    for name in ["dm", "spearman", "proportionality", "copula"]:
        variants[name] = layer_scores[name].copy()

    # Full adaptive (same as main pipeline)
    variants["full_adaptive"] = pipe._W.copy()

    # Leave-one-out
    for exclude in ["dm", "spearman", "proportionality", "copula"]:
        included = [k for k in ["dm", "spearman", "proportionality", "copula"] if k != exclude]
        scores = {k: layer_scores[k] for k in included}
        scores_norm = {k: normalize_scores(v) for k, v in scores.items()}
        K = len(scores_norm)
        weights = np.ones(K) / K
        matrices = [scores_norm[k] for k in sorted(scores_norm.keys())]
        W = np.tensordot(weights, np.stack(matrices), axes=([0], [0]))
        W = normalize_scores(W)
        W = symmetrize(W, method="max")
        variants[f"no_{exclude}"] = W

    # Fixed 4-layer (always include Spearman, no adaptive exclusion)
    all_scores = {k: layer_scores[k] for k in ["dm", "spearman", "proportionality", "copula"]}
    all_norm = {k: normalize_scores(v) for k, v in all_scores.items()}
    K = 4
    weights = np.ones(K) / K
    matrices = [all_norm[k] for k in sorted(all_norm.keys())]
    W_fixed = np.tensordot(weights, np.stack(matrices), axes=([0], [0]))
    W_fixed = normalize_scores(W_fixed)
    W_fixed = symmetrize(W_fixed, method="max")
    variants["fixed_4layer"] = W_fixed

    return variants, kept_mask


def _embed_scores(filtered_scores: np.ndarray, kept_mask: np.ndarray, p: int) -> np.ndarray:
    """Embed filtered-taxon scores back into full p×p space (zeros for removed taxa)."""
    full = np.zeros((p, p))
    idx = np.where(kept_mask)[0]
    full[np.ix_(idx, idx)] = filtered_scores
    return full


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

METHODS_WITH_AUPRC = {
    "AdaCoNet": _run_adaconet,
    "SparCC": lambda c: _run_sparcc(c, n_boot=10),
    "REBACCA": _run_rebacca,
    "CCLasso": _run_cclasso,
    "FastSpar": _run_fastSpar,
    "Spearman": _run_correlation,
    "Proportionality": _run_proportionality,
    "Glasso": _run_glasso,
    "SPIEC-EASI": _run_spiecasi,
}


def run_v4_experiments(n_repeats: int) -> dict:
    """Run v4 benchmark with AUPRC + ablation."""
    configs = [
        (200, 50),
        (500, 200),
        (500, 500),
        (1000, 500),
        (1000, 1000),
    ]
    results = {}

    for n_samples, n_taxa in configs:
        _log(f"--- v4 N={n_samples}, P={n_taxa} ---")
        config_key = f"v4 N={n_samples},P={n_taxa}"
        config_results = {"methods": {}, "ablation": {}}

        method_auroc = {m: [] for m in METHODS_WITH_AUPRC}
        method_auprc = {m: [] for m in METHODS_WITH_AUPRC}
        method_time = {m: [] for m in METHODS_WITH_AUPRC}
        ablation_auroc = {v: [] for v in [
            "dm", "spearman", "proportionality", "copula",
            "no_dm", "no_spearman", "no_proportionality", "no_copula",
            "fixed_4layer", "full_adaptive",
        ]}

        for seed in range(n_repeats):
            rng = np.random.default_rng(42 + seed)
            true_adj = _generate_ground_truth_network(n_taxa, density=0.1, rng=rng)
            counts = _generate_compositional_data(n_samples, n_taxa, true_adj, rng=rng)

            _log(f"  Seed {seed}: data {counts.shape}, zero frac {np.mean(counts==0)*100:.1f}%")

            # Full methods with AUPRC
            for method_name, method_fn in METHODS_WITH_AUPRC.items():
                # Skip slow methods for P=1000
                if n_taxa >= 1000 and method_name in ["SparCC", "CCLasso", "FastSpar"]:
                    continue
                t0 = time.time()
                try:
                    scores = method_fn(counts)
                    dt = time.time() - t0
                    metrics = compute_auroc_auprc(true_adj, scores)
                    method_auroc[method_name].append(metrics["auroc"])
                    method_auprc[method_name].append(metrics["auprc"])
                    method_time[method_name].append(dt)
                except Exception as e:
                    _log(f"    {method_name} FAILED: {e}")

            # Ablation
            try:
                variants, kept_mask = run_ablation_variants(counts)
                p_full = counts.shape[1]
                for variant_name, W_filt in variants.items():
                    W_full = _embed_scores(W_filt, kept_mask, p_full)
                    m = compute_auroc_auprc(true_adj, W_full)
                    ablation_auroc[variant_name].append(m["auroc"])
            except Exception as e:
                _log(f"    Ablation FAILED: {e}")

        # Aggregate
        for method_name in method_auroc:
            if method_auroc[method_name]:
                config_results["methods"][method_name] = {
                    "auroc_mean": float(np.mean(method_auroc[method_name])),
                    "auroc_std": float(np.std(method_auroc[method_name])),
                    "auprc_mean": float(np.mean(method_auprc[method_name])),
                    "auprc_std": float(np.std(method_auprc[method_name])),
                    "time_mean": float(np.mean(method_time[method_name])),
                }

        for variant_name in ablation_auroc:
            if ablation_auroc[variant_name]:
                config_results["ablation"][variant_name] = {
                    "auroc_mean": float(np.mean(ablation_auroc[variant_name])),
                    "auroc_std": float(np.std(ablation_auroc[variant_name])),
                }

        results[config_key] = config_results

    return results


def run_sd2_experiments() -> dict:
    """Run SparseDOSSA2 benchmark with AUPRC + ablation."""
    import pandas as pd

    data_dir = os.path.join(SCRIPT_DIR, "data", "simulated", "sparsedossa2")
    configs = [
        ("Stool332_N200", "SD2 N200"),
        ("Stool332_N500", "SD2 N500"),
    ]
    results = {}

    for label, result_key in configs:
        counts_file = os.path.join(data_dir, f"{label}_counts.csv")
        truth_file = os.path.join(data_dir, f"{label}_truth.csv")

        if not os.path.exists(counts_file):
            _log(f"  SparseDOSSA2 data not found: {counts_file}")
            _log(f"  Run `Rscript run_sparsedossa2_benchmark.R` first")
            continue

        _log(f"--- {result_key} ({label}) ---")
        counts_df = pd.read_csv(counts_file)
        truth_df = pd.read_csv(truth_file, index_col=0)

        counts = counts_df.values.astype(int)
        true_corr = truth_df.values.astype(float)
        true_edges = np.abs(true_corr) > 0.1
        np.fill_diagonal(true_edges, False)

        n, p = counts.shape
        _log(f"  Data: {n} x {p}, zero frac {np.mean(counts==0)*100:.1f}%")
        _log(f"  True edges: {true_edges.sum()//2}")

        config_results = {"methods": {}, "ablation": {}}

        # Full methods with AUPRC
        for method_name, method_fn in METHODS_WITH_AUPRC.items():
            t0 = time.time()
            try:
                scores = method_fn(counts)
                dt = time.time() - t0
                # AdaCoNet may return filtered scores; embed to full space
                if scores.shape[0] != p:
                    from adaconet import AdaCoNetPipeline
                    # Re-run pipeline to get kept_mask
                    _pipe = AdaCoNetPipeline(n_folds=3, n_subsamples_stars=10, verbose=False)
                    _pipe.fit(counts)
                    scores = _embed_scores(scores, _pipe._kept_mask, p)
                metrics = compute_auroc_auprc(true_edges, scores)
                config_results["methods"][method_name] = {
                    "auroc_mean": metrics["auroc"],
                    "auprc_mean": metrics["auprc"],
                    "time_mean": dt,
                }
                _log(f"  {method_name}: AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f} ({dt:.1f}s)")
            except Exception as e:
                _log(f"  {method_name} FAILED: {e}")

        # Ablation
        try:
            variants, kept_mask = run_ablation_variants(counts)
            for variant_name, W_filt in variants.items():
                W_full = _embed_scores(W_filt, kept_mask, p)
                m = compute_auroc_auprc(true_edges, W_full)
                config_results["ablation"][variant_name] = {
                    "auroc_mean": m["auroc"],
                    "auprc_mean": m["auprc"],
                }
                _log(f"  Ablation {variant_name}: AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}")
        except Exception as e:
            _log(f"  Ablation FAILED: {e}")

        results[result_key] = config_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/ablation_auprc.json")
    parser.add_argument("--skip-v4", action="store_true")
    parser.add_argument("--skip-sd2", action="store_true")
    args = parser.parse_args()

    all_results = {}

    if not args.skip_v4:
        _log("=== v4 Direct-Covariance Benchmark ===")
        v4_results = run_v4_experiments(args.seeds)
        all_results.update(v4_results)

    if not args.skip_sd2:
        _log("=== SparseDOSSA2 Validation ===")
        sd2_results = run_sd2_experiments()
        all_results.update(sd2_results)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    _log(f"\nResults saved to {args.output}")

    # Summary
    _log("\n" + "="*80)
    _log("SUMMARY")
    _log("="*80)
    for config_key, config_data in all_results.items():
        _log(f"\n{config_key}:")
        if config_data.get("methods"):
            _log(f"  {'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'Time':>10}")
            for m, d in sorted(config_data["methods"].items()):
                _log(f"  {m:<20} {d['auroc_mean']:>8.3f} {d['auprc_mean']:>8.3f} {d['time_mean']:>10.2f}s")
        if config_data.get("ablation"):
            _log(f"  {'Ablation':<25} {'AUROC':>8}")
            for v, d in sorted(config_data["ablation"].items()):
                _log(f"  {v:<25} {d['auroc_mean']:>8.3f}")


if __name__ == "__main__":
    main()
