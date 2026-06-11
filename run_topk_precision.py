#!/usr/bin/env python3
"""Top-k precision and precision@FPR<5% analysis for AdaCoNet.

Addresses reviewer DA concern: AUROC may not reflect practical utility
for sparse networks where users only care about high-confidence edges.

Reports:
  - precision@k for k in {10, 50, 100, 500}
  - precision@FPR<5%
  - Compared across AdaCoNet, SparCC, REBACCA, Proportionality, Spearman CLR
"""
import os, sys, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from run_benchmarks import (
    _generate_ground_truth_network,
    _generate_compositional_data,
    _run_adaconet, _run_sparcc, _run_rebacca,
    _run_proportionality, _run_correlation,
)
from adaconet import AdaCoNetPipeline


def precision_at_k(y_true_flat, y_score_flat, k):
    """Precision among top-k predicted edges."""
    idx = np.argsort(-y_score_flat)[:k]
    return float(y_true_flat[idx].mean())


def precision_at_fpr(y_true_flat, y_score_flat, max_fpr=0.05):
    """Precision among predictions with FPR <= max_fpr.

    Find threshold where FPR = max_fpr, then report precision above it.
    """
    neg_mask = y_true_flat == 0
    pos_mask = y_true_flat == 1
    if neg_mask.sum() == 0 or pos_mask.sum() == 0:
        return 0.0
    neg_scores = y_score_flat[neg_mask]
    # Threshold: score above which FPR = max_fpr
    threshold = np.percentile(neg_scores, 100 * (1 - max_fpr))
    above = y_score_flat >= threshold
    if above.sum() == 0:
        return 0.0
    return float(y_true_flat[above].mean())


def compute_topk_metrics(true_adj, pred_scores):
    """Compute top-k precision metrics."""
    iu = np.triu_indices_from(true_adj, k=1)
    y_true = true_adj[iu].astype(int)
    y_scores = np.abs(pred_scores[iu])

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    results = {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_total": len(y_true),
    }

    for k in [10, 50, 100, 500]:
        if k <= len(y_true):
            results[f"prec@{k}"] = round(precision_at_k(y_true, y_scores, k), 4)

    for fpr in [0.01, 0.05, 0.10]:
        results[f"prec@FPR{int(fpr*100)}"] = round(
            precision_at_fpr(y_true, y_scores, max_fpr=fpr), 4
        )

    return results


METHODS = {
    "AdaCoNet": _run_adaconet,
    "SparCC": lambda c: _run_sparcc(c, n_boot=10),
    "REBACCA": _run_rebacca,
    "Proportionality": _run_proportionality,
    "Spearman_CLR": _run_correlation,
}


def main():
    results = {}

    # --- v4 N=500, P=500 ---
    print("=== v4 N=500, P=500 (seed=0) ===")
    rng = np.random.default_rng(42)
    true_adj = _generate_ground_truth_network(500, density=0.1, rng=rng)
    counts = _generate_compositional_data(500, 500, true_adj, rng=rng)
    truth_binary = (true_adj > 0).astype(float)

    v4_results = {}
    for name, fn in METHODS.items():
        try:
            scores = fn(counts)
            # AdaCoNet may return filtered
            if scores.shape[0] != 500:
                pipe = AdaCoNetPipeline(verbose=False)
                pipe.fit(counts)
                from run_ablation import _embed_scores
                scores = _embed_scores(scores, pipe._kept_mask, 500)
            m = compute_topk_metrics(truth_binary, scores)
            v4_results[name] = m
            print(f"  {name}: prec@50={m.get('prec@50','N/A')}, prec@FPR5={m.get('prec@FPR5','N/A')}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    results["v4_N500P500"] = v4_results

    # --- SparseDOSSA2 N=500 ---
    print("\n=== SparseDOSSA2 N=500 ===")
    sd2_dir = os.path.join(SCRIPT_DIR, "data", "simulated", "sparsedossa2")
    counts_df = pd.read_csv(os.path.join(sd2_dir, "Stool332_N500_counts.csv"))
    truth_df = pd.read_csv(os.path.join(sd2_dir, "Stool332_N500_truth.csv"), index_col=0)
    counts_sd2 = counts_df.values.astype(int)
    true_corr_sd2 = truth_df.values.astype(float)
    true_edges_sd2 = (np.abs(true_corr_sd2) > 0.1).astype(float)
    np.fill_diagonal(true_edges_sd2, 0)
    p_sd2 = counts_sd2.shape[1]

    sd2_results = {}
    for name, fn in METHODS.items():
        try:
            scores = fn(counts_sd2)
            if scores.shape[0] != p_sd2:
                pipe = AdaCoNetPipeline(verbose=False)
                pipe.fit(counts_sd2)
                from run_ablation import _embed_scores
                scores = _embed_scores(scores, pipe._kept_mask, p_sd2)
            m = compute_topk_metrics(true_edges_sd2, scores)
            sd2_results[name] = m
            print(f"  {name}: prec@50={m.get('prec@50','N/A')}, prec@FPR5={m.get('prec@FPR5','N/A')}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    results["SD2_N500"] = sd2_results

    # Save
    out_path = os.path.join(SCRIPT_DIR, "results", "topk_precision.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
