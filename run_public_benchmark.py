#!/usr/bin/env python3
"""Run benchmarks on real public microbiome datasets.

Datasets:
  - enterotype: N=280, P=553 (human gut, Arumugam et al. 2011)
  - MovingPictures: N=1967, P=926 (human microbiome, Caporaso et al. 2011)

For simulated data: accuracy metrics (AUROC, AUPRC, F1).
For real data: network topology metrics (modularity, CC, degree) + runtime.

Usage:
    python run_public_benchmark.py --dataset enterotype
    python run_public_benchmark.py --dataset moving_pictures
    python run_public_benchmark.py --dataset both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "public")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(name: str) -> Tuple[np.ndarray, str]:
    """Load a public dataset, return (count_matrix, dataset_name).

    Pre-filters: remove samples (rows) and taxa (columns) with zero total reads.
    """
    if name == "enterotype":
        path = os.path.join(DATA_DIR, "enterotype.csv")
    elif name == "moving_pictures":
        path = os.path.join(DATA_DIR, "moving_pictures_filt.csv")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = pd.read_csv(path, index_col=0)
    counts = df.values.astype(int)
    _log(f"Loaded {name}: {counts.shape[0]} samples x {counts.shape[1]} taxa")

    # Pre-filter: remove zero-read samples and zero-read taxa
    keep_rows = counts.sum(axis=1) > 0
    keep_cols = counts.sum(axis=0) > 0
    n_rm_r, n_rm_c = (~keep_rows).sum(), (~keep_cols).sum()
    counts = counts[keep_rows][:, keep_cols]
    if n_rm_r > 0 or n_rm_c > 0:
        _log(f"  Filtered: removed {n_rm_r} empty samples, {n_rm_c} empty taxa")

    _log(f"  After filter: {counts.shape[0]} samples x {counts.shape[1]} taxa")
    _log(f"  Zero fraction: {np.mean(counts == 0):.1%}")
    _log(f"  Mean library size: {counts.sum(axis=1).mean():.0f}")
    label = f"{name} (N={counts.shape[0]}, P={counts.shape[1]})"
    return counts, label


# ---------------------------------------------------------------------------
# Method wrappers (import from run_benchmarks)
# ---------------------------------------------------------------------------
def _import_methods():
    """Import method functions from run_benchmarks."""
    from run_benchmarks import (
        _run_adaconet,
        _run_correlation,
        _run_sparcc,
        _run_glasso,
        _run_spiecasi,
        _run_proportionality,
    )
    return {
        "AdaCoNet": _run_adaconet,
        "Spearman CLR": _run_correlation,
        "SparCC": _run_sparcc,
        "Graphical Lasso": _run_glasso,
        "SPIEC-EASI": _run_spiecasi,
        "Proportionality": _run_proportionality,
    }


# ---------------------------------------------------------------------------
# Network topology analysis
# ---------------------------------------------------------------------------
def analyze_network(
    score_matrix: np.ndarray,
    top_k_frac: float = 0.05,
) -> Dict[str, float]:
    """Analyze network topology from a score matrix.

    Binarizes at top_k_frac of all possible edges, then computes
    modularity, connected components, and degree statistics.
    """
    p = score_matrix.shape[0]
    n_possible = p * (p - 1) // 2
    top_k = int(top_k_frac * n_possible)

    # Get upper-tri scores and threshold
    idx = np.triu_indices(p, k=1)
    scores = score_matrix[idx]
    if top_k > 0 and top_k < len(scores):
        threshold = np.sort(scores)[::-1][top_k - 1]
    else:
        threshold = 0.0

    # Build adjacency
    adj = np.zeros((p, p), dtype=int)
    edges = np.argwhere(score_matrix >= threshold)
    for i, j in edges:
        if i != j:
            adj[i, j] = 1
            adj[j, i] = 1

    # Degree statistics
    degrees = adj.sum(axis=1)
    n_edges = adj.sum() // 2

    # Connected components (BFS)
    visited = np.zeros(p, dtype=bool)
    components = []
    for start in range(p):
        if visited[start]:
            continue
        # BFS
        queue = [start]
        visited[start] = True
        cc_size = 0
        while queue:
            node = queue.pop(0)
            cc_size += 1
            neighbors = np.where(adj[node] > 0)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(cc_size)

    components.sort(reverse=True)
    max_cc = components[0] if components else 0
    n_components = len(components)

    # Modularity (greedy community detection — Louvain-like via label propagation)
    modularity = _compute_modularity(adj)

    # Network density
    density = n_edges / n_possible if n_possible > 0 else 0

    return {
        "n_edges": n_edges,
        "density": density,
        "max_degree": int(degrees.max()) if len(degrees) > 0 else 0,
        "mean_degree": float(degrees.mean()) if len(degrees) > 0 else 0,
        "max_cc": max_cc,
        "n_components": n_components,
        "modularity": modularity,
    }


def _compute_modularity(adj: np.ndarray) -> float:
    """Compute Newman-Girvan modularity Q using label propagation.

    Simple but effective community detection: iteratively assign each node
    to the community that maximizes local modularity gain.
    """
    n = adj.shape[0]
    m = adj.sum() / 2.0
    if m == 0:
        return 0.0

    k = adj.sum(axis=1).astype(float)  # degrees

    # Initialize: each node in its own community
    labels = np.arange(n)

    # Label propagation iterations
    for _ in range(20):
        changed = False
        order = np.random.permutation(n)
        for node in order:
            current_label = labels[node]
            # Find neighbor communities
            neighbors = np.where(adj[node] > 0)[0]
            if len(neighbors) == 0:
                continue

            neighbor_labels = labels[neighbors]
            unique_labels = np.unique(neighbor_labels)

            best_label = current_label
            best_gain = 0.0

            for label in unique_labels:
                if label == current_label:
                    continue
                # Modularity gain of moving node to 'label'
                # Sum of edges to community 'label'
                in_label = np.sum(adj[node, labels == label])
                in_current = np.sum(adj[node, labels == current_label])
                # Sum of degrees in each community
                k_label = k[labels == label].sum()
                k_current = k[labels == current_label].sum()
                k_node = k[node]

                gain = (in_label - in_current) / (2 * m) - \
                       k_node * (k_label - k_current + k_node) / (4 * m * m)

                if gain > best_gain:
                    best_gain = gain
                    best_label = label

            if best_label != current_label:
                labels[node] = best_label
                changed = True

        if not changed:
            break

    # Compute Q
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj[i, j] - k[i] * k[j] / (2 * m)
    Q /= (2 * m)

    return float(Q)


# ---------------------------------------------------------------------------
# Simulated benchmark (for accuracy metrics)
# ---------------------------------------------------------------------------
def run_simulated_benchmark(
    n_samples: int,
    n_taxa: int,
    methods: Dict[str, Any],
    n_seeds: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulated benchmark at given N, P dimensions."""
    from run_benchmarks import (
        _generate_ground_truth_network,
        _generate_compositional_data,
        compute_metrics,
    )

    metric_rows = []
    perf_rows = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(42 + seed)
        true_adj = _generate_ground_truth_network(n_taxa, density=0.08, rng=rng)
        counts = _generate_compositional_data(n_samples, n_taxa, true_adj, rng=rng)

        n_edges = true_adj.sum() // 2
        _log(f"  Seed {seed}: N={n_samples}, P={n_taxa}, GT edges={n_edges}")

        for name, func in methods.items():
            _log(f"    Running {name}...")
            t0 = time.perf_counter()
            try:
                scores = func(counts)
                wall = time.perf_counter() - t0
                metrics = compute_metrics(true_adj, scores)
                _log(f"    {name}: AUROC={metrics['auroc']:.3f} "
                     f"AUPRC={metrics['auprc']:.3f} F1={metrics['f1']:.3f} "
                     f"({wall:.1f}s)")
                metric_rows.append({
                    "method": name,
                    "seed": seed,
                    "n_samples": n_samples,
                    "n_taxa": n_taxa,
                    **metrics,
                })
                perf_rows.append({
                    "method": name,
                    "seed": seed,
                    "n_samples": n_samples,
                    "n_taxa": n_taxa,
                    "wall_time_sec": wall,
                })
            except Exception as e:
                _log(f"    [ERROR] {name}: {e}")
                traceback.print_exc()

    return pd.DataFrame(metric_rows), pd.DataFrame(perf_rows)


# ---------------------------------------------------------------------------
# Real data benchmark
# ---------------------------------------------------------------------------
def run_real_benchmark(
    counts: np.ndarray,
    dataset_name: str,
    methods: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Run all methods on real data, compute network topology metrics."""
    topology_rows = []
    score_matrices = {}

    for name, func in methods.items():
        _log(f"  Running {name}...")
        t0 = time.perf_counter()
        try:
            scores = func(counts)
            wall = time.perf_counter() - t0

            # Network topology at top 5%
            topo = analyze_network(scores, top_k_frac=0.05)
            topo["method"] = name
            topo["wall_time_sec"] = wall
            topo["dataset"] = dataset_name
            topology_rows.append(topo)

            score_matrices[name] = scores
            _log(f"  {name}: modularity={topo['modularity']:.3f} "
                 f"maxCC={topo['max_cc']} maxDeg={topo['max_degree']} "
                 f"({wall:.2f}s)")
        except Exception as e:
            _log(f"  [ERROR] {name}: {e}")
            traceback.print_exc()

    return pd.DataFrame(topology_rows), score_matrices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["enterotype", "moving_pictures", "both"],
        default="enterotype",
    )
    parser.add_argument("--skip-simulated", action="store_true")
    parser.add_argument("--skip-sparcc", action="store_true",
                        help="Skip SparCC (very slow at P>500)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    methods = _import_methods()

    # Optionally skip SparCC for large datasets
    if args.skip_sparcc and "SparCC" in methods:
        _log("Skipping SparCC (too slow for P>500)")
        del methods["SparCC"]

    all_metrics = []
    all_perf = []
    all_topology = []

    # ------------------------------------------------------------------
    # Simulated benchmarks (matching dimensions of real datasets)
    # ------------------------------------------------------------------
    if not args.skip_simulated:
        _log("=" * 60)
        _log("SIMULATED BENCHMARKS")
        _log("=" * 60)

        configs = [
            (280, 553),   # matching enterotype
        ]
        if args.dataset in ("moving_pictures", "both"):
            configs.append((500, 926))  # subsample of MovingPictures

        for n_s, n_t in configs:
            _log(f"\n--- Simulated: N={n_s}, P={n_t} ---")
            m_df, p_df = run_simulated_benchmark(n_s, n_t, methods, n_seeds=3)
            all_metrics.append(m_df)
            all_perf.append(p_df)

    # ------------------------------------------------------------------
    # Real data benchmarks
    # ------------------------------------------------------------------
    _log("=" * 60)
    _log("REAL DATA BENCHMARKS")
    _log("=" * 60)

    datasets = []
    if args.dataset in ("enterotype", "both"):
        datasets.append("enterotype")
    if args.dataset in ("moving_pictures", "both"):
        datasets.append("moving_pictures")

    for ds_name in datasets:
        _log(f"\n--- {ds_name} ---")
        counts, label = load_dataset(ds_name)
        topo_df, score_mats = run_real_benchmark(counts, label, methods)
        all_topology.append(topo_df)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if all_metrics:
        metric_df = pd.concat(all_metrics, ignore_index=True)
        metric_df.to_csv(os.path.join(RESULTS_DIR, "public_metrics.csv"), index=False)
        _log(f"\nSaved public_metrics.csv ({len(metric_df)} rows)")

        # Print summary
        _log("\n=== Simulated Accuracy Summary ===")
        summary = (
            metric_df
            .groupby(["method", "n_samples", "n_taxa"])
            .agg({"auroc": ["mean", "std"], "auprc": ["mean", "std"], "f1": ["mean", "std"]})
            .round(3)
        )
        print(summary.to_string())

    if all_perf:
        perf_df = pd.concat(all_perf, ignore_index=True)
        perf_df.to_csv(os.path.join(RESULTS_DIR, "public_performance.csv"), index=False)

    if all_topology:
        topo_df = pd.concat(all_topology, ignore_index=True)
        topo_df.to_csv(os.path.join(RESULTS_DIR, "public_topology.csv"), index=False)
        _log(f"\nSaved public_topology.csv ({len(topo_df)} rows)")

        _log("\n=== Real Data Topology Summary ===")
        print(topo_df[["dataset", "method", "n_edges", "modularity",
                        "max_cc", "max_degree", "mean_degree", "wall_time_sec"]]
              .to_string(index=False))

    # Save results as JSON for easy figure generation
    results = {}
    if all_metrics:
        results["simulated_metrics"] = metric_df.to_dict(orient="records")
    if all_perf:
        results["simulated_performance"] = perf_df.to_dict(orient="records")
    if all_topology:
        results["real_topology"] = topo_df.to_dict(orient="records")

    with open(os.path.join(RESULTS_DIR, "public_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _log("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()
