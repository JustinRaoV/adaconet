#!/usr/bin/env python3
"""
Independent benchmark using SparseDOSSA2 simulated data.
SparseDOSSA2 uses a zero-inflated truncated log-normal + Gaussian copula model,
completely different from our v4 simulator (MVN → exp → multinomial).

Ground truth: empirical Spearman correlation of absolute abundances.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_benchmarks import (
    _run_adaconet, _run_correlation, _run_sparcc,
    _run_cclasso, _run_rebacca, _run_fastSpar,
    _run_glasso, _run_spiecasi, _run_proportionality,
)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "simulated", "sparsedossa2")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results", "sparsedossa2_results.json")

configs = [
    ("Stool332_N200", 200),
    ("Stool332_N500", 500),
]

def evaluate(scores, true_corr, true_edges):
    """Compute AUROC, AUPRC, F1 from score matrix against ground truth."""
    iu = np.triu_indices(scores.shape[0], k=1)
    pred = np.abs(scores[iu])
    binary = true_edges[iu]
    auroc = roc_auc_score(binary, pred)
    auprc = average_precision_score(binary, pred)
    prec, rec, thr = precision_recall_curve(binary, pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    f1 = float(np.max(f1s))
    return auroc, auprc, f1

all_results = []

for label, n_expected in configs:
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"{'='*60}")
    
    counts_file = os.path.join(DATA_DIR, f"{label}_counts.csv")
    truth_file = os.path.join(DATA_DIR, f"{label}_truth.csv")
    
    counts_df = pd.read_csv(counts_file)
    truth_df = pd.read_csv(truth_file, index_col=0)
    
    counts = counts_df.values.astype(int)
    true_corr = truth_df.values.astype(float)
    
    n, p = counts.shape
    print(f"Data shape: {n} x {p}, zero frac: {np.mean(counts == 0)*100:.1f}%")
    
    # Ground truth: binary edges
    edge_thresh = 0.1
    true_edges = np.abs(true_corr) > edge_thresh
    np.fill_diagonal(true_edges, False)
    n_true_edges = true_edges.sum() // 2
    n_pairs = p * (p - 1) // 2
    print(f"True edges (|r| > {edge_thresh}): {n_true_edges} / {n_pairs} ({n_true_edges/n_pairs*100:.1f}%)")
    
    iu = np.triu_indices(p, k=1)
    results = {}
    
    methods = {
        "AdaCoNet": lambda c: _run_adaconet(c),
        "SparCC": lambda c: _run_sparcc(c, n_boot=10 if p > 200 else 20),
        "CCLasso": lambda c: _run_cclasso(c, n_boot=10),
        "REBACCA": lambda c: _run_rebacca(c, n_boot=5),
        "FastSpar": lambda c: _run_fastSpar(c, n_boot=10, n_iter=5),
        "Spearman CLR": lambda c: _run_correlation(c),
        "Proportionality": lambda c: _run_proportionality(c),
        "Glasso": lambda c: _run_glasso(c),
        "SPIEC-EASI": lambda c: _run_spiecasi(c),
    }
    
    for method_name, method_fn in methods.items():
        print(f"\n--- {method_name} ---")
        t0 = time.time()
        try:
            scores = method_fn(counts)
            dt = time.time() - t0
            auroc, auprc, f1 = evaluate(scores, true_corr, true_edges)
            results[method_name] = {"AUROC": auroc, "AUPRC": auprc, "F1": f1, "time": dt}
            print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  F1={f1:.3f}  time={dt:.2f}s")
        except Exception as e:
            dt = time.time() - t0
            results[method_name] = {"AUROC": 0, "AUPRC": 0, "F1": 0, "time": dt, "error": str(e)}
            print(f"  ERROR: {e}")
    
    all_results.append({
        "config": label,
        "p": p,
        "n": n,
        "true_edges": n_true_edges,
        "methods": results
    })

# Save results
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print(f"\n\nResults saved to {RESULTS_FILE}")

# Summary table
print("\n" + "="*80)
print("SUMMARY: SparseDOSSA2 Independent Benchmark")
print("Ground truth: empirical Spearman correlation of absolute abundances")
print("Data generator: SparseDOSSA2 (zero-inflated truncated log-normal + Gaussian copula)")
print("="*80)
for cfg_result in all_results:
    print(f"\n{cfg_result['config']} (p={cfg_result['p']}, n={cfg_result['n']}, edges={cfg_result['true_edges']})")
    print(f"{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Time':>10}")
    print("-" * 60)
    for method, metrics in sorted(cfg_result['methods'].items()):
        err = " *" if "error" in metrics else ""
        print(f"{method:<20} {metrics['AUROC']:>8.3f} {metrics['AUPRC']:>8.3f} {metrics['F1']:>8.3f} {metrics['time']:>10.2f}s{err}")
