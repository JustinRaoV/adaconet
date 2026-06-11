#!/usr/bin/env python3
"""StARS subsample stability analysis (reviewer S5).

Tests how the number of StARS subsamples affects edge selection stability.
For each dataset, we run AdaCoNet once to get layer scores, then vary
n_subsamples from 5 to 50 and measure:
  - Selected threshold tau
  - Number of edges
  - Edge set Jaccard similarity across repeated StARS runs
"""
import os, sys, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from adaconet import AdaCoNetPipeline
from adaconet.ensemble import AdaptiveEnsemble
from adaconet.utils import normalize_scores, symmetrize

from run_benchmarks import (
    _generate_ground_truth_network,
    _generate_compositional_data,
)

SUBSAMPLE_VALUES = [5, 10, 15, 20, 30, 50]
N_REPEATS = 5  # repeat StARS with different random seeds per n_subsamples


def run_stars_with_n_subsamples(
    layer_scores, theory_weights, X_filt, spearman_reliable,
    n_subsamples, stars_seed
):
    """Run StARS threshold selection with given n_subsamples and seed."""
    ensemble = AdaptiveEnsemble(n_folds=3, n_subsamples=n_subsamples)
    n = X_filt.shape[0]
    p = X_filt.shape[1]

    S_dm = layer_scores["dm"]
    S_copula = layer_scores["copula"]

    scores_dict = {
        "dm": S_dm,
        "proportionality": layer_scores["proportionality"],
        "copula": S_copula,
    }
    if spearman_reliable:
        scores_dict["spearman"] = layer_scores["spearman"]

    scores_norm = ensemble.normalize_scores(scores_dict)
    names = sorted(scores_norm.keys())
    weight_vec = np.array([theory_weights.get(nm, 1.0) for nm in names])
    weight_vec = weight_vec / weight_vec.sum()

    W = ensemble.compute_final_score(scores_norm, weight_vec)

    # StARS subsampling
    stars_rng = np.random.default_rng(seed=stars_seed)
    sub_rate = 0.8
    sub_corr_list = []

    for b in range(n_subsamples):
        sub_size = max(int(n * sub_rate), 10)
        sub_idx = stars_rng.choice(n, size=sub_size, replace=False)
        X_sub = X_filt[sub_idx]

        # Recompute Spearman + Prop on subsample (same as pipeline)
        from scipy.stats import rankdata
        n_sub = X_sub.shape[0]
        np_sub_ratio = n_sub / p

        if np_sub_ratio > 2.0:
            # Not recomputing DM (expensive), use full-data scores
            pass

        # Proportionality on subsample
        X_prop = X_sub.astype(np.float64) + 0.5
        rel = X_prop / X_prop.sum(axis=1, keepdims=True)
        Z = np.log(rel) - np.log(rel).mean(axis=1, keepdims=True)
        var_z = Z.var(axis=0, ddof=1)
        Z_c = Z - Z.mean(axis=0, keepdims=True)
        cov = (Z_c.T @ Z_c) / (n_sub - 1)
        vlr = var_z[:, None] + var_z[None, :] - 2.0 * cov
        denom = np.maximum(var_z[:, None] + var_z[None, :], 1e-15)
        rho_p = 1.0 - vlr / denom
        np.clip(rho_p, -1, 1, out=rho_p)
        np.fill_diagonal(rho_p, 0)
        S_prop_sub = np.abs(rho_p)

        sub_scores = {
            "dm": S_dm,
            "proportionality": S_prop_sub,
            "copula": S_copula,
        }
        if spearman_reliable:
            # Spearman on subsample
            X_f = X_sub.astype(np.float64) + 0.5
            rel_f = X_f / X_f.sum(axis=1, keepdims=True)
            Z_clr = np.log(rel_f) - np.log(rel_f).mean(axis=1, keepdims=True)
            Z_clr = (Z_clr - Z_clr.mean(axis=0, keepdims=True)) / np.maximum(
                Z_clr.std(axis=0, keepdims=True), 1e-10
            )
            ranked = np.apply_along_axis(rankdata, 0, Z_clr)
            S_spear_sub = np.abs(np.corrcoef(ranked, rowvar=False))
            np.fill_diagonal(S_spear_sub, 0)
            sub_scores["spearman"] = S_spear_sub

        sub_norm = ensemble.normalize_scores(sub_scores)
        sub_names = sorted(sub_norm.keys())
        sub_wv = np.array([theory_weights.get(nm, 1.0) for nm in sub_names])
        sub_wv = sub_wv / sub_wv.sum()
        sub_mats = [sub_norm[nm] for nm in sub_names]
        W_sub = np.tensordot(sub_wv, np.stack(sub_mats), axes=([0], [0]))
        W_sub = normalize_scores(W_sub)
        W_sub = symmetrize(W_sub, method="max")
        sub_corr_list.append(W_sub)

    # Select tau
    off_diag = W[np.triu_indices(p, k=1)]
    tau_min = max(off_diag.min(), 0.0)
    tau_max = off_diag.max()
    if tau_max - tau_min < 1e-10:
        tau = float(tau_min)
    else:
        tau_grid = np.linspace(tau_min, tau_max, 20)
        n_edges = p * (p - 1) // 2
        triu_idx = np.triu_indices(p, k=1)
        instability = np.zeros(len(tau_grid))
        for t_idx, tau_cand in enumerate(tau_grid):
            edge_counts = np.zeros(n_edges)
            for W_sub in sub_corr_list:
                adj_sub = (W_sub >= tau_cand).astype(float)
                np.fill_diagonal(adj_sub, 0)
                adj_sub = np.maximum(adj_sub, adj_sub.T)
                edge_counts += adj_sub[triu_idx]
            edge_probs = edge_counts / n_subsamples
            edge_inst = 2.0 * edge_probs * (1.0 - edge_probs)
            instability[t_idx] = edge_inst.mean()
        best_idx = int(np.argmin(instability))
        tau = float(tau_grid[best_idx])
    tau = max(tau, 0.05)

    # Build edge set
    edges = set()
    triu_i, triu_j = np.triu_indices(p, k=1)
    for idx in range(len(triu_i)):
        if W[triu_i[idx], triu_j[idx]] >= tau:
            edges.add((triu_i[idx], triu_j[idx]))

    return tau, len(edges), edges, float(np.min(instability) if tau_max - tau_min > 1e-10 else 0)


def analyze_stability(label, counts):
    """Run stability analysis on a dataset."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    pipe = AdaCoNetPipeline(n_folds=3, n_subsamples_stars=10, verbose=True)
    pipe.fit(counts)

    layer_scores = pipe._layer_scores
    tw = pipe._theory_weights
    X_filt = pipe._X_filtered
    alpha_p = pipe._alpha_per_taxon
    zf = pipe._zero_frac
    spearman_ok = alpha_p >= 0.05 and zf < 0.5

    print(f"  alpha/p = {alpha_p:.4f}, zero_frac = {zf:.3f}")
    print(f"  Spearman: {'included' if spearman_ok else 'excluded'}")

    results = {}
    for n_sub in SUBSAMPLE_VALUES:
        edge_sets = []
        taus = []
        n_edges_list = []
        min_instabilities = []

        for rep in range(N_REPEATS):
            tau, n_e, edges, min_inst = run_stars_with_n_subsamples(
                layer_scores, tw, X_filt, spearman_ok,
                n_subsamples=n_sub, stars_seed=42 + rep * 100 + n_sub
            )
            edge_sets.append(edges)
            taus.append(tau)
            n_edges_list.append(n_e)
            min_instabilities.append(min_inst)

        # Jaccard similarity across repeats
        jaccards = []
        for i in range(len(edge_sets)):
            for j in range(i + 1, len(edge_sets)):
                inter = len(edge_sets[i] & edge_sets[j])
                union = len(edge_sets[i] | edge_sets[j])
                if union > 0:
                    jaccards.append(inter / union)

        avg_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
        avg_tau = float(np.mean(taus))
        std_tau = float(np.std(taus))
        avg_edges = float(np.mean(n_edges_list))
        avg_inst = float(np.mean(min_instabilities))

        results[str(n_sub)] = {
            "n_subsamples": n_sub,
            "avg_tau": round(avg_tau, 4),
            "std_tau": round(std_tau, 4),
            "avg_edges": round(avg_edges, 1),
            "avg_jaccard": round(avg_jaccard, 4),
            "avg_instability": round(avg_inst, 6),
        }
        print(
            f"  n_sub={n_sub:2d}: tau={avg_tau:.4f}±{std_tau:.4f}, "
            f"edges={avg_edges:.0f}, Jaccard={avg_jaccard:.3f}, "
            f"instability={avg_inst:.5f}"
        )

    return results


def main():
    all_results = {}

    # v4 N=500, P=500
    rng_gt = np.random.default_rng(42)
    true_adj = _generate_ground_truth_network(500, density=0.1, rng=rng_gt)
    rng_data = np.random.default_rng(42)
    counts_v4 = _generate_compositional_data(500, 500, true_adj, rng=rng_data)
    all_results["v4_N500P500"] = analyze_stability(
        "v4 Simulator (N=500, P=500)", counts_v4
    )

    # SparseDOSSA2 N=500
    sd2_path = os.path.join(SCRIPT_DIR, "data", "simulated", "sparsedossa2")
    counts_df = pd.read_csv(os.path.join(sd2_path, "Stool332_N500_counts.csv"))
    counts_sd2 = counts_df.values.astype(int)
    all_results["SD2_N500"] = analyze_stability(
        "SparseDOSSA2 Stool (N=500)", counts_sd2
    )

    out_path = os.path.join(SCRIPT_DIR, "results", "stars_stability.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
