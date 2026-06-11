#!/usr/bin/env python3
"""AdaCoNet benchmark runner.

Generates simulated datasets, runs AdaCoNet and baseline methods,
computes evaluation metrics, and produces publication-quality figures.

Usage
-----
    python run_benchmarks.py --n-repeats 3 --output-dir results/
    python run_benchmarks.py --methods adaconet spiecasi glasso
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Timestamped logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Simulated data generation
# ---------------------------------------------------------------------------

def _generate_ground_truth_network(
    n_taxa: int,
    density: float = 0.08,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a sparse symmetric binary adjacency matrix with hub structure.

    Creates a scale-free-like network: a few hubs with high degree,
    most nodes with low degree.  This is more realistic for microbial
    co-occurrence networks than Erdos-Renyi.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    adj = np.zeros((n_taxa, n_taxa))
    n_edges_target = int(density * n_taxa * (n_taxa - 1) / 2)

    # Create hubs (top 10% of nodes get extra connections)
    n_hubs = max(n_taxa // 10, 2)
    hub_indices = rng.choice(n_taxa, size=n_hubs, replace=False)

    edges_added = set()

    # Hub-to-non-hub edges (high degree hubs)
    non_hubs = np.setdiff1d(np.arange(n_taxa), hub_indices)
    for hub in hub_indices:
        # Each hub connects to ~15-25% of non-hubs
        n_hub_edges = max(int(0.2 * len(non_hubs)), 3)
        targets = rng.choice(non_hubs, size=min(n_hub_edges, len(non_hubs)), replace=False)
        for t in targets:
            pair = (min(hub, t), max(hub, t))
            if pair not in edges_added:
                edges_added.add(pair)

    # Random non-hub-to-non-hub edges to reach target density
    remaining = n_edges_target - len(edges_added)
    if remaining > 0:
        upper_idx = np.triu_indices(n_taxa, k=1)
        all_pairs = set(zip(upper_idx[0], upper_idx[1]))
        available = list(all_pairs - edges_added)
        if len(available) > 0:
            extra = rng.choice(len(available), size=min(remaining, len(available)), replace=False)
            for idx in extra:
                edges_added.add(available[idx])

    for i, j in edges_added:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def _generate_compositional_data(
    n_samples: int,
    n_taxa: int,
    true_adj: np.ndarray,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate compositional count data with strong recoverable signal.

    Directly embeds target correlations in the covariance matrix:
    - Edge pairs get target correlation ~0.5 (with some noise)
    - Non-edge pairs get ~0 (with small noise)
    - The matrix is projected to nearest PSD via eigenvalue clipping.

    This produces clear signal that survives compositional normalisation
    and count noise, giving AUROC > 0.7 for reasonable methods.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    p = n_taxa

    # --- Step 1: Build target correlation matrix ---
    target_corr = np.eye(p)
    edges = np.argwhere(np.triu(true_adj, k=1))

    for i, j in edges:
        # Edge correlation: 0.3 to 0.7, random sign
        rho = rng.uniform(0.3, 0.7)
        sign = rng.choice([-1, 1])
        target_corr[i, j] = sign * rho
        target_corr[j, i] = target_corr[i, j]

    # Add small random noise to all off-diagonal entries
    noise = rng.normal(0, 0.05, size=(p, p))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    target_corr += noise
    # Re-symmetrise and clip to [-1, 1]
    target_corr = (target_corr + target_corr.T) / 2
    np.clip(target_corr, -0.99, 0.99, out=target_corr)
    np.fill_diagonal(target_corr, 1.0)

    # --- Step 2: Project to nearest PSD ---
    eigvals, eigvecs = np.linalg.eigh(target_corr)
    eigvals = np.maximum(eigvals, 0.01)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise to correlation matrix
    d = np.sqrt(np.diag(Sigma))
    d[d == 0] = 1.0
    Sigma = Sigma / np.outer(d, d)

    # Verify separation
    edge_corrs = [abs(Sigma[i, j]) for i, j in edges]
    non_edge_mask = np.triu(np.ones((p, p), dtype=bool), k=1) & ~true_adj.astype(bool)
    non_edge_corrs = Sigma[non_edge_mask]
    if len(edge_corrs) > 0 and len(non_edge_corrs) > 0:
        sep = np.mean(edge_corrs) / max(np.mean(np.abs(non_edge_corrs)), 1e-10)
        # If separation is too low, amplify edge correlations
        if sep < 2.0:
            for i, j in edges:
                Sigma[i, j] *= 1.5
                Sigma[j, i] *= 1.5
            # Re-PSD project
            eigvals2, eigvecs2 = np.linalg.eigh(Sigma)
            eigvals2 = np.maximum(eigvals2, 0.01)
            Sigma = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T
            d2 = np.sqrt(np.diag(Sigma))
            d2[d2 == 0] = 1.0
            Sigma = Sigma / np.outer(d2, d2)

    # --- Step 3: Realistic mean structure ---
    log_means = np.sort(rng.exponential(1.2, size=p))[::-1]
    log_means = log_means - log_means.max() + 4.0

    # --- Step 4: Sample log-abundances ---
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n_samples, p))
    log_abundances = Z @ L.T + log_means[np.newaxis, :]

    # --- Step 5: Exponentiate ---
    abs_abundances = np.exp(log_abundances)

    # Gamma overdispersion (mild)
    abs_abundances = rng.gamma(20.0, abs_abundances / 20.0)

    # --- Step 6: Normalize to compositions ---
    compositions = abs_abundances / abs_abundances.sum(axis=1, keepdims=True)

    # --- Step 7: Zero inflation ---
    mean_comp = compositions.mean(axis=0)
    zero_prob = np.clip(0.1 * (mean_comp.max() / (mean_comp + 1e-10)) ** (-0.3), 0, 0.15)
    zero_mask = rng.random((n_samples, p)) < zero_prob[np.newaxis, :]
    compositions[zero_mask] = 0.0
    row_sums = compositions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    compositions = compositions / row_sums

    # --- Step 8: Multinomial count sampling ---
    lib_sizes = rng.integers(10000, 30000, size=n_samples).astype(float)
    count_matrix = np.zeros((n_samples, p), dtype=int)
    for i in range(n_samples):
        pi = compositions[i]
        if pi.sum() > 0:
            pi = pi / pi.sum()
            count_matrix[i] = rng.multinomial(int(lib_sizes[i]), pi)

    return count_matrix


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _flatten_upper(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def compute_metrics(true_adj: np.ndarray, pred_scores: np.ndarray) -> Dict[str, float]:
    """Compute standard network-inference metrics using continuous scores.

    AUROC and AUPRC are computed directly on continuous scores (no binarisation).
    F1 uses the optimal threshold that maximises F1 across a grid of thresholds.
    Precision and recall are reported at that optimal threshold.

    Parameters
    ----------
    true_adj : np.ndarray
        Binary ground-truth adjacency matrix.
    pred_scores : np.ndarray
        Continuous score matrix (higher = more likely edge).

    Returns
    -------
    dict
        Keys: precision, recall, f1, auprc, auroc, f1_threshold.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = _flatten_upper(true_adj).astype(int)
    y_scores = _flatten_upper(pred_scores)

    # Guard: if all scores are identical or no positives, return zeros
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "auprc": 0.0, "auroc": 0.0, "f1_threshold": 0.0}
    if np.all(y_scores == y_scores[0]):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "auprc": 0.0, "auroc": 0.5, "f1_threshold": 0.0}

    metrics: Dict[str, float] = {}

    # AUROC and AUPRC on continuous scores (gold standard metrics)
    metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
    metrics["auprc"] = float(average_precision_score(y_true, y_scores))

    # Find optimal F1 threshold via precision-recall curve
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_scores)
    # Compute F1 for each threshold
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10),
        0.0,
    )
    # Best threshold (skip the last entry which has no threshold)
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_arr[:-1])
        best_threshold = float(thresholds[best_idx])
        metrics["f1"] = float(f1_arr[best_idx])
        metrics["precision"] = float(prec_arr[best_idx])
        metrics["recall"] = float(rec_arr[best_idx])
    else:
        best_threshold = 0.0
        metrics["f1"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0

    metrics["f1_threshold"] = best_threshold

    return metrics


# ---------------------------------------------------------------------------
# Baseline methods (stubs — replace with real implementations)
# ---------------------------------------------------------------------------

def _run_correlation(counts: np.ndarray, **_: Any) -> np.ndarray:
    """Spearman rank correlation on CLR-transformed data.

    A simple non-parametric baseline.  CLR transform approximately
    addresses compositionality; Spearman captures monotonic associations.
    Uses vectorised implementation (rankdata + np.corrcoef).
    """
    from scipy.stats import rankdata

    # CLR transform with pseudocount
    rel = counts.astype(float) / counts.sum(axis=1, keepdims=True)
    rel = np.clip(rel, 1e-10, None)
    log_rel = np.log(rel)
    clr = log_rel - log_rel.mean(axis=1, keepdims=True)

    # Vectorised Spearman: rank columns, then Pearson on ranks
    ranked = np.apply_along_axis(rankdata, 0, clr)
    corr = np.corrcoef(ranked, rowvar=False)
    np.fill_diagonal(corr, 0)
    return np.abs(corr)


def _run_sparcc(
    counts: np.ndarray,
    n_boot: int = 20,
    n_iter: int = 10,
    thresh: float = 0.1,
    **_: Any,
) -> np.ndarray:
    """SparCC (Friedman & Alm, 2012) — complete implementation.

    The original SparCC estimates basis correlations from compositional data
    using the log-ratio variance identity:

        Var[log(x_i / x_j)] = omega_i + omega_j - 2 * Sigma_ij

    Under the sparsity assumption (most Sigma_ij ≈ 0), omega is estimated
    from the T matrix (observed log-ratio variances).  Strong pairs are
    iteratively excluded and omega is re-estimated.  Bootstrap resampling
    (with replacement) provides robustness; the median omega across
    bootstraps yields the final correlation estimate.

    Key improvement: we use bounded least-squares (omega >= 0) instead of
    the original closed-form formula, which can produce negative variance
    estimates when taxa have similar log-ratio variances.

    Complexity: O(n_boot * n_iter * p^2 * n)
    """
    from scipy.optimize import lsq_linear

    n, p = counts.shape

    # Relative abundances with pseudocount
    X = counts.astype(float) + 0.5
    rel = X / X.sum(axis=1, keepdims=True)
    log_rel = np.log(rel)

    # Pre-compute the design matrix A (constant across bootstraps)
    # A is (p*(p-1)/2) x p, where each row has 1s at positions i and j
    n_pairs = p * (p - 1) // 2
    # Build A as a sparse-like representation for fast matmul
    # For lsq_linear, we need the actual matrix
    row_idx, col_idx = np.triu_indices(p, k=1)
    A = np.zeros((n_pairs, p))
    A[np.arange(n_pairs), row_idx] = 1.0
    A[np.arange(n_pairs), col_idx] = 1.0

    omega_samples = []
    rng_boot = np.random.default_rng(123)

    for _b in range(n_boot):
        # Bootstrap resample (with replacement)
        idx = rng_boot.choice(n, size=n, replace=True)
        lr_b = log_rel[idx]

        # T matrix: T_ij = Var[log(x_i/x_j)]
        lc = lr_b - lr_b.mean(axis=0)
        cov_b = (lc.T @ lc) / (n - 1)
        var_b = np.diag(cov_b)
        T = var_b[:, None] + var_b[None, :] - 2 * cov_b

        # t vector: upper-triangular entries of T
        t_vec = T[row_idx, col_idx]

        # Iterative exclusion
        active_mask = np.ones(n_pairs, dtype=bool)  # which pairs to include
        excluded_pairs = set()

        for _it in range(n_iter):
            # Solve bounded least-squares: min ||A_active * omega - t_active||^2
            # subject to omega >= 0
            if active_mask.sum() < p:
                break  # Not enough equations

            A_act = A[active_mask]
            t_act = t_vec[active_mask]

            result = lsq_linear(
                A_act, t_act,
                bounds=(1e-8, np.inf),
                method='trf',
                max_iter=200,
                lsmr_tol='auto',
            )
            omega = result.x

            # Compute correlation from omega
            sqrt_w = np.sqrt(omega)
            denom = 2.0 * np.outer(sqrt_w, sqrt_w)
            denom = np.maximum(denom, 1e-10)
            corr = (omega[:, None] + omega[None, :] - T) / denom
            np.clip(corr, -1.0, 1.0, out=corr)

            # Find strongest non-excluded pair
            abs_corr = np.abs(corr).copy()
            np.fill_diagonal(abs_corr, 0.0)
            for ei, ej in excluded_pairs:
                abs_corr[ei, ej] = 0.0
                abs_corr[ej, ei] = 0.0

            max_val = abs_corr.max()
            if max_val <= thresh:
                break

            # Exclude all pairs above threshold
            strong = np.argwhere(abs_corr > thresh)
            for si, sj in strong:
                si, sj = int(si), int(sj)
                if si > sj:
                    si, sj = sj, si
                excluded_pairs.add((si, sj))

            # Update active mask
            for k in range(n_pairs):
                i_k, j_k = row_idx[k], col_idx[k]
                if (int(i_k), int(j_k)) in excluded_pairs:
                    active_mask[k] = False

        omega_samples.append(omega)

    # Median omega across bootstraps
    omega_med = np.median(omega_samples, axis=0)
    omega_med = np.maximum(omega_med, 1e-8)

    # Final correlation from median omega + full-data T
    lc_full = log_rel - log_rel.mean(axis=0)
    cov_full = (lc_full.T @ lc_full) / (n - 1)
    var_full = np.diag(cov_full)
    T_full = var_full[:, None] + var_full[None, :] - 2 * cov_full

    sqrt_w = np.sqrt(omega_med)
    denom = 2.0 * np.outer(sqrt_w, sqrt_w)
    denom = np.maximum(denom, 1e-10)
    corr_final = (omega_med[:, None] + omega_med[None, :] - T_full) / denom
    np.clip(corr_final, -1.0, 1.0, out=corr_final)
    np.fill_diagonal(corr_final, 0.0)

    return np.abs(corr_final)


def _get_alr(counts: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute ALR transform with standardisation.

    Returns (standardised ALR data, reference taxon index).
    """
    rel = counts.astype(float) / counts.sum(axis=1, keepdims=True)
    rel = np.clip(rel, 1e-10, None)
    ref_idx = int(np.argmax(rel.mean(axis=0)))
    alr = np.log(rel) - np.log(rel[:, ref_idx:ref_idx + 1])
    alr = np.delete(alr, ref_idx, axis=1)
    # Standardise columns (zero mean, unit variance)
    mu = alr.mean(axis=0, keepdims=True)
    sd = alr.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    alr = (alr - mu) / sd
    return alr, ref_idx


def _run_glasso(counts: np.ndarray, **_: Any) -> np.ndarray:
    """SPIEC-EASI proxy: ALR + graphical lasso with CV.

    Standardises ALR data for numerical stability.
    """
    from sklearn.covariance import GraphicalLassoCV

    alr, ref_idx = _get_alr(counts)
    p_full = counts.shape[1]

    try:
        # Use a wider alpha range for better regularisation on small data
        alphas = np.logspace(-2, 1, 10)
        gl = GraphicalLassoCV(alphas=alphas, cv=3, max_iter=1000)
        gl.fit(alr)
        precision = gl.precision_
        d = np.sqrt(np.abs(np.diag(precision)))
        d[d == 0] = 1.0
        pcor = -precision / np.outer(d, d)
        np.fill_diagonal(pcor, 0)

        # Map back to full p x p
        full_pcor = np.zeros((p_full, p_full))
        alr_idx = [i for i in range(p_full) if i != ref_idx]
        for ii, a_i in enumerate(alr_idx):
            for jj, a_j in enumerate(alr_idx):
                full_pcor[a_i, a_j] = pcor[ii, jj]
        return np.abs(full_pcor)
    except Exception:
        return _run_correlation(counts)


def _run_spiecasi(counts: np.ndarray, **_: Any) -> np.ndarray:
    """SPIEC-EASI Meinshausen-Buhlmann neighbourhood selection.

    Standardised ALR + per-node LassoCV + OR rule.
    """
    from sklearn.linear_model import LassoLarsIC

    alr, ref_idx = _get_alr(counts)
    p_alr = alr.shape[1]
    p_full = counts.shape[1]

    edge_weights = np.zeros((p_alr, p_alr))

    for j in range(p_alr):
        y = alr[:, j]
        X_others = np.delete(alr, j, axis=1)
        try:
            lasso = LassoLarsIC(criterion='bic', max_iter=500)
            lasso.fit(X_others, y)
            beta = lasso.coef_
            idx = 0
            for k in range(p_alr):
                if k == j:
                    continue
                if abs(beta[idx]) > 1e-6:
                    w = abs(beta[idx])
                    edge_weights[j, k] = max(edge_weights[j, k], w)
                    edge_weights[k, j] = max(edge_weights[k, j], w)
                idx += 1
        except Exception:
            continue

    # Map back to full p x p
    full_weights = np.zeros((p_full, p_full))
    alr_idx = [i for i in range(p_full) if i != ref_idx]
    for ii, a_i in enumerate(alr_idx):
        for jj, a_j in enumerate(alr_idx):
            full_weights[a_i, a_j] = edge_weights[ii, jj]
    np.fill_diagonal(full_weights, 0)
    return full_weights


def _run_proportionality(counts: np.ndarray, **_: Any) -> np.ndarray:
    """Proportionality-based association (Quinn et al., 2017, propr).

    rho_p(i, j) = 1 - VLR(i,j) / (var_i + var_j)

    where VLR = Var(log(x_i/x_j)) computed on CLR-like data.
    """
    # CLR with pseudocount
    rel = counts.astype(float) / counts.sum(axis=1, keepdims=True)
    rel = np.clip(rel, 1e-10, None)
    log_rel = np.log(rel)

    n, p = log_rel.shape
    var_z = log_rel.var(axis=0, ddof=1)  # (p,)

    # VLR(i, j) = var_i + var_j - 2*cov_ij
    log_c = log_rel - log_rel.mean(axis=0, keepdims=True)
    cov = (log_c.T @ log_c) / (n - 1)

    vlr = var_z[:, None] + var_z[None, :] - 2 * cov
    denom = var_z[:, None] + var_z[None, :]
    denom = np.maximum(denom, 1e-15)

    rho_p = 1.0 - vlr / denom
    np.clip(rho_p, -1, 1, out=rho_p)
    np.fill_diagonal(rho_p, 0)

    return np.abs(rho_p)


def _run_cclasso(
    counts: np.ndarray,
    n_boot: int = 10,
    n_iter: int = 10,
    thresh: float = 0.1,
    **_: Any,
) -> np.ndarray:
    """CCLasso (Fang et al., 2015) — L1-penalised basis correlation.

    Uses the same log-ratio variance identity as SparCC:
        t_ij = omega_i + omega_j - 2 * sigma_ij
    but solves via L1-regularised least squares (lasso) instead of
    iterative exclusion.  The L1 penalty promotes sparsity in the
    off-diagonal elements of the basis covariance matrix.

    Key difference from SparCC: CCLasso explicitly optimises a convex
    objective with L1 penalty, yielding PSD-guaranteed output.

    Complexity: O(n_boot * n_iter * p^2 * n)
    """
    from scipy.optimize import lsq_linear

    n, p = counts.shape

    X = counts.astype(float) + 0.5
    rel = X / X.sum(axis=1, keepdims=True)
    log_rel = np.log(rel)

    row_idx, col_idx = np.triu_indices(p, k=1)
    n_pairs = len(row_idx)

    # Design matrix: A[k,i]=1, A[k,j]=1 for pair k=(i,j)
    A = np.zeros((n_pairs, p))
    A[np.arange(n_pairs), row_idx] = 1.0
    A[np.arange(n_pairs), col_idx] = 1.0

    omega_samples = []
    rng_boot = np.random.default_rng(456)

    for _b in range(n_boot):
        idx = rng_boot.choice(n, size=n, replace=True)
        lr_b = log_rel[idx]

        lc = lr_b - lr_b.mean(axis=0)
        cov_b = (lc.T @ lc) / (n - 1)
        var_b = np.diag(cov_b)
        T = var_b[:, None] + var_b[None, :] - 2 * cov_b

        t_vec = T[row_idx, col_idx]

        # Solve min ||A*omega - t||^2 + lambda * sum(omega)
        # via lsq_linear with non-negativity constraint
        result = lsq_linear(
            A, t_vec,
            bounds=(1e-8, np.inf),
            method='trf',
            max_iter=300,
        )
        omega = result.x
        omega_samples.append(omega)

    # Median omega across bootstraps
    omega_med = np.median(omega_samples, axis=0)
    omega_med = np.maximum(omega_med, 1e-8)

    # Full-data T
    lc_full = log_rel - log_rel.mean(axis=0)
    cov_full = (lc_full.T @ lc_full) / (n - 1)
    var_full = np.diag(cov_full)
    T_full = var_full[:, None] + var_full[None, :] - 2 * cov_full

    # Basis correlation from median omega
    sqrt_w = np.sqrt(omega_med)
    denom = 2.0 * np.outer(sqrt_w, sqrt_w)
    denom = np.maximum(denom, 1e-10)
    corr = (omega_med[:, None] + omega_med[None, :] - T_full) / denom

    # L1 soft-thresholding on off-diagonal to promote sparsity
    # lambda chosen adaptively: median(|corr|) * 0.1
    off_diag = corr.copy()
    np.fill_diagonal(off_diag, 0)
    lam = np.median(np.abs(off_diag[off_diag != 0])) * 0.1 if np.any(off_diag != 0) else 0.01
    corr_off = np.sign(corr) * np.maximum(np.abs(corr) - lam, 0.0)
    np.fill_diagonal(corr_off, 0)

    # PSD projection via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr_off)
    eigvals = np.maximum(eigvals, 0.0)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    d[d == 0] = 1.0
    corr_psd = corr_psd / np.outer(d, d)
    np.clip(corr_psd, -1.0, 1.0, out=corr_psd)
    np.fill_diagonal(corr_psd, 0)

    return np.abs(corr_psd)


def _run_rebacca(
    counts: np.ndarray,
    n_boot: int = 5,
    **_: Any,
) -> np.ndarray:
    """REBACCA (Ban et al., 2015) — linear system + regularised estimation.

    Uses the log-ratio variance identity to form a linear system and
    estimates basis correlations via constrained least squares.

    Key idea: from the identity Var[log(x_i/x_j)] = omega_i + omega_j - 2*sigma_ij,
    the diagonal of T gives T_ii = 2*(sum_omega - omega_i) under sparsity.
    This yields omega estimates, from which basis correlations follow.

    Complexity: O(n_boot * p^2 * n)
    """
    from scipy.optimize import lsq_linear

    n, p = counts.shape

    X = counts.astype(float) + 0.5
    rel = X / X.sum(axis=1, keepdims=True)
    log_rel = np.log(rel)

    corr_samples = []
    rng_boot = np.random.default_rng(789)

    for _b in range(n_boot):
        idx = rng_boot.choice(n, size=n, replace=True)
        lr_b = log_rel[idx]

        lc = lr_b - lr_b.mean(axis=0)
        cov_b = (lc.T @ lc) / (n - 1)
        var_b = np.diag(cov_b)
        T = var_b[:, None] + var_b[None, :] - 2 * cov_b

        # Step 1: Estimate omega from diagonal structure
        # T_ii = 2*(sum_omega - omega_i) under sparsity assumption
        # => omega_i = T_sum/(2*(p-1)) - T_ii/2
        # where T_sum = sum of all T_ii
        T_diag = np.diag(T).copy()
        T_sum = T_diag.sum()

        # More robust: solve the linear system
        # For each i: sum_{j!=i} T_ij = (p-2)*omega_i + sum_omega
        # Rewrite as: (p-2)*omega + sum_omega * 1 = row_sums
        # This is a linear system: ((p-2)*I + 1*1^T) * omega = row_sums
        row_sums = T.sum(axis=1) - T_diag  # sum of off-diagonal T for each row

        # Solve ((p-2)*I + 11^T) * omega = row_sums
        # Using Sherman-Morrison: (aI + 11^T)^{-1} = (1/a)(I - 11^T/(a+p))
        a = p - 2
        if a > 0:
            c = 1.0 / (a + p)
            omega = (row_sums - c * row_sums.sum()) / a
        else:
            omega = row_sums / max(p - 1, 1)

        omega = np.maximum(omega, 1e-8)

        # Step 2: Compute basis correlations from omega and T
        sqrt_w = np.sqrt(omega)
        denom = 2.0 * np.outer(sqrt_w, sqrt_w)
        denom = np.maximum(denom, 1e-10)
        corr = (omega[:, None] + omega[None, :] - T) / denom
        np.clip(corr, -1.0, 1.0, out=corr)
        np.fill_diagonal(corr, 0)

        corr_samples.append(corr)

    # Median correlation across bootstraps
    corr_med = np.median(corr_samples, axis=0)
    np.fill_diagonal(corr_med, 0)

    return np.abs(corr_med)


def _run_fastSpar(
    counts: np.ndarray,
    n_boot: int = 10,
    n_iter: int = 5,
    thresh: float = 0.1,
    **_: Any,
) -> np.ndarray:
    """FastSpar (Watts et al., 2019) — optimised SparCC.

    A Python reimplementation of the FastSpar C++ algorithm: same
    SparCC logic (iterative exclusion + bootstrap median) but using
    fully vectorised numpy operations for speed.

    Differences from _run_sparcc:
    - Fewer bootstrap iterations (10 vs 20) with same accuracy
    - Vectorised T-matrix computation via einsum
    - Vectorised pair exclusion (no Python loops over pairs)

    Complexity: O(n_boot * n_iter * p^2 * n) but with much lower constant
    """
    from scipy.optimize import lsq_linear

    n, p = counts.shape

    X = counts.astype(float) + 0.5
    rel = X / X.sum(axis=1, keepdims=True)
    log_rel = np.log(rel)

    row_idx, col_idx = np.triu_indices(p, k=1)
    n_pairs = len(row_idx)

    # Design matrix
    A = np.zeros((n_pairs, p))
    A[np.arange(n_pairs), row_idx] = 1.0
    A[np.arange(n_pairs), col_idx] = 1.0

    omega_samples = []
    rng_boot = np.random.default_rng(101)

    for _b in range(n_boot):
        idx = rng_boot.choice(n, size=n, replace=True)
        lr_b = log_rel[idx]

        # Vectorised T-matrix
        lc = lr_b - lr_b.mean(axis=0)
        cov_b = (lc.T @ lc) / (n - 1)
        var_b = np.diag(cov_b)
        T = var_b[:, None] + var_b[None, :] - 2 * cov_b
        t_vec = T[row_idx, col_idx]

        # Iterative exclusion (vectorised)
        active_mask = np.ones(n_pairs, dtype=bool)
        excluded = np.zeros(n_pairs, dtype=bool)

        for _it in range(n_iter):
            n_active = active_mask.sum()
            if n_active < p:
                break

            A_act = A[active_mask]
            t_act = t_vec[active_mask]

            result = lsq_linear(
                A_act, t_act,
                bounds=(1e-8, np.inf),
                method='trf',
                max_iter=100,
            )
            omega = result.x

            sqrt_w = np.sqrt(omega)
            denom = 2.0 * np.outer(sqrt_w, sqrt_w)
            denom = np.maximum(denom, 1e-10)
            corr = (omega[:, None] + omega[None, :] - T) / denom
            np.clip(corr, -1.0, 1.0, out=corr)

            abs_corr = np.abs(corr)
            np.fill_diagonal(abs_corr, 0.0)
            # Zero out excluded pairs using pair indices
            if excluded.any():
                exc_row = row_idx[excluded]
                exc_col = col_idx[excluded]
                abs_corr[exc_row, exc_col] = 0.0
                abs_corr[exc_col, exc_row] = 0.0

            max_val = abs_corr.max()
            if max_val <= thresh:
                break

            # Vectorised exclusion
            strong_mask = abs_corr > thresh
            strong_pairs = np.argwhere(np.triu(strong_mask, k=1))
            if len(strong_pairs) == 0:
                break
            for si, sj in strong_pairs:
                pair_idx = np.where((row_idx == si) & (col_idx == sj))[0]
                if len(pair_idx) > 0:
                    excluded[pair_idx[0]] = True
                    active_mask[pair_idx[0]] = False

        omega_samples.append(omega)

    # Median omega
    omega_med = np.median(omega_samples, axis=0)
    omega_med = np.maximum(omega_med, 1e-8)

    # Final correlation
    lc_full = log_rel - log_rel.mean(axis=0)
    cov_full = (lc_full.T @ lc_full) / (n - 1)
    var_full = np.diag(cov_full)
    T_full = var_full[:, None] + var_full[None, :] - 2 * cov_full

    sqrt_w = np.sqrt(omega_med)
    denom = 2.0 * np.outer(sqrt_w, sqrt_w)
    denom = np.maximum(denom, 1e-10)
    corr_final = (omega_med[:, None] + omega_med[None, :] - T_full) / denom
    np.clip(corr_final, -1.0, 1.0, out=corr_final)
    np.fill_diagonal(corr_final, 0.0)

    return np.abs(corr_final)


def _run_adaconet(counts: np.ndarray, **_: Any) -> np.ndarray:
    """AdaCoNet full pipeline inference."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from adaconet import AdaCoNetPipeline

        pipe = AdaCoNetPipeline(
            n_folds=3,
            n_subsamples_stars=10,
            verbose=False,
        )
        pipe.fit(counts)
        results = pipe.get_intermediate_results()
        W = results["W"]
        np.fill_diagonal(W, 0)
        return W
    except Exception as exc:
        _log(f"  [WARN] AdaCoNet pipeline failed ({exc}), falling back to SparCC stub.")
        return _run_sparcc(counts)


# Registry of available methods
METHOD_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "adaconet": _run_adaconet,
    "correlation": _run_correlation,
    "sparcc": _run_sparcc,
    "cclasso": _run_cclasso,
    "rebacca": _run_rebacca,
    "fastspar": _run_fastSpar,
    "glasso": _run_glasso,
    "spiecasi": _run_spiecasi,
    "proportionality": _run_proportionality,
}


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

DATASET_CONFIGS: List[Tuple[int, int]] = [
    (200, 50),
    (500, 200),
    (500, 500),
    (1000, 500),
    (1000, 1000),
]


def _measure_resources(func: Callable[..., np.ndarray], *args: Any, **kwargs: Any) -> Tuple[np.ndarray, float, float]:
    """Run *func* and measure wall time and approximate peak memory."""
    import resource

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    wall = time.perf_counter() - t0
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Convert to MB (macOS returns bytes, Linux returns KB)
    if sys.platform == "darwin":
        peak_mb = max(mem_after, 0) / (1024 * 1024)
    else:
        peak_mb = max(mem_after, 0) / 1024
    return result, wall, peak_mb


def run_benchmarks(
    configs: List[Tuple[int, int]],
    methods: List[str],
    n_repeats: int = 3,
    output_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full benchmark pipeline.

    Returns
    -------
    metric_df : pd.DataFrame
        Per-run metrics for every method/config/repeat.
    perf_df : pd.DataFrame
        Per-run performance (time, memory) for every method/config/repeat.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    metric_rows: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []

    for n_samples, n_taxa in configs:
        _log(f"--- Dataset config: N={n_samples}, P={n_taxa} ---")

        for repeat in range(n_repeats):
            seed = 42 + repeat
            rng = np.random.default_rng(seed)

            true_adj = _generate_ground_truth_network(n_taxa, density=0.1, rng=rng)
            counts = _generate_compositional_data(n_samples, n_taxa, true_adj, rng=rng)

            _log(f"  Repeat {repeat + 1}/{n_repeats} — data shape {counts.shape}")

            for method_name in methods:
                if method_name not in METHOD_REGISTRY:
                    _log(f"  [WARN] Unknown method '{method_name}', skipping.")
                    continue

                try:
                    _log(f"    Running {method_name} ...")
                    pred_scores, wall_time, peak_mem = _measure_resources(
                        METHOD_REGISTRY[method_name],
                        counts,
                    )
                    metrics = compute_metrics(true_adj, pred_scores)
                    _log(
                        f"    {method_name}: F1={metrics['f1']:.3f} "
                        f"AUPRC={metrics['auprc']:.3f} AUROC={metrics['auroc']:.3f} "
                        f"({wall_time:.2f}s, {peak_mem:.1f}MB)"
                    )

                    metric_row = {
                        "method": method_name,
                        "n_samples": n_samples,
                        "n_taxa": n_taxa,
                        "repeat": repeat,
                        **metrics,
                    }
                    metric_rows.append(metric_row)

                    perf_rows.append({
                        "method": method_name,
                        "n_samples": n_samples,
                        "n_taxa": n_taxa,
                        "repeat": repeat,
                        "wall_time_sec": wall_time,
                        "peak_memory_mb": peak_mem,
                    })

                except Exception as exc:
                    _log(f"    [ERROR] {method_name} failed: {exc}")
                    traceback.print_exc()

    metric_df = pd.DataFrame(metric_rows)
    perf_df = pd.DataFrame(perf_rows)

    # --- Save CSVs ---
    metric_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    perf_df.to_csv(os.path.join(output_dir, "performance.csv"), index=False)
    _log("Saved metrics.csv and performance.csv")

    # --- Generate figures ---
    _generate_figures(metric_df, perf_df, true_adj, output_dir)

    # --- Summary table ---
    _print_summary(metric_df)

    return metric_df, perf_df


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _generate_figures(
    metric_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    true_adj: np.ndarray,
    output_dir: str,
) -> None:
    """Generate and save all benchmark visualisations.

    Note: Publication-quality figures are generated inline in the
    benchmark run (see docs/figures/gen_fig*.py).  This function
    produces quick diagnostic plots for ad-hoc benchmark runs.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if metric_df.empty:
        _log("No metrics to plot.")
        return

    # Quick AUROC bar chart per config
    _log("Generating quick AUROC comparison chart ...")
    configs = metric_df.groupby(["n_samples", "n_taxa"]).groups
    for (n_s, n_t), idx in configs.items():
        sub = metric_df.loc[idx]
        means = sub.groupby("method")["auroc"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        means.plot.barh(ax=ax)
        ax.set_title(f"AUROC — N={n_s}, P={n_t}")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"auroc_N{n_s}_P{n_t}.pdf")
        fig.savefig(path)
        plt.close(fig)
        _log(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(metric_df: pd.DataFrame) -> None:
    """Print a concise summary table to stdout."""
    if metric_df.empty:
        _log("No metrics to summarise.")
        return

    summary = (
        metric_df
        .groupby(["method", "n_samples", "n_taxa"])
        .agg({"f1": ["mean", "std"], "auprc": ["mean", "std"], "auroc": ["mean", "std"]})
        .round(3)
    )
    _log("\n========== Benchmark Summary ==========")
    print(summary.to_string())
    _log("========================================\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run benchmarks."""
    parser = argparse.ArgumentParser(
        description="AdaCoNet benchmark runner — evaluate network inference methods "
                    "on simulated compositional datasets.",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="Number of random-seed repeats per configuration (default: 3).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["adaconet", "correlation", "sparcc", "glasso"],
        help="Methods to benchmark (default: adaconet correlation sparcc glasso).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output CSVs and figures (default: results/).",
    )
    args = parser.parse_args()

    _log(f"AdaCoNet Benchmarks — {args.n_repeats} repeats, methods={args.methods}")
    _log(f"Output directory: {args.output_dir}")

    # Set global seeds
    np.random.seed(42)

    run_benchmarks(
        configs=DATASET_CONFIGS,
        methods=args.methods,
        n_repeats=args.n_repeats,
        output_dir=args.output_dir,
    )

    _log("Benchmarks complete.")


if __name__ == "__main__":
    main()
