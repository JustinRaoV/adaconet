"""Benchmark runner for comparing AdaCoNet with baseline network inference methods."""

import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.covariance import GraphicalLassoCV

from .metrics import NetworkMetrics


# ---------------------------------------------------------------------------
# Helper transforms
# ---------------------------------------------------------------------------


def _add_pseudocount(counts: np.ndarray, pseudo: float = 0.5) -> np.ndarray:
    """Add a pseudocount to avoid log(0)."""
    return counts.astype(float) + pseudo


def _clr_transform(counts: np.ndarray) -> np.ndarray:
    """Centered log-ratio (CLR) transform.

    Args:
        counts: (n, p) count matrix (pseudocounts should be added first).

    Returns:
        (n, p) CLR-transformed matrix.
    """
    log_x: np.ndarray = np.log(counts)
    return log_x - log_x.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Default method implementations
# ---------------------------------------------------------------------------


def _adaconet_proxy(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Placeholder AdaCoNet inference using CLR + Pearson correlation.

    This is a **proxy** used when the real ``adaconet.pipeline.AdaCoNetPipeline``
    is not importable.  Replace with the real pipeline in production.

    Returns:
        ``(pred_adj, pred_scores)`` — binary adjacency and continuous scores.
    """
    try:
        from adaconet.pipeline import AdaCoNetPipeline  # type: ignore[import-untyped]

        pipeline = AdaCoNetPipeline()
        result = pipeline.fit(counts)
        return result["adjacency"], result["scores"]
    except ImportError:
        pass

    # Fallback: CLR + absolute Pearson correlation, thresholded at 0.3
    X_clr: np.ndarray = _clr_transform(_add_pseudocount(counts))
    corr: np.ndarray = np.corrcoef(X_clr, rowvar=False)
    scores: np.ndarray = np.abs(corr)
    np.fill_diagonal(scores, 0.0)
    adj: np.ndarray = (scores >= 0.3).astype(int)
    np.fill_diagonal(adj, 0)
    return adj, scores


def _sparcc(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified SparCC-like correlation estimation.

    Uses log-ratio variances to approximate the basis covariance structure
    (Friedman & Alm 2012, simplified without iterative refinement).

    Returns:
        ``(pred_adj, pred_scores)``
    """
    X: np.ndarray = _add_pseudocount(counts)
    n, p = X.shape

    # Log-ratio matrix: L_ij = log(X_i / X_j) for each sample
    log_X: np.ndarray = np.log(X)

    # Variance of pairwise log-ratios across samples: V[a,b] = var(log(X_a/X_b))
    # Compute efficiently: V_ab = var(log_X[:,a] - log_X[:,b])
    # = var_a + var_b - 2*cov_ab  (in log-space)
    var_log: np.ndarray = np.var(log_X, axis=0, ddof=1)  # (p,)

    # V[a,b] = var_a + var_b - 2*cov(log_X_a, log_X_b)
    cov_log: np.ndarray = np.cov(log_X, rowvar=False)  # (p, p)
    V: np.ndarray = var_log[:, None] + var_log[None, :] - 2.0 * cov_log

    # Component variances t_i ≈ mean of V row (SparCC approximation)
    t: np.ndarray = V.mean(axis=1)
    t = np.clip(t, 1e-10, None)

    # SparCC correlation: cor_ab = (t_a + t_b - V_ab) / (2 * sqrt(t_a * t_b))
    scores: np.ndarray = (t[:, None] + t[None, :] - V) / (
        2.0 * np.sqrt(t[:, None] * t[None, :])
    )
    scores = np.clip(scores, -1.0, 1.0)
    np.fill_diagonal(scores, 0.0)

    # Binary adjacency: threshold on absolute correlation
    abs_scores: np.ndarray = np.abs(scores)
    threshold: float = float(np.percentile(abs_scores[abs_scores > 0], 80))
    adj: np.ndarray = (abs_scores >= threshold).astype(int)
    np.fill_diagonal(adj, 0)

    return adj, np.abs(scores)


def _spiec_easi_proxy(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SPIEC-EASI proxy: CLR transform followed by graphical lasso.

    Uses ``sklearn.covariance.GraphicalLassoCV`` to estimate a sparse precision
    matrix.  Non-zero off-diagonal entries in the precision matrix are taken as
    edges.

    Returns:
        ``(pred_adj, pred_scores)``
    """
    X_clr: np.ndarray = _clr_transform(_add_pseudocount(counts))

    # GraphicalLassoCV can fail on very high-dimensional or degenerate data
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=200)
        gl.fit(X_clr)
        precision: np.ndarray = gl.precision_
    except Exception:
        # Fallback: use empirical correlation
        precision = np.corrcoef(X_clr, rowvar=False)

    # Score = absolute partial correlation (normalized precision matrix)
    diag: np.ndarray = np.sqrt(np.abs(np.diag(precision)))
    diag = np.clip(diag, 1e-10, None)
    scores: np.ndarray = np.abs(precision / np.outer(diag, diag))
    np.fill_diagonal(scores, 0.0)

    # Binary adjacency: threshold on normalized partial correlation
    abs_scores_upper: np.ndarray = scores[np.triu_indices_from(scores, k=1)]
    gl_threshold: float = (
        float(np.percentile(abs_scores_upper, 80))
        if len(abs_scores_upper) > 0
        else 0.05
    )
    adj: np.ndarray = (scores >= gl_threshold).astype(int)
    np.fill_diagonal(adj, 0)

    return adj, scores


def _proportionality(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Proportionality metric (Lovell et al. 2015, ``propr``-style).

    Computes the proportionality coefficient:
        phi(a, b) = var(log(a/b)) / (var(log(a)) + var(log(b)))
        rho_p = 1 - phi

    Returns:
        ``(pred_adj, pred_scores)``
    """
    X: np.ndarray = _add_pseudocount(counts)
    log_X: np.ndarray = np.log(X)

    var_log: np.ndarray = np.var(log_X, axis=0, ddof=1)  # (p,)
    cov_log: np.ndarray = np.cov(log_X, rowvar=False)  # (p, p)

    # var(log(a/b)) = var_a + var_b - 2*cov_ab
    V: np.ndarray = var_log[:, None] + var_log[None, :] - 2.0 * cov_log
    denom: np.ndarray = var_log[:, None] + var_log[None, :]
    denom = np.clip(denom, 1e-10, None)

    phi: np.ndarray = V / denom
    scores: np.ndarray = 1.0 - phi  # proportionality in (-1, 1]
    np.fill_diagonal(scores, 0.0)

    # Binary adjacency: top 20% strongest proportionality scores
    abs_scores: np.ndarray = np.abs(scores)
    upper_vals: np.ndarray = abs_scores[np.triu_indices_from(abs_scores, k=1)]
    threshold: float = float(np.percentile(upper_vals, 80)) if len(upper_vals) > 0 else 0.5
    adj: np.ndarray = (abs_scores >= threshold).astype(int)
    np.fill_diagonal(adj, 0)

    return adj, abs_scores


def _spearman(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman rank correlation on CLR-transformed data.

    Returns:
        ``(pred_adj, pred_scores)``
    """
    X_clr: np.ndarray = _clr_transform(_add_pseudocount(counts))
    n, p = X_clr.shape

    # Compute Spearman correlation pairwise
    scores: np.ndarray = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            r, _ = sp_stats.spearmanr(X_clr[:, i], X_clr[:, j])
            scores[i, j] = abs(r) if not np.isnan(r) else 0.0
            scores[j, i] = scores[i, j]

    # Binary adjacency: top 20% strongest correlations
    upper_vals: np.ndarray = scores[np.triu_indices_from(scores, k=1)]
    threshold: float = float(np.percentile(upper_vals, 80)) if len(upper_vals) > 0 else 0.3
    adj: np.ndarray = (scores >= threshold).astype(int)
    np.fill_diagonal(adj, 0)

    return adj, scores


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates benchmark experiments comparing multiple inference methods.

    Each method is a callable with signature::

        method(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]

    returning ``(pred_adj, pred_scores)`` where *pred_adj* is a binary
    ``(p, p)`` symmetric adjacency matrix and *pred_scores* is a continuous
    ``(p, p)`` symmetric score matrix.
    """

    _DEFAULT_METHODS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
        "AdaCoNet": _adaconet_proxy,
        "SparCC": _sparcc,
        "SPIEC-EASI-proxy": _spiec_easi_proxy,
        "Proportionality": _proportionality,
        "Spearman": _spearman,
    }

    def __init__(
        self,
        methods: Optional[Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]]] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            methods: Mapping of ``{name: callable}``.  Defaults to the five
                built-in methods when *None*.
        """
        self.methods: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = (
            dict(self._DEFAULT_METHODS) if methods is None else dict(methods)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        simulator_results: Dict[str, Any],
        method_options: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Run all registered methods on a single simulated dataset.

        Args:
            simulator_results: Output of
                :meth:`~MicrobialNetworkSimulator.generate`.  Must contain at
                least ``counts`` and ``adjacency``.
            method_options: Optional per-method kwargs, e.g.
                ``{'SparCC': {'threshold': 0.4}}``.  Currently reserved for
                future use.

        Returns:
            A :class:`~pandas.DataFrame` with one row per method and columns:
            ``method, n_samples, n_taxa, precision, recall, f1, auprc, auroc,
            degree_ks, cluster_corr, modularity, hub_recovery,
            wall_time_sec, peak_memory_mb``.
        """
        counts: np.ndarray = simulator_results["counts"]
        true_adj: np.ndarray = simulator_results["adjacency"]
        n_samples: int = counts.shape[0]
        n_taxa: int = counts.shape[1]

        rows: List[Dict[str, Any]] = []

        for name, method_fn in self.methods.items():
            row: Dict[str, Any] = {
                "method": name,
                "n_samples": n_samples,
                "n_taxa": n_taxa,
            }

            try:
                pred_adj, pred_scores, wall_time, peak_mem = self._run_single(
                    method_fn, counts
                )

                # Compute metrics
                metrics: Dict[str, float] = NetworkMetrics.compute_all(
                    true_adj, pred_adj, pred_scores
                )
                row["precision"] = metrics["precision"]
                row["recall"] = metrics["recall"]
                row["f1"] = metrics["f1"]
                row["auprc"] = metrics["auprc"]
                row["auroc"] = metrics["auroc"]
                row["degree_ks"] = metrics["degree_ks"]
                row["cluster_corr"] = metrics["cluster_corr"]
                row["modularity"] = metrics["modularity_pred"]
                row["hub_recovery"] = metrics["hub_recovery"]
                row["wall_time_sec"] = wall_time
                row["peak_memory_mb"] = peak_mem

            except Exception:
                # Method failed — record NaN for every metric
                row["precision"] = float("nan")
                row["recall"] = float("nan")
                row["f1"] = float("nan")
                row["auprc"] = float("nan")
                row["auroc"] = float("nan")
                row["degree_ks"] = float("nan")
                row["cluster_corr"] = float("nan")
                row["modularity"] = float("nan")
                row["hub_recovery"] = float("nan")
                row["wall_time_sec"] = float("nan")
                row["peak_memory_mb"] = float("nan")

            rows.append(row)

        return pd.DataFrame(rows)

    def run_scaling_benchmark(
        self,
        configs: List[Dict[str, Any]],
        n_repeats: int = 3,
    ) -> pd.DataFrame:
        """Run scaling experiments across multiple (n_samples, n_taxa) configs.

        Each configuration is repeated ``n_repeats`` times with different random
        seeds.

        Args:
            configs: List of dicts with at least ``n_samples`` and ``n_taxa``.
            n_repeats: Number of independent repeats per configuration.

        Returns:
            Concatenated :class:`~pandas.DataFrame` with all runs.
        """
        from .simulator import MicrobialNetworkSimulator

        all_frames: List[pd.DataFrame] = []

        for config_idx, config in enumerate(configs):
            for repeat in range(n_repeats):
                seed: int = config.get("seed", 42) + config_idx * 1000 + repeat
                sim = MicrobialNetworkSimulator(
                    n_taxa=config["n_taxa"],
                    n_samples=config["n_samples"],
                    seed=seed,
                )
                sim_result: Dict[str, Any] = sim.generate(
                    scale_free=config.get("scale_free", True),
                    density=config.get("density", 0.1),
                    zero_fraction=config.get("zero_fraction", 0.3),
                    overdispersion=config.get("overdispersion", 1.0),
                )
                sim_result["n_samples"] = config["n_samples"]
                sim_result["n_taxa"] = config["n_taxa"]

                df: pd.DataFrame = self.run_benchmark(sim_result)
                df["repeat"] = repeat
                all_frames.append(df)

        return pd.concat(all_frames, ignore_index=True)

    @staticmethod
    def summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean +/- std for each metric, grouped by (method, n_samples, n_taxa).

        Args:
            results_df: DataFrame produced by :meth:`run_benchmark` or
                :meth:`run_scaling_benchmark`.

        Returns:
            Summary DataFrame with ``mean`` and ``std`` multi-level columns.
        """
        metric_cols: List[str] = [
            "precision",
            "recall",
            "f1",
            "auprc",
            "auroc",
            "degree_ks",
            "cluster_corr",
            "modularity",
            "hub_recovery",
            "wall_time_sec",
            "peak_memory_mb",
        ]
        # Only include columns that actually exist
        metric_cols = [c for c in metric_cols if c in results_df.columns]
        group_cols: List[str] = ["method", "n_samples", "n_taxa"]

        summary: pd.DataFrame = (
            results_df.groupby(group_cols)[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_single(
        method_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
        counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Execute a single method with timing and memory tracking.

        Returns:
            ``(pred_adj, pred_scores, wall_time_sec, peak_memory_mb)``
        """
        tracemalloc.start()
        t0: float = time.time()

        pred_adj, pred_scores = method_fn(counts)

        wall_time: float = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # peak is in bytes; convert to megabytes
        peak_mb: float = peak / (1024.0 * 1024.0)

        return pred_adj, pred_scores, wall_time, peak_mb
