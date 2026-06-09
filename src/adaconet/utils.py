"""
Utility functions for AdaCoNet.

Provides data validation, filtering, transformations, and score normalization
used across all layers of the pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_count_matrix(X: NDArray[np.number]) -> NDArray[np.int64]:
    """Validate and cast an input count matrix.

    Checks performed:
      - Must be 2-D.
      - All entries must be finite, non-negative integers.
      - No all-zero rows (samples) or all-zero columns (taxa).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_taxa)
        Raw OTU / ASV count table.

    Returns
    -------
    X_int : ndarray of int64, shape (n_samples, n_taxa)
        Validated integer count matrix.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2-D count matrix; got array with {X.ndim} dimensions."
        )

    # Check for finite values before casting to int
    if not np.all(np.isfinite(X)):
        raise ValueError("Count matrix contains non-finite (NaN or Inf) values.")

    # Cast to integer
    X_int = X.astype(np.int64)

    # Non-negativity
    if np.any(X_int < 0):
        raise ValueError("Count matrix contains negative values.")

    # All-zero rows (samples with no reads)
    row_sums = X_int.sum(axis=1)
    if np.any(row_sums == 0):
        bad_rows = np.where(row_sums == 0)[0]
        raise ValueError(
            f"Samples (rows) with zero total reads detected at indices: "
            f"{bad_rows.tolist()}. Remove empty samples before analysis."
        )

    # All-zero columns (taxa never observed)
    col_sums = X_int.sum(axis=0)
    if np.any(col_sums == 0):
        bad_cols = np.where(col_sums == 0)[0]
        raise ValueError(
            f"Taxa (columns) with zero total reads detected at indices: "
            f"{bad_cols.tolist()}. Remove unobserved taxa before analysis."
        )

    return X_int


# ---------------------------------------------------------------------------
# Low-prevalence filtering
# ---------------------------------------------------------------------------

def filter_low_prevalence(
    X: NDArray[np.number],
    min_prevalence: float = 0.05,
    min_abundance: int = 10,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Remove rare taxa that are unlikely to yield reliable associations.

    A taxon (column) is retained if **both** conditions are met:
      1. It is present (count > 0) in at least ``min_prevalence`` fraction of
         samples.
      2. Its total abundance across all samples is >= ``min_abundance``.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_taxa)
        Raw count matrix (should already pass ``validate_count_matrix``).
    min_prevalence : float, default 0.05
        Minimum fraction of samples in which the taxon must be observed.
    min_abundance : int, default 10
        Minimum total read count across all samples.

    Returns
    -------
    X_filtered : ndarray of int64, shape (n_samples, n_kept)
        Filtered count matrix.
    kept_mask : ndarray of bool, shape (n_taxa,)
        Boolean mask indicating which taxa were retained (``True`` = kept).
    """
    X_int = np.asarray(X, dtype=np.int64)
    n_samples, n_taxa = X_int.shape

    # Prevalence: fraction of samples where count > 0
    prevalence = (X_int > 0).sum(axis=0) / n_samples  # shape (n_taxa,)

    # Total abundance
    abundance = X_int.sum(axis=0)  # shape (n_taxa,)

    kept_mask = (prevalence >= min_prevalence) & (abundance >= min_abundance)

    # Safety: keep at least two taxa so downstream pairwise analysis is possible
    if kept_mask.sum() < 2:
        raise ValueError(
            f"Only {kept_mask.sum()} taxa survived filtering (min_prevalence="
            f"{min_prevalence}, min_abundance={min_abundance}). Relax filter "
            f"criteria or provide a richer dataset."
        )

    return X_int[:, kept_mask], kept_mask


# ---------------------------------------------------------------------------
# Centered log-ratio (CLR) transform
# ---------------------------------------------------------------------------

def compute_clr(posterior_means: NDArray[np.floating]) -> NDArray[np.float64]:
    """Centered log-ratio transform for compositional data.

    For a composition **p** = (p_1, ..., p_D) with p_j > 0:

        CLR(p)_j = ln(p_j) - (1/D) * sum_{k=1}^{D} ln(p_k)

    The CLR transform maps the simplex to real space while preserving
    sub-compositional coherence, making it suitable for downstream
    correlation and MI calculations.

    Parameters
    ----------
    posterior_means : ndarray, shape (n_samples, n_taxa)
        Positive-valued posterior mean compositions (e.g. from DM posterior).
        Must be strictly positive; zeros should be handled before calling
        this function (the DM posterior naturally avoids zeros).

    Returns
    -------
    Z_clr : ndarray of float64, shape (n_samples, n_taxa)
        CLR-transformed matrix.
    """
    pm = np.asarray(posterior_means, dtype=np.float64)

    # Clamp tiny values to avoid log(0); the DM posterior should not produce
    # exact zeros, but floating-point underflow is possible for very small alpha.
    pm = np.maximum(pm, 1e-300)

    log_pm = np.log(pm)                          # (n, p)
    geo_mean_log = log_pm.mean(axis=1, keepdims=True)  # (n, 1)

    return log_pm - geo_mean_log


# ---------------------------------------------------------------------------
# Numerically stable log
# ---------------------------------------------------------------------------

def safe_log(x: NDArray[np.floating], eps: float = 1e-10) -> NDArray[np.float64]:
    """Numerically stable natural logarithm.

    Computes ``ln(max(x, eps))`` to avoid ``-inf`` for zero or near-zero
    inputs that are common in sparse microbial count tables.

    Parameters
    ----------
    x : array-like
        Input values (expected non-negative).
    eps : float, default 1e-10
        Floor value applied before taking the log.

    Returns
    -------
    log_x : ndarray of float64
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), eps))


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def normalize_scores(S: NDArray[np.floating]) -> NDArray[np.float64]:
    """Min-max normalize a score matrix to the [0, 1] range.

    Diagonal elements are forced to zero (self-edges are meaningless in
    co-occurrence networks).  If all off-diagonal values are identical
    the result is a zero matrix.

    Parameters
    ----------
    S : ndarray, shape (p, p)
        Raw pairwise score matrix (need not be symmetric).

    Returns
    -------
    S_norm : ndarray of float64, shape (p, p)
        Normalized score matrix with values in [0, 1] and zero diagonal.
    """
    S = np.asarray(S, dtype=np.float64).copy()
    p = S.shape[0]

    # Zero out diagonal so it does not affect normalization range
    np.fill_diagonal(S, 0.0)

    s_min = S.min()
    s_max = S.max()

    if s_max - s_min < 1e-15:
        # All (off-diagonal) values are essentially equal
        return np.zeros((p, p), dtype=np.float64)

    S_norm = (S - s_min) / (s_max - s_min)
    np.fill_diagonal(S_norm, 0.0)
    return S_norm


# ---------------------------------------------------------------------------
# Symmetrize
# ---------------------------------------------------------------------------

def symmetrize(M: NDArray[np.floating], method: str = "max") -> NDArray[np.float64]:
    """Make a square matrix symmetric.

    Parameters
    ----------
    M : ndarray, shape (p, p)
        Square matrix.
    method : {'max', 'min', 'mean'}, default 'max'
        Strategy for combining M[i,j] and M[j,i]:
        - ``'max'``:  take the larger absolute value (preserves strongest signal).
        - ``'min'``:  take the smaller absolute value (conservative).
        - ``'mean'``: arithmetic average.

    Returns
    -------
    M_sym : ndarray of float64, shape (p, p)
        Symmetric matrix with zero diagonal.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected a square matrix; got shape {M.shape}.")

    if method == "max":
        M_sym = np.maximum(M, M.T)
    elif method == "min":
        M_sym = np.minimum(M, M.T)
    elif method == "mean":
        M_sym = (M + M.T) / 2.0
    else:
        raise ValueError(
            f"Unknown symmetrization method '{method}'. "
            f"Choose from 'max', 'min', 'mean'."
        )

    np.fill_diagonal(M_sym, 0.0)
    return M_sym
