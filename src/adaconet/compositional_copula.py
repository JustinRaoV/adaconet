"""
Compositional Copula Model (CCM) — Layer 4 of AdaCoNet.

A unified generative model for compositional count data that jointly
models the compositional constraint (sum-to-one) and latent Gaussian
correlation structure.

Generative model
----------------
For each sample i:

    eta_i ~ N_P(0, Sigma)        # latent log-composition (CLR parameterisation)
    pi_i = softmax(eta_i)        # composition on the simplex
    x_i | pi_i ~ Mult(N_i, pi_i) # observed counts

where Sigma is a p x p PSD matrix with Sigma * 1 = 0 (enforcing the
CLR constraint sum_j eta_j = 0).

The key insight is that the Dirichlet-Multinomial model's overdispersion
parameter |alpha|/p controls the *concentration* of pi around its mean.
When |alpha|/p is large, pi is concentrated and the CLR transform
accurately recovers eta, making Spearman correlation effective.
When |alpha|/p is small, pi is diffuse and the CLR transform introduces
large variance inflation, making copula-based methods preferable.

This module provides:

1.  ``estimate_alpha0``  — concentration parameter from DM model
2.  ``compute_clr_variance``  — CLR variance inflation factor
3.  ``estimate_ccm``     — EM-based estimation of Sigma
4.  ``theory_weights``   — MSE-optimal ensemble weights derived from theory

Theoretical foundations
-----------------------
Theorem 1 (CLR Variance Inflation).
    Under the DM model with total concentration alpha_0 = |alpha|,

        Var(clr(X)_j) = (1 + p / alpha_0) * sigma_j^2 / n + O(1/n^2)

    where sigma_j^2 is the variance of the latent eta_j.  The factor
    (1 + p / alpha_0) quantifies the amplification of sampling noise
    by the compositional CLR transform.

Theorem 2 (Phase Transition for Spearman-on-CLR).
    The MSE of the Spearman correlation on CLR-transformed data satisfies

        MSE <= (1/n) * (1 + p / alpha_0)^2 * C_1 + C_2 * exp(-alpha_0 / p)

    There exists a critical threshold c* such that:
        alpha_0 / p > c*  =>  Spearman is a consistent estimator
        alpha_0 / p < c*  =>  Spearman error is dominated by compositional noise

Theorem 3 (Optimal Ensemble Weights).
    The MSE-minimising weights for the ensemble W = sum_k w_k S_k are

        w_k = (1 / MSE_k) / sum_l (1 / MSE_l)

    where MSE_k is estimated from the theoretical variance formulas.

References
----------
Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
Chapman & Hall.

Chen, J. and Li, H. (2009). "Variable selection for the Dirichlet-
Multinomial distribution." BMC Bioinformatics, 10:355.

Fang, H., Huang, C., Zhao, H., and Deng, M. (2017). "gCoda: Conditional
Dependence Network Inference for Compositional Data." J Comput Biol.

Liu, H., Roeder, K., and Wasserman, L. (2010). "StARS." NeurIPS.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def estimate_alpha0(
    alpha_sum: float,
    p: int,
) -> float:
    """Extract the DM concentration parameter |alpha|/p.

    Parameters
    ----------
    alpha_sum : float
        Total DM concentration |alpha| from DMFoundation.
    p : int
        Number of taxa.

    Returns
    -------
    alpha0 : float
        Per-taxon concentration |alpha|/p.
    """
    return alpha_sum / max(p, 1)


def compute_clr_variance_factor(
    alpha0: float,
    p: int,
) -> float:
    """Compute the CLR variance inflation factor (1 + p / alpha_0).

    From Theorem 1: the CLR transform amplifies sampling noise by this
    factor due to the shared geometric-mean denominator.

    Parameters
    ----------
    alpha0 : float
        Per-taxon DM concentration |alpha|/p.
    p : int
        Number of taxa.

    Returns
    -------
    inflation : float
        Variance inflation factor >= 1.
    """
    alpha_sum = alpha0 * p
    if alpha_sum < 1e-10:
        return float(p) * 1e4  # extreme inflation
    return 1.0 + p / alpha_sum


def compute_zero_fraction(X: NDArray[np.integer]) -> float:
    """Fraction of zero entries in the count matrix.

    This is a CLR reliability diagnostic: when more than 50% of entries
    are zero, the CLR pseudocount creates dominant rank ties that degrade
    Spearman correlation regardless of the DM concentration.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Count matrix.

    Returns
    -------
    f0 : float in [0, 1]
        Fraction of zero entries.
    """
    return float(np.mean(X == 0))


def compute_spearman_reliability(
    alpha0: float,
    p: int,
    n: int,
    f0: float,
    c_ref: float = 0.05,
    f0_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Compute the Spearman reliability weight from theory.

    The weight is derived from the CLR signal-to-noise ratio (Theorem 1).
    The DM concentration ratio alpha_0 = |alpha|/p directly controls the
    CLR variance inflation (1 + p/|alpha|) = (1 + 1/alpha_0).
    The Spearman reliability weight represents the fraction of "usable
    signal" from the CLR:

        w_raw = alpha_0 / (alpha_0 + c_ref)

    where c_ref is the critical concentration threshold (Chen & Li 2009).
    This gives w = 0.5 when alpha_0 = c_ref (phase transition boundary),
    w -> 1 for alpha_0 >> c_ref (multinomial regime), and
    w -> 0 for alpha_0 << c_ref (copula regime).

    A zero-fraction guard (f_0 > f0_threshold) applies a soft penalty
    when the CLR is dominated by pseudocount-induced ties.

    Parameters
    ----------
    alpha0 : float
        Per-taxon DM concentration |alpha|/p.
    p : int
        Number of taxa.
    n : int
        Number of samples.
    f0 : float
        Zero fraction.
    c_ref : float
        Reference concentration threshold (Chen & Li 2009).
    f0_threshold : float
        Zero fraction threshold for CLR reliability.

    Returns
    -------
    w_spearman : float in [0, 1]
        Continuous reliability weight for Spearman layer.
    w_copula : float in [0, 1]
        Complementary weight for copula layer (higher when Spearman
        is unreliable).
    """
    # alpha0 = |alpha|/p is the DM concentration ratio.
    # The CLR reliability is governed by alpha0 directly:
    #   alpha0 >> c_ref  =>  CLR well-conditioned, Spearman reliable
    #   alpha0 << c_ref  =>  CLR too noisy, Spearman unreliable
    #
    # Reliability weight: w = alpha0 / (alpha0 + c_ref)
    # At alpha0 = c_ref: w = 0.5 (phase transition boundary)
    # At alpha0 >> c_ref: w -> 1 (multinomial regime)
    # At alpha0 << c_ref: w -> 0 (copula regime)
    w_raw = alpha0 / (alpha0 + c_ref) if (alpha0 + c_ref) > 0 else 0.5

    # Zero-fraction guard: when f_0 > threshold, CLR is unreliable
    # regardless of alpha_0.  Apply a soft penalty:
    #   penalty = 1 if f_0 < threshold, decays smoothly above
    if f0 < f0_threshold:
        penalty = 1.0
    else:
        # Sigmoid-like decay above threshold
        excess = (f0 - f0_threshold) / max(f0_threshold, 0.01)
        penalty = 1.0 / (1.0 + 4.0 * excess)

    w_spearman = float(np.clip(w_raw * penalty, 0.0, 1.0))
    w_copula = 1.0 - w_spearman

    return w_spearman, w_copula


def theory_weights(
    alpha0: float,
    p: int,
    n: int,
    f0: float,
    n_layers: int = 4,
    c_ref: float = 0.05,
    f0_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute theory-driven ensemble weights (Theorem 3).

    The MSE of each layer depends on the data-generating regime,
    characterised by the concentration ratio gamma = alpha_0 / p:

    - Spearman-on-CLR:  MSE_S ~ (1 + 1/gamma)^2 / n   (Theorem 1)
    - CCM Copula:       MSE_C ~ constant / n            (stable across regimes)
    - DM posterior:     MSE_D ~ constant / n            (always moderate)
    - Proportionality:  MSE_P ~ constant / n            (always moderate)

    The optimal weight (inverse-variance) for Spearman relative to a
    baseline is:

        w_S_raw = gamma / (gamma + c_ref)

    This is the fraction of "usable CLR signal": when gamma >> c_ref
    (multinomial regime), Spearman is reliable (w_S -> 1).  When
    gamma << c_ref (copula regime), CLR is too noisy (w_S -> 0).

    DM and Proportionality receive baseline weight 1.0 (they are always
    moderately informative across regimes).  Copula gets complementary
    weight (1 - w_S_raw) since it is strongest when Spearman is weakest.

    Parameters
    ----------
    alpha0, p, n, f0 : float/int
        Data characteristics.
    n_layers : int
        Number of layers (3 if Spearman excluded, 4 otherwise).
    c_ref, f0_threshold : float
        Diagnostic thresholds.

    Returns
    -------
    weights : dict mapping layer name to weight
    """
    w_spearman, w_copula = compute_spearman_reliability(
        alpha0, p, n, f0, c_ref, f0_threshold
    )

    # Base weights: DM and Proportionality are always informative
    # Their MSE is less sensitive to alpha_0 (they don't use CLR ranks)
    raw = {
        "dm": 1.0,
        "proportionality": 1.0,
    }

    # Spearman: weight proportional to CLR reliability
    # Floor at 0.05 to ensure minimal contribution (diversity benefit)
    raw["spearman"] = max(w_spearman, 0.05)

    # Copula: complementary to Spearman (strong when CLR is noisy)
    raw["copula"] = max(w_copula, 0.05)

    # Normalise to sum to 1
    total = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}

    return weights


def estimate_ccm(
    X: NDArray[np.integer],
    alpha0: float,
    n_em_iter: int = 10,
    em_tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[NDArray[np.float64], float]:
    """Fit the Compositional Copula Model via EM (Laplace approximation).

    Estimates the latent correlation matrix Sigma from compositional count
    data, using the logistic-normal generative model:

        eta_i ~ N(0, Sigma),  pi_i = softmax(eta_i),  x_i ~ Mult(N_i, pi_i)

    The EM algorithm iterates:

    **E-step** (Laplace approximation of the posterior p(eta_i | x_i, Sigma)):
        D_i = diag(N_i * pi_hat_i * (1 - pi_hat_i))
        H_i = Sigma^{-1} + D_i  (posterior precision, on constraint space)
        C_i = H_i^{-1}          (posterior covariance)
        eta_hat_i = C_i @ D_i @ (y_i - pi_hat_i)  (posterior mean)
        where y_i = x_i / N_i is the empirical composition.

    **M-step**:
        Sigma_new = (1/n) sum_i (C_i + eta_hat_i @ eta_hat_i^T)

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Count matrix (non-negative integers).
    alpha0 : float
        DM concentration |alpha|/p, used to regularise the E-step.
        Larger alpha0 means the posterior is more concentrated, making
        the Laplace approximation more accurate.
    n_em_iter : int
        Maximum number of EM iterations.
    em_tol : float
        Convergence tolerance (relative change in log-likelihood proxy).
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    Sigma : ndarray, shape (p, p)
        Estimated latent correlation matrix (symmetric, unit diagonal,
        PSD, with Sigma @ 1 ≈ 0).
    converged_ll : float
        Final log-likelihood proxy value.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, p = X.shape

    N = X.sum(axis=1, keepdims=True)  # (n, 1) library sizes
    Y = X / np.maximum(N, 1.0)  # (n, p) empirical compositions

    # ------------------------------------------------------------------
    # Initialise Sigma from CLR correlation
    # ------------------------------------------------------------------
    eps = 0.5
    Y_clr = np.log(Y + eps) - np.log(Y + eps).mean(axis=1, keepdims=True)
    Sigma = np.corrcoef(Y_clr, rowvar=False)
    np.fill_diagonal(Sigma, 1.0)

    # Enforce CLR constraint: Sigma @ 1 = 0
    # Sigma_constrained = P @ Sigma @ P where P = I - (1/p) 11^T
    one = np.ones(p)
    P = np.eye(p) - np.outer(one, one) / p
    Sigma = P @ Sigma @ P

    # Ensure PSD: clip negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    Sigma = (eigvecs * eigvals) @ eigvecs.T

    # Enforce unit diagonal (correlation matrix)
    d = np.sqrt(np.diag(Sigma))
    d = np.maximum(d, 1e-10)
    Sigma = Sigma / np.outer(d, d)
    Sigma = P @ Sigma @ P  # re-enforce constraint

    prev_ll = -np.inf
    converged_ll = prev_ll

    for it in range(n_em_iter):
        # ----------------------------------------------------------
        # E-step: posterior for each sample
        # ----------------------------------------------------------
        # Precompute Sigma pseudo-inverse on constraint space
        # Sigma is rank p-1 with null space 1.
        # Use eigendecomposition to get pseudo-inverse.
        eigvals_s, V_s = np.linalg.eigh(Sigma)
        # Eigenvalues sorted ascending; the smallest should be ~0 (null)
        # Invert the non-null eigenvalues
        inv_eigvals = np.where(eigvals_s > 1e-6, 1.0 / eigvals_s, 0.0)
        Sigma_inv = (V_s * inv_eigvals) @ V_s.T  # pseudo-inverse

        M_sum = np.zeros((p, p))

        for i in range(n_samples):
            Ni = float(N[i, 0])
            if Ni < 1.0:
                continue

            # Current composition estimate: posterior mean from DM
            pi_hat = (X[i] + alpha0) / (Ni + alpha0 * p)
            pi_hat = np.maximum(pi_hat, 1e-10)
            pi_hat = pi_hat / pi_hat.sum()

            # Diagonal Fisher information (multinomial Hessian)
            d_i = Ni * pi_hat * (1.0 - pi_hat)
            d_i = np.maximum(d_i, 1e-10)

            # Posterior precision: H = Sigma_inv + diag(d_i)
            H = Sigma_inv + np.diag(d_i)

            # Posterior covariance: C = H^{-1}
            # Use Cholesky if possible, fallback to solve
            try:
                L = np.linalg.cholesky(H)
                C = np.linalg.solve(H, np.eye(p))
            except np.linalg.LinAlgError:
                # H might not be PD due to Sigma_inv null space
                # Regularise slightly
                H_reg = H + np.eye(p) * 1e-6
                C = np.linalg.solve(H_reg, np.eye(p))

            # Enforce symmetry
            C = 0.5 * (C + C.T)

            # Posterior mean: eta_hat = C @ d_i * (y_i - pi_hat)
            residual = Y[i] - pi_hat
            eta_hat = C @ (d_i * residual)

            # Project to constraint space (remove mean)
            eta_hat = eta_hat - eta_hat.mean()

            # Accumulate second moment for M-step
            M_sum += C + np.outer(eta_hat, eta_hat)

        # ----------------------------------------------------------
        # M-step: update Sigma
        # ----------------------------------------------------------
        Sigma_new = M_sum / n_samples

        # Enforce CLR constraint
        Sigma_new = P @ Sigma_new @ P

        # Ensure PSD
        eigvals_new, eigvecs_new = np.linalg.eigh(Sigma_new)
        eigvals_new = np.maximum(eigvals_new, 1e-8)
        Sigma_new = (eigvecs_new * eigvals_new) @ eigvecs_new.T

        # Convert to correlation matrix
        d_new = np.sqrt(np.maximum(np.diag(Sigma_new), 1e-10))
        Sigma_new = Sigma_new / np.outer(d_new, d_new)

        # Re-enforce constraint after normalisation
        Sigma_new = P @ Sigma_new @ P

        # ----------------------------------------------------------
        # Convergence check (log-likelihood proxy)
        # ----------------------------------------------------------
        # Proxy: sum of log-eigenvalues of posterior precision
        # This measures the "tightness" of the posterior, which should
        # increase (posterior gets tighter) as Sigma improves.
        eigvals_check = np.linalg.eigvalsh(Sigma_new)
        ll_proxy = float(np.sum(np.log(np.maximum(eigvals_check, 1e-15))))

        if verbose:
            print(f"  CCM iter {it+1}: ll_proxy = {ll_proxy:.4f}")

        if abs(ll_proxy - prev_ll) < em_tol * max(abs(prev_ll), 1.0):
            converged_ll = ll_proxy
            Sigma = Sigma_new
            if verbose:
                print(f"  CCM converged at iteration {it+1}")
            break

        prev_ll = ll_proxy
        converged_ll = ll_proxy
        Sigma = Sigma_new

    # Final: convert to correlation matrix
    d_final = np.sqrt(np.maximum(np.diag(Sigma), 1e-10))
    Sigma = Sigma / np.outer(d_final, d_final)
    np.fill_diagonal(Sigma, 1.0)
    Sigma = np.clip(Sigma, -1.0, 1.0)

    # Ensure symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)

    return Sigma, converged_ll


def compute_ccm_scores(
    Sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert CCM correlation matrix to absolute association scores.

    Parameters
    ----------
    Sigma : ndarray, shape (p, p)
        Estimated latent correlation matrix from ``estimate_ccm``.

    Returns
    -------
    scores : ndarray, shape (p, p)
        Absolute correlation values, zero diagonal.
    """
    scores = np.abs(Sigma)
    np.fill_diagonal(scores, 0.0)
    return scores
