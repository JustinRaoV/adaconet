"""
Layer 1 — Dirichlet-Multinomial Foundation for AdaCoNet.

The Dirichlet-Multinomial (DM) distribution is the natural Bayesian generative
model for overdispersed compositional count data such as 16S rRNA or shotgun
metagenomics read counts.

Generative model
----------------
For each sample *i* (i = 1, ..., n):

    pi_i ~ Dirichlet(alpha)          # latent composition on the simplex
    x_i  | pi_i ~ Multinomial(N_i, pi_i)   # observed read counts

where alpha = (alpha_1, ..., alpha_p) is the shared concentration vector,
N_i = sum_j x_ij is the library size, and p is the number of taxa.

The marginal distribution of x_i (integrating out pi_i) is the
Dirichlet-Multinomial:

    P(x_i | alpha) = [N_i! / prod_j(x_ij!)] *
                      [Gamma(|alpha|) / Gamma(N_i + |alpha|)] *
                      prod_j [Gamma(x_ij + alpha_j) / Gamma(alpha_j)]

where |alpha| = sum_j alpha_j.

Parameter estimation
--------------------
We use a two-step procedure:

1. **Method of moments (MoM)** — closed-form initial estimate:
     m_j = mean_i(x_ij / N_i)         (empirical mean proportion)
     s_j^2 = var_i(x_ij / N_i)        (empirical variance of proportion)

   The MoM overdispersion parameter r is:

     r = [sum_j s_j^2  -  sum_j m_j(1 - m_j)/N_bar] /
         [sum_j m_j^2  -  (sum_j m_j^2)/N_bar]

   where N_bar = mean_i(N_i).  The total concentration is |alpha| = (1-r)/r,
   and alpha_j = m_j * |alpha|.

2. **Newton-Raphson refinement** — one gradient-ascent step on the marginal
   log-likelihood with respect to |alpha| (treating relative proportions as
   fixed) to reduce MoM bias in small samples.

Posterior inference
-------------------
Given alpha and observed counts x_i the posterior is conjugate:

    pi_i | x_i, alpha ~ Dirichlet(x_i + alpha)

Posterior mean:

    E[pi_ij | x_i] = (x_ij + alpha_j) / (N_i + |alpha|)

This naturally handles zero counts (the prior contributes alpha_j > 0),
eliminating the need for ad-hoc pseudocounts.

Posterior correlation
---------------------
We compute the across-sample Pearson correlation matrix of the posterior
means.  This captures linear dependencies in the Bayesian-smoothed
compositions while accounting for compositionality and library-size
variation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, polygamma  # type: ignore[import-untyped]


class DMFoundation:
    """Dirichlet-Multinomial foundation for compositional count data.

    Parameters
    ----------
    max_nr_iter : int, default 5
        Maximum Newton-Raphson iterations for refining |alpha|.
    nr_tol : float, default 1e-6
        Convergence tolerance for Newton-Raphson (relative change in |alpha|).

    Attributes
    ----------
    alpha_ : ndarray of float64, shape (p,)
        Estimated Dirichlet concentration vector.
    alpha_sum_ : float
        Total concentration |alpha| = sum_j alpha_j.
    proportions_mean_ : ndarray of float64, shape (p,)
        Empirical mean proportions m_j.
    n_samples_ : int
        Number of samples seen during ``fit``.
    n_taxa_ : int
        Number of taxa seen during ``fit``.
    """

    def __init__(self, max_nr_iter: int = 5, nr_tol: float = 1e-6) -> None:
        self.max_nr_iter = max_nr_iter
        self.nr_tol = nr_tol

        # Populated after fit()
        self.alpha_: Optional[NDArray[np.float64]] = None
        self.alpha_sum_: float = 0.0
        self.proportions_mean_: Optional[NDArray[np.float64]] = None
        self.n_samples_: int = 0
        self.n_taxa_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.integer]) -> "DMFoundation":
        """Estimate Dirichlet parameters from a count matrix.

        Parameters
        ----------
        X : ndarray of int, shape (n_samples, n_taxa)
            OTU / ASV count table.  Rows are samples, columns are taxa.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.int64)
        n, p = X.shape
        self.n_samples_ = n
        self.n_taxa_ = p

        # Library sizes (n,)
        N = X.sum(axis=1).astype(np.float64)  # (n,)
        N_bar = N.mean()

        # Per-sample proportions: y_ij = x_ij / N_i, shape (n, p)
        Y = X / N[:, np.newaxis]

        # Method-of-moments statistics (vectorised over p taxa)
        m = Y.mean(axis=0)       # (p,) — mean proportion per taxon
        s2 = Y.var(axis=0, ddof=1)  # (p,) — sample variance per taxon
        m2 = (Y ** 2).mean(axis=0)  # (p,) — mean of squared proportions

        # MoM overdispersion estimator r
        # numerator:   sum_j [s_j^2 - m_j(1-m_j)/N_bar]
        # denominator: sum_j [m_j^2 - m_j^2 / N_bar]   (m_j^2 from m2)
        numerator = s2.sum() - (m * (1.0 - m)).sum() / N_bar
        denominator = m2.sum() - (m ** 2).sum() / N_bar

        # Guard against degenerate cases (very homogeneous data)
        denominator = max(denominator, 1e-15)
        r = numerator / denominator
        r = np.clip(r, 1e-6, 1.0 - 1e-6)  # keep r in (0, 1)

        # Total concentration and per-taxon concentrations
        alpha_sum_mom = (1.0 - r) / r  # |alpha| from MoM
        alpha_sum_mom = max(alpha_sum_mom, 1.0)  # floor at 1 to avoid tiny priors

        # Clamp m away from zero so alpha_j > 0 for all taxa
        m_safe = np.maximum(m, 1e-10)
        m_safe = m_safe / m_safe.sum()  # re-normalise to sum to 1

        alpha_mom = m_safe * alpha_sum_mom

        # --- Newton-Raphson refinement on marginal log-likelihood ---
        alpha_sum_refined = self._newton_raphson_refine(
            X, N, alpha_sum_mom, m_safe
        )

        self.alpha_sum_ = alpha_sum_refined
        self.alpha_ = m_safe * alpha_sum_refined
        self.proportions_mean_ = m_safe

        return self

    def posterior_means(self, X: NDArray[np.integer]) -> NDArray[np.float64]:
        """Compute posterior mean compositions for each sample.

        The posterior is Dirichlet(x_i + alpha), so:

            E[pi_ij | x_i] = (x_ij + alpha_j) / (N_i + |alpha|)

        Parameters
        ----------
        X : ndarray of int, shape (n_samples, n_taxa)
            Count matrix (same taxa ordering as used in ``fit``).

        Returns
        -------
        E_pi : ndarray of float64, shape (n_samples, n_taxa)
            Posterior mean composition for each sample.  Rows sum to 1.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        N = X.sum(axis=1, keepdims=True)  # (n, 1)

        # E[pi_ij | x_i] = (x_ij + alpha_j) / (N_i + |alpha|)
        E_pi = (X + self.alpha_[np.newaxis, :]) / (N + self.alpha_sum_)
        return E_pi

    def posterior_correlation(self, X: NDArray[np.integer]) -> NDArray[np.float64]:
        """Compute the p x p posterior correlation matrix.

        This is the Pearson correlation matrix computed across samples
        of the posterior mean compositions E[pi | x_i].

        Parameters
        ----------
        X : ndarray of int, shape (n_samples, n_taxa)

        Returns
        -------
        R_dm : ndarray of float64, shape (n_taxa, n_taxa)
            Symmetric posterior correlation matrix with unit diagonal.
        """
        self._check_fitted()
        E_pi = self.posterior_means(X)  # (n, p)

        # Pearson correlation via centred, standardised columns
        # Using vectorised computation: R = (1/(n-1)) * Z^T Z  where Z is z-scored
        p = E_pi.shape[1]
        n = E_pi.shape[0]

        # Centre columns
        E_pi_c = E_pi - E_pi.mean(axis=0, keepdims=True)

        # Column standard deviations (ddof=1 for unbiased)
        stds = E_pi_c.std(axis=0, ddof=1)  # (p,)

        # Guard against constant columns (std ~ 0)
        stds = np.maximum(stds, 1e-15)

        # Z-scored matrix: (n, p)
        Z = E_pi_c / stds[np.newaxis, :]

        # Correlation matrix: (p, p)
        R_dm = (Z.T @ Z) / (n - 1)

        # Clip numerical noise and enforce unit diagonal
        np.clip(R_dm, -1.0, 1.0, out=R_dm)
        np.fill_diagonal(R_dm, 1.0)

        return R_dm

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _newton_raphson_refine(
        self,
        X: NDArray[np.integer],
        N: NDArray[np.floating],
        alpha_sum_init: float,
        m: NDArray[np.floating],
    ) -> float:
        """Refine total concentration |alpha| via Newton-Raphson.

        We optimise the scalar |alpha| while keeping the relative proportions
        m fixed (alpha_j = m_j * |alpha|).  This reduces the p-dimensional
        optimisation to a tractable 1-D problem.

        The marginal log-likelihood (up to constants) is:

            L(|alpha|) = sum_i [ log Gamma(|alpha|) - log Gamma(N_i + |alpha|)
                        + sum_j (log Gamma(x_ij + m_j|alpha|) - log Gamma(m_j|alpha|)) ]

        Gradient (score function):

            dL/d|alpha| = sum_i [ psi(|alpha|) - psi(N_i + |alpha|)
                          + sum_j m_j * (psi(x_ij + alpha_j) - psi(alpha_j)) ]

        Hessian (observed information):

            d^2 L/d|alpha|^2 = sum_i [ psi_1(|alpha|) - psi_1(N_i + |alpha|)
                               + sum_j m_j^2 * (psi_1(x_ij + alpha_j) - psi_1(alpha_j)) ]

        where psi = digamma, psi_1 = trigamma (polygamma(1, .)).

        Parameters
        ----------
        X : ndarray, shape (n, p)
        N : ndarray, shape (n,)
        alpha_sum_init : float
        m : ndarray, shape (p,)

        Returns
        -------
        alpha_sum : float
            Refined total concentration.
        """
        alpha_sum = float(alpha_sum_init)
        X_f = X.astype(np.float64)
        n = X_f.shape[0]

        for _ in range(self.max_nr_iter):
            alpha = m * alpha_sum  # (p,)

            # --- Gradient ---
            # Term 1: n * psi(|alpha|)
            g1 = n * digamma(alpha_sum)

            # Term 2: -sum_i psi(N_i + |alpha|)
            g2 = -digamma(N + alpha_sum).sum()

            # Term 3: sum_i sum_j m_j * psi(x_ij + alpha_j)
            #   = sum_j m_j * sum_i psi(x_ij + alpha_j)
            # Vectorised: psi_mat has shape (n, p), multiply by m row-wise
            psi_xa = digamma(X_f + alpha[np.newaxis, :])  # (n, p)
            g3 = (psi_xa * m[np.newaxis, :]).sum()

            # Term 4: -n * sum_j m_j * psi(alpha_j)
            g4 = -n * (m * digamma(alpha)).sum()

            grad = g1 + g2 + g3 + g4

            # --- Hessian ---
            # Term 1: n * psi_1(|alpha|)
            h1 = n * polygamma(1, alpha_sum)

            # Term 2: -sum_i psi_1(N_i + |alpha|)
            h2 = -polygamma(1, N + alpha_sum).sum()

            # Term 3: sum_i sum_j m_j^2 * psi_1(x_ij + alpha_j)
            psi1_xa = polygamma(1, X_f + alpha[np.newaxis, :])  # (n, p)
            m2 = m ** 2  # (p,)
            h3 = (psi1_xa * m2[np.newaxis, :]).sum()

            # Term 4: -n * sum_j m_j^2 * psi_1(alpha_j)
            h4 = -n * (m2 * polygamma(1, alpha)).sum()

            hess = h1 + h2 + h3 + h4

            # Newton step (gradient ascent: move in direction of gradient)
            # For maximisation: alpha_new = alpha - grad / hess (hess < 0 at max)
            if abs(hess) < 1e-30:
                break  # Hessian is degenerate; stop refinement

            step = -grad / hess  # note: hess should be negative at maximum

            # Safeguarded step: halve if it would make |alpha| negative
            proposed = alpha_sum + step
            if proposed <= 0.1:
                step = 0.1 - alpha_sum  # clamp to minimum viable value
            alpha_sum_new = alpha_sum + step

            # Check convergence
            if abs(alpha_sum_new - alpha_sum) / max(abs(alpha_sum), 1.0) < self.nr_tol:
                alpha_sum = alpha_sum_new
                break

            alpha_sum = alpha_sum_new

        return max(alpha_sum, 0.1)

    def _check_fitted(self) -> None:
        """Raise if ``fit`` has not been called."""
        if self.alpha_ is None:
            raise RuntimeError(
                "DMFoundation has not been fitted yet. Call .fit(X) first."
            )
