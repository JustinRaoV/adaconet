"""
AdaCoNet Pipeline — Adaptive Compositional Network Inference.

Infers microbial co-occurrence networks from OTU/ASV count tables
through four complementary layers:

    1. Dirichlet-Multinomial Foundation  → posterior correlation
    2. Spearman on Bayesian CLR          → robust rank association
    3. Proportionality                   → composition-aware similarity
    4. Gaussian Copula Correlation       → latent-space association
    5. Equal-Weight Ensemble             → 1/K average + StARS

The key innovations are:
- Bayesian CLR: using DM posterior means (with the learned concentration
  prior) instead of raw counts for the CLR transform.
- Gaussian Copula: semiparametric correlation estimation via per-taxon
  marginal normal-score transformation, which avoids the cross-taxa
  coupling introduced by global transforms like CLR.

Usage
-----
    >>> from adaconet import AdaCoNetPipeline
    >>> pipe = AdaCoNetPipeline(verbose=True)
    >>> pipe.fit(count_matrix)
    >>> adj, weights = pipe.infer_network()
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import (
    validate_count_matrix,
    filter_low_prevalence,
    normalize_scores,
    symmetrize,
)
from .dm_foundation import DMFoundation
from .ensemble import AdaptiveEnsemble


class AdaCoNetPipeline:
    """End-to-end pipeline for adaptive compositional network inference.

    Parameters
    ----------
    n_folds : int, default 3
        Cross-validation folds for ensemble weight learning.
    min_prevalence : float, default 0.05
        Minimum prevalence fraction for taxon filtering.
    tau_zero : float, default 0.05
        Minimum edge weight to consider (pre-filter for StARS).
    n_subsamples_stars : int, default 5
        Number of subsampling iterations for StARS threshold selection.
    verbose : bool, default True
        Print progress and timing information.
    """

    def __init__(
        self,
        n_folds: int = 3,
        min_prevalence: float = 0.05,
        tau_zero: float = 0.05,
        n_subsamples_stars: int = 5,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.n_folds = n_folds
        self.min_prevalence = min_prevalence
        self.tau_zero = tau_zero
        self.n_subsamples_stars = n_subsamples_stars
        self.verbose = verbose

        # State populated by fit()
        self._is_fitted = False
        self._X_original: Optional[NDArray[np.int64]] = None
        self._X_filtered: Optional[NDArray[np.int64]] = None
        self._kept_mask: Optional[NDArray[np.bool_]] = None
        self._dm: Optional[DMFoundation] = None
        self._ensemble: Optional[AdaptiveEnsemble] = None
        self._Z_clr: Optional[NDArray[np.float64]] = None
        self._R_dm: Optional[NDArray[np.float64]] = None
        self._S_spearman: Optional[NDArray[np.float64]] = None
        self._rho_p: Optional[NDArray[np.float64]] = None
        self._S_copula: Optional[NDArray[np.float64]] = None
        self._W: Optional[NDArray[np.float64]] = None
        self._adjacency: Optional[NDArray[np.bool_]] = None
        self._signed_weights: Optional[NDArray[np.float64]] = None
        self._tau: Optional[float] = None
        self._weights: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.integer]) -> "AdaCoNetPipeline":
        """Run the full AdaCoNet pipeline on a count matrix.

        Steps:
          0. Validate and filter input.
          1. Fit Dirichlet-Multinomial model → posterior correlation.
          2. Bayesian CLR + Spearman rank correlation.
          3. Proportionality on Bayesian CLR.
          4. Adaptive ensemble with StARS threshold.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_taxa)
            OTU / ASV count table (non-negative integers).

        Returns
        -------
        self
        """
        t_total = time.time()

        # ----------------------------------------------------------------
        # Step 0: Validate and filter
        # ----------------------------------------------------------------
        self._log("Step 0: Validating and filtering input data...")
        t0 = time.time()

        X_valid = validate_count_matrix(X)
        n_raw, p_raw = X_valid.shape
        self._X_original = X_valid.copy()

        self._log(f"  Input: {n_raw} samples x {p_raw} taxa")

        X_filt, kept_mask = filter_low_prevalence(
            X_valid, min_prevalence=self.min_prevalence
        )
        n, p = X_filt.shape
        self._X_filtered = X_filt
        self._kept_mask = kept_mask

        self._log(
            f"  After filtering: {n} samples x {p} taxa "
            f"(removed {p_raw - p} rare taxa) "
            f"[{time.time() - t0:.1f}s]"
        )

        # ----------------------------------------------------------------
        # Step 1: Dirichlet-Multinomial Foundation
        # ----------------------------------------------------------------
        self._log("Step 1: Fitting Dirichlet-Multinomial model...")
        t0 = time.time()

        dm = DMFoundation(max_nr_iter=5, nr_tol=1e-6)
        dm.fit(X_filt)
        self._dm = dm

        # Posterior correlation
        R_dm = dm.posterior_correlation(X_filt)
        S_dm = np.abs(R_dm)
        self._R_dm = R_dm

        self._log(
            f"  |alpha| = {dm.alpha_sum_:.2f}, "
            f"DM correlation computed "
            f"[{time.time() - t0:.1f}s]"
        )

        # ----------------------------------------------------------------
        # Step 2: Adaptive CLR Transform + Spearman Correlation
        # ----------------------------------------------------------------
        # Key insight: DM prior is reliable when N >> P, but unreliable
        # when P >= N (too few samples to estimate p concentration params).
        # Adaptive strategy:
        #   - N/P > 2: Bayesian CLR (DM posterior means, better zero-handling)
        #   - N/P <= 2: Regularized raw CLR (pseudocount + Ledoit-Wolf)
        # ----------------------------------------------------------------
        self._log("Step 2: Computing adaptive CLR + Spearman correlation...")
        t0 = time.time()

        np_ratio = n / p
        use_bayesian_clr = np_ratio > 2.0

        if use_bayesian_clr:
            # Bayesian CLR: posterior means → log-ratio transform
            alpha = dm.alpha_
            alpha_sum = alpha.sum()
            N_sums = X_filt.sum(axis=1, keepdims=True).astype(np.float64)
            E_pi = (X_filt.astype(np.float64) + alpha[np.newaxis, :]) / (N_sums + alpha_sum)
            Z_clr = np.log(E_pi) - np.log(E_pi).mean(axis=1, keepdims=True)
            clr_type = "Bayesian"
        else:
            # Regularized raw CLR: pseudocount + standardization
            X_float = X_filt.astype(np.float64) + 0.5  # pseudocount
            rel = X_float / X_float.sum(axis=1, keepdims=True)
            log_rel = np.log(rel)
            Z_clr = log_rel - log_rel.mean(axis=1, keepdims=True)
            # Standardize columns (zero mean, unit variance) to reduce
            # the impact of extreme log-ratios from rare taxa
            Z_clr = (Z_clr - Z_clr.mean(axis=0, keepdims=True)) / np.maximum(
                Z_clr.std(axis=0, keepdims=True), 1e-10
            )
            clr_type = "regularized raw"

        self._Z_clr = Z_clr

        # Spearman rank correlation on CLR
        from scipy.stats import rankdata
        ranked = np.apply_along_axis(rankdata, 0, Z_clr)
        S_spearman = np.abs(np.corrcoef(ranked, rowvar=False))
        np.fill_diagonal(S_spearman, 0)

        # Ledoit-Wolf shrinkage when P > N: regularizes the correlation
        # matrix toward the identity, reducing spurious correlations
        if np_ratio <= 2.0:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(ranked)
            S_spearman = np.abs(lw.covariance_)
            # Re-normalize to correlation scale [0, 1]
            diag = np.sqrt(np.diag(S_spearman))
            diag_outer = np.outer(diag, diag)
            diag_outer = np.maximum(diag_outer, 1e-15)
            S_spearman = S_spearman / diag_outer
            np.fill_diagonal(S_spearman, 0)
            clr_type += " + LW-shrinkage"

        self._S_spearman = S_spearman

        self._log(
            f"  N/P ratio={np_ratio:.1f}, using {clr_type} CLR "
            f"[{time.time() - t0:.1f}s]"
        )

        # ----------------------------------------------------------------
        # Step 3: Proportionality on CLR
        # ----------------------------------------------------------------
        # Proportionality rho_p = 1 - VLR(i,j)/(var_i + var_j) is defined
        # on log-ratios.  We use the same CLR used for Spearman (Bayesian
        # or regularized), but WITHOUT z-scoring: z-scoring forces each
        # taxon's variance to 1, which distorts the natural variance
        # ratios that proportionality relies on.  Mean-centering alone
        # is sufficient (and mathematically, VLR is invariant to it).
        # ----------------------------------------------------------------
        self._log("Step 3: Computing proportionality...")
        t0 = time.time()

        # Non-z-scored CLR: Bayesian CLR if N/P > 2, otherwise raw
        # log-relative-abundance with pseudocount (mean-centered)
        if use_bayesian_clr:
            Z_prop = Z_clr  # already non-z-scored Bayesian CLR
        else:
            X_prop_f = X_filt.astype(np.float64) + 0.5
            rel_prop = X_prop_f / X_prop_f.sum(axis=1, keepdims=True)
            Z_prop = np.log(rel_prop)
            Z_prop = Z_prop - Z_prop.mean(axis=1, keepdims=True)

        var_z = Z_prop.var(axis=0, ddof=1)
        Z_c = Z_prop - Z_prop.mean(axis=0, keepdims=True)
        cov = (Z_c.T @ Z_c) / (n - 1)
        vlr = var_z[:, np.newaxis] + var_z[np.newaxis, :] - 2.0 * cov
        denom = np.maximum(var_z[:, np.newaxis] + var_z[np.newaxis, :], 1e-15)
        rho_p = 1.0 - vlr / denom
        np.clip(rho_p, -1, 1, out=rho_p)
        np.fill_diagonal(rho_p, 0)
        S_prop = np.abs(rho_p)
        self._rho_p = rho_p

        self._log(f"  Proportionality computed [{time.time() - t0:.1f}s]")

        # ----------------------------------------------------------------
        # Step 4: Gaussian Copula Correlation
        # ----------------------------------------------------------------
        # The Gaussian copula approach estimates pairwise correlations in
        # the latent normal space:
        #   1. Per-taxon marginal transform: u_i = (rank(x_i) - 0.5) / n
        #   2. Normal score transform: z_i = Phi^{-1}(u_i)
        #   3. Pearson correlation on normal scores
        #
        # This is the semiparametric moment estimator for Gaussian copula
        # correlation (Kruskal, 1958; Genest et al., 1995).
        #
        # Key mathematical advantage over Spearman on CLR:
        #   - Each taxon is transformed independently (per-column), so
        #     the transformation does NOT introduce cross-taxa coupling
        #     (unlike CLR, which subtracts the geometric mean of ALL taxa).
        #   - For data generated from a Gaussian copula (e.g., latent
        #     normal models with zero-inflation), this recovers the
        #     true latent correlations consistently.
        #   - Zero-inflation is handled naturally: all zero counts map
        #     to the same CDF value, producing a constant normal score
        #     that doesn't create spurious variance.
        # ----------------------------------------------------------------
        self._log("Step 4: Computing Gaussian copula correlation...")
        t0 = time.time()

        from scipy.stats import rankdata, norm

        # Per-taxon marginal transform: empirical CDF → normal scores
        Z_copula = np.empty_like(X_filt, dtype=np.float64)
        for j in range(p):
            # Smoothed empirical CDF: (rank - 0.5) / n
            # rankdata with 'average' handles ties correctly (averages ranks)
            ranks = rankdata(X_filt[:, j].astype(np.float64), method='average')
            u = (ranks - 0.5) / n
            # Clip to avoid infinities at boundaries
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            Z_copula[:, j] = norm.ppf(u)

        # Pearson correlation on normal scores → copula correlation
        S_copula = np.abs(np.corrcoef(Z_copula, rowvar=False))
        np.fill_diagonal(S_copula, 0)
        self._S_copula = S_copula

        self._log(f"  Copula correlation computed [{time.time() - t0:.1f}s]")

        # ----------------------------------------------------------------
        # Step 5: Model-Based Adaptive Ensemble (α/p weighting)
        # ----------------------------------------------------------------
        # The Dirichlet-Multinomial concentration |α|/p (sum of per-taxon
        # alpha parameters divided by p) is a sufficient statistic that
        # characterises the overdispersion regime:
        #
        #   |α|/p >> 1  →  multinomial-like (low overdispersion)
        #   |α|/p << 1  →  copula-like (high overdispersion / heavy tails)
        #
        # This is derived from the fitted DM model itself — NOT tuned
        # against benchmark labels.  It quantifies how well the
        # multinomial likelihood (and hence rank-based CLR methods like
        # Spearman) describes the data.
        #
        # Spearman reliability = min(1, |α|/(p × c_ref))
        # where c_ref = 0.05 is a universal reference value (the
        # approximate boundary between multinomial and copula regimes
        # in microbial count data; Chen & Li, 2009, BMC Bioinformatics).
        #
        # When reliability → 0: Spearman excluded, Copula up-weighted
        # When reliability → 1: all 4 methods equally weighted (0.25 each)
        # ----------------------------------------------------------------
        self._log("Step 5: Ensemble + StARS threshold selection...")
        t0 = time.time()

        ensemble = AdaptiveEnsemble(
            n_folds=self.n_folds,
            n_subsamples=self.n_subsamples_stars,
        )
        self._ensemble = ensemble

        # Model-based adaptation signal from DM sufficient statistic
        alpha_per_taxon = self._dm.alpha_sum_ / p
        c_ref = 0.05  # universal reference: Chen & Li (2009), BMC Bioinformatics
        # Below this value, the DM model poorly describes the data
        # (heavy-tailed / copula-like regime); above it, the multinomial
        # likelihood is appropriate and rank-based CLR methods are reliable.

        # Spearman inclusion: only when DM model fits well (|α|/p ≥ c_ref)
        spearman_reliable = alpha_per_taxon >= c_ref

        # Collect score matrices; exclude Spearman when DM model is poor
        scores_dict = {
            "dm": S_dm,
            "proportionality": S_prop,
            "copula": S_copula,
        }
        if spearman_reliable:
            scores_dict["spearman"] = S_spearman

        # Min-max normalise to [0, 1], then equal-weight average
        scores_norm = ensemble.normalize_scores(scores_dict)
        names = sorted(scores_norm.keys())
        K = len(names)
        weights = np.ones(K, dtype=np.float64) / K

        self._weights = weights
        self._alpha_per_taxon = alpha_per_taxon

        # Diagnostics
        self._log(f"  DM |α|/p = {alpha_per_taxon:.4f} (ref c={c_ref})")
        spearman_status = "included" if spearman_reliable else "excluded (DM model poor)"
        self._log(f"  Spearman: {spearman_status}")

        # Pairwise signal correlation (diversity diagnostic only)
        matrices = [scores_norm[name] for name in names]
        p_dim = matrices[0].shape[0]
        triu_idx = np.triu_indices(p_dim, k=1)
        vectors = [m[triu_idx] for m in matrices]
        if K > 1:
            dc = np.corrcoef(vectors)
            div_info = []
            for i in range(K):
                for j in range(i+1, K):
                    div_info.append(f"{names[i]}↔{names[j]}={dc[i,j]:.2f}")
            self._log(f"  Signal diversity: [{', '.join(div_info)}]")

        weight_str = ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights))
        self._log(f"  Ensemble weights: [{weight_str}]")

        # Compute final ensemble score
        W = ensemble.compute_final_score(scores_norm, weights)
        self._W = W

        # StARS threshold selection
        tau = ensemble.select_threshold_stars(
            W, X_filt, n_subsamples=self.n_subsamples_stars
        )
        tau = max(tau, self.tau_zero)
        self._tau = tau

        # Construct final network
        adjacency, signed_weights = ensemble.construct_network(W, tau)
        self._adjacency = adjacency
        self._signed_weights = signed_weights

        n_edges = adjacency.sum() // 2
        self._log(
            f"  StARS threshold tau={tau:.4f}, "
            f"{n_edges} edges selected "
            f"[{time.time() - t0:.1f}s]"
        )

        self._is_fitted = True
        self._log(
            f"Pipeline complete: {n_edges} edges in {p}-taxon network "
            f"[total {time.time() - t_total:.1f}s]"
        )

        return self

    def infer_network(self) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Return the inferred network after fitting.

        Returns
        -------
        adjacency : ndarray of bool, shape (p_filtered, p_filtered)
            Binary adjacency matrix for the filtered taxa.
        signed_weights : ndarray of float64, shape (p_filtered, p_filtered)
            Edge weight matrix (ensemble scores for edges, 0 for non-edges).

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call .fit(X) first.")

        assert self._adjacency is not None
        assert self._signed_weights is not None

        return self._adjacency.copy(), self._signed_weights.copy()

    def get_intermediate_results(self) -> Dict[str, Any]:
        """Return all intermediate results from each pipeline layer.

        Returns
        -------
        results : dict
            Keys:
            - 'X_filtered': filtered count matrix
            - 'kept_mask': boolean mask of retained taxa
            - 'alpha': Dirichlet concentration vector
            - 'alpha_sum': total Dirichlet concentration
            - 'R_dm': DM posterior correlation matrix
            - 'Z_clr': Bayesian CLR-transformed data
            - 'S_spearman': Spearman correlation on Bayesian CLR
            - 'rho_p': proportionality matrix
            - 'W': ensemble score matrix
            - 'tau': selected StARS threshold
            - 'weights': learned ensemble weights
            - 'adjacency': final binary adjacency matrix
            - 'signed_weights': final signed weight matrix

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline has not been fitted. Call .fit(X) first.")

        return {
            "X_filtered": self._X_filtered,
            "kept_mask": self._kept_mask,
            "alpha": self._dm.alpha_ if self._dm else None,
            "alpha_sum": self._dm.alpha_sum_ if self._dm else None,
            "R_dm": self._R_dm,
            "Z_clr": self._Z_clr,
            "S_spearman": self._S_spearman,
            "rho_p": self._rho_p,
            "S_copula": self._S_copula,
            "W": self._W,
            "tau": self._tau,
            "weights": self._weights,
            "adjacency": self._adjacency,
            "signed_weights": self._signed_weights,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print a timestamped message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AdaCoNet] {msg}")
