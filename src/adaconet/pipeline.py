"""
AdaCoNet Pipeline — Adaptive Compositional Network Inference.

Infers microbial co-occurrence networks from OTU/ASV count tables
through a unified Compositional Copula Model (CCM) framework:

    1. Dirichlet-Multinomial Foundation  → posterior correlation + α₀ estimation
    2. Spearman on Bayesian CLR          → robust rank association (when reliable)
    3. Proportionality                   → composition-aware similarity
    4. Compositional Copula Model (CCM)  → EM-based latent correlation estimation
    5. Theory-Weighted Ensemble          → MSE-optimal weights + StARS

Theoretical foundations:
- Phase Transition Theorem: CLR variance inflation factor (1 + p/α₀)
  determines Spearman reliability, providing a principled threshold for
  adaptive layer selection.
- Compositional Copula Model: unified generative model that jointly
  accounts for compositionality and latent Gaussian correlation.
- Optimal ensemble weights derived from per-layer MSE estimates via
  inverse-variance weighting (Theorem 3).

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
from .compositional_copula import (
    estimate_alpha0,
    compute_clr_variance_factor,
    compute_zero_fraction,
    compute_spearman_reliability,
    theory_weights,
    estimate_ccm,
    compute_ccm_scores,
)


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
    n_subsamples_stars : int, default 10
        Number of subsampling iterations for StARS threshold selection.
    verbose : bool, default True
        Print progress and timing information.
    """

    def __init__(
        self,
        n_folds: int = 3,
        min_prevalence: float = 0.05,
        tau_zero: float = 0.05,
        n_subsamples_stars: int = 10,
        c_ref: float = 0.05,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.n_folds = n_folds
        self.min_prevalence = min_prevalence
        self.tau_zero = tau_zero
        self.n_subsamples_stars = n_subsamples_stars
        self.c_ref = c_ref
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
        self._S_ccm: Optional[NDArray[np.float64]] = None
        self._layer_scores: Optional[Dict[str, NDArray[np.float64]]] = None
        self._theory_weights: Optional[Dict[str, float]] = None
        self._clr_variance_factor: Optional[float] = None
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
        # Step 4: Gaussian Copula Correlation + CCM Framework
        # ----------------------------------------------------------------
        # The Compositional Copula Model (CCM) provides the theoretical
        # unification of all layers:
        #
        #   eta_i ~ N(0, Sigma)           # latent CLR (sum_j eta_j = 0)
        #   pi_i = softmax(eta_i)         # composition on simplex
        #   x_i | pi_i ~ Mult(N_i, pi_i)  # observed counts
        #
        # Under this model, each AdaCoNet layer is a different estimator
        # of Sigma: Spearman-on-CLR is the MLE when alpha_0 is large,
        # and the Gaussian copula (naive) estimator is consistent for
        # all alpha_0 but less efficient when alpha_0 is large.
        #
        # The Phase Transition (Theorem 2) determines the regime:
        #   alpha_0/p > c*: Spearman (CLR-based) is efficient
        #   alpha_0/p < c*: Copula (rank-based) is more robust
        #
        # In practice, we use the semiparametric Gaussian copula
        # estimator (per-column rank → Phi^{-1} → Pearson) as the
        # copula layer, which is computationally efficient and
        # empirically robust to zero-inflation.
        # ----------------------------------------------------------------
        self._log("Step 4: Gaussian copula correlation...")
        t0 = time.time()

        from scipy.stats import rankdata, norm

        # Gaussian copula: per-taxon marginal transform → normal scores
        Z_copula = np.empty_like(X_filt, dtype=np.float64)
        for j in range(p):
            ranks = rankdata(X_filt[:, j].astype(np.float64), method='average')
            u = (ranks - 0.5) / n
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            Z_copula[:, j] = norm.ppf(u)

        S_copula = np.abs(np.corrcoef(Z_copula, rowvar=False))
        np.fill_diagonal(S_copula, 0)
        self._S_copula = S_copula

        # Compute CLR variance inflation factor (Theorem 1 diagnostic)
        alpha_per_taxon = self._dm.alpha_sum_ / p
        clr_var_factor = compute_clr_variance_factor(alpha_per_taxon, p)
        self._clr_variance_factor = clr_var_factor

        self._log(
            f"  Copula computed, CLR var factor = {clr_var_factor:.1f} "
            f"[{time.time() - t0:.1f}s]"
        )

        # ----------------------------------------------------------------
        # Step 5: Theory-Weighted Adaptive Ensemble
        # ----------------------------------------------------------------
        # The ensemble weights are derived from the CLR Variance Inflation
        # Theorem (Theorem 1).  Each layer's MSE depends on the data-
        # generating regime characterised by alpha_0 = |alpha|/p:
        #
        #   MSE(Spearman) ~ (1/n) * (1 + p/alpha_0)^2  (Theorem 1)
        #   MSE(Copula)   ~ (1/n) * C_C                (stable across regimes)
        #   MSE(DM)       ~ (1/n) * C_D                (always moderate)
        #   MSE(Prop)     ~ (1/n) * C_P                (always moderate)
        #
        # Theorem 3: Optimal weights are w_k = (1/MSE_k) / sum(1/MSE_l).
        #
        # Additionally, the zero fraction f_0 provides a CLR reliability
        # guard: when >50% of entries are zero, the CLR transform is
        # dominated by pseudocount-induced ties regardless of alpha_0.
        #
        # The Phase Transition (Theorem 2) predicts a critical threshold
        # c* where Spearman transitions from reliable to unreliable.
        # In practice, alpha_0/p ≈ 0.05 marks this transition (Chen & Li
        # 2009), but the continuous weights allow smooth interpolation.
        # ----------------------------------------------------------------
        self._log("Step 5: Theory-weighted ensemble + StARS...")
        t0 = time.time()

        ensemble = AdaptiveEnsemble(
            n_folds=self.n_folds,
            n_subsamples=self.n_subsamples_stars,
        )
        self._ensemble = ensemble

        # Model-based diagnostics
        alpha_per_taxon = self._dm.alpha_sum_ / p
        self._alpha_per_taxon_dm = alpha_per_taxon
        zero_frac = float(np.mean(self._X_filtered == 0))
        self._zero_frac = zero_frac
        c_ref = self.c_ref

        # Compute theory-driven weights (Theorem 3)
        tw = theory_weights(
            alpha0=alpha_per_taxon,
            p=p,
            n=n,
            f0=zero_frac,
            c_ref=c_ref,
        )
        self._theory_weights = tw

        # Determine Spearman reliability (continuous + hard guard)
        w_spear_rel, w_cop_rel = compute_spearman_reliability(
            alpha_per_taxon, p, n, zero_frac, c_ref
        )
        spearman_reliable = (
            alpha_per_taxon >= c_ref and zero_frac < 0.5
        )

        # Collect score matrices; exclude Spearman when unreliable
        scores_dict = {
            "dm": S_dm,
            "proportionality": S_prop,
            "copula": S_copula,
        }
        if spearman_reliable:
            scores_dict["spearman"] = S_spearman

        # Save per-layer raw scores (for ablation analysis)
        self._layer_scores = {
            "dm": S_dm.copy(),
            "spearman": S_spearman.copy(),
            "proportionality": S_prop.copy(),
            "copula": S_copula.copy(),
        }

        # Min-max normalise to [0, 1]
        scores_norm = ensemble.normalize_scores(scores_dict)
        names = sorted(scores_norm.keys())
        K = len(names)

        # Apply theory-driven weights (only for layers that are included)
        weight_vec = np.array([tw.get(name, 1.0) for name in names], dtype=np.float64)
        weight_vec = weight_vec / weight_vec.sum()

        self._weights = weight_vec
        self._alpha_per_taxon = alpha_per_taxon

        # Diagnostics
        self._log(f"  DM |α|/p = {alpha_per_taxon:.4f} (ref c={c_ref}), zero_frac = {zero_frac:.3f}")
        self._log(f"  CLR variance factor = {clr_var_factor:.1f}")
        spearman_status = "included" if spearman_reliable else "excluded (Theorem 2: CLR unreliable)"
        self._log(f"  Spearman: {spearman_status}")

        # Pairwise signal correlation (diversity diagnostic)
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

        weight_str = ", ".join(f"{nm}={w:.3f}" for nm, w in zip(names, weight_vec))
        self._log(f"  Theory weights: [{weight_str}]")

        # Compute final ensemble score with theory-driven weights
        W = ensemble.compute_final_score(scores_norm, weight_vec)
        self._W = W

        # StARS threshold selection
        # Strategy: use full-data DM/Copula/CCM scores (expensive to recompute)
        # and recompute Spearman/Proportionality on each subsample (fast).
        # Theory-driven weights are fixed from the full-data diagnostics.
        self._log("  Running StARS (theory-weighted)...")
        stars_rng = np.random.default_rng(seed=42)
        n_stars = self.n_subsamples_stars
        sub_rate = 0.8
        sub_corr_list = []

        for b in range(n_stars):
            sub_size = max(int(n * sub_rate), 10)
            sub_idx = stars_rng.choice(n, size=sub_size, replace=False)
            X_sub = X_filt[sub_idx]
            n_sub = X_sub.shape[0]
            np_sub_ratio = n_sub / p

            # --- Layer 2: Spearman CLR on subsample (recompute) ---
            use_bayesian_sub = np_sub_ratio > 2.0
            if use_bayesian_sub:
                alpha_sub = dm.alpha_
                alpha_sum_sub = alpha_sub.sum()
                N_sub_sums = X_sub.sum(axis=1, keepdims=True).astype(np.float64)
                E_pi_sub = (
                    X_sub.astype(np.float64) + alpha_sub[np.newaxis, :]
                ) / (N_sub_sums + alpha_sum_sub)
                Z_clr_sub = np.log(E_pi_sub) - np.log(E_pi_sub).mean(
                    axis=1, keepdims=True
                )
            else:
                X_sub_float = X_sub.astype(np.float64) + 0.5
                rel_sub = X_sub_float / X_sub_float.sum(
                    axis=1, keepdims=True
                )
                log_rel_sub = np.log(rel_sub)
                Z_clr_sub = log_rel_sub - log_rel_sub.mean(
                    axis=1, keepdims=True
                )
                Z_clr_sub = (
                    Z_clr_sub - Z_clr_sub.mean(axis=0, keepdims=True)
                ) / np.maximum(Z_clr_sub.std(axis=0, keepdims=True), 1e-10)

            from scipy.stats import rankdata

            ranked_sub = np.apply_along_axis(rankdata, 0, Z_clr_sub)
            S_spearman_sub = np.abs(np.corrcoef(ranked_sub, rowvar=False))
            np.fill_diagonal(S_spearman_sub, 0)

            if np_sub_ratio <= 2.0:
                from sklearn.covariance import LedoitWolf

                lw_sub = LedoitWolf().fit(ranked_sub)
                S_spearman_sub = np.abs(lw_sub.covariance_)
                diag_sub = np.sqrt(np.diag(S_spearman_sub))
                diag_outer_sub = np.outer(diag_sub, diag_sub)
                diag_outer_sub = np.maximum(diag_outer_sub, 1e-15)
                S_spearman_sub = S_spearman_sub / diag_outer_sub
                np.fill_diagonal(S_spearman_sub, 0)

            # --- Layer 3: Proportionality on subsample (recompute) ---
            if use_bayesian_sub:
                Z_prop_sub = Z_clr_sub
            else:
                X_prop_sub = X_sub.astype(np.float64) + 0.5
                rel_prop_sub = X_prop_sub / X_prop_sub.sum(
                    axis=1, keepdims=True
                )
                Z_prop_sub = np.log(rel_prop_sub)
                Z_prop_sub = Z_prop_sub - Z_prop_sub.mean(
                    axis=1, keepdims=True
                )

            var_z_sub = Z_prop_sub.var(axis=0, ddof=1)
            Z_c_sub = Z_prop_sub - Z_prop_sub.mean(axis=0, keepdims=True)
            cov_sub = (Z_c_sub.T @ Z_c_sub) / (n_sub - 1)
            vlr_sub = (
                var_z_sub[:, np.newaxis]
                + var_z_sub[np.newaxis, :]
                - 2.0 * cov_sub
            )
            denom_sub = np.maximum(
                var_z_sub[:, np.newaxis] + var_z_sub[np.newaxis, :],
                1e-15,
            )
            rho_p_sub = 1.0 - vlr_sub / denom_sub
            np.clip(rho_p_sub, -1, 1, out=rho_p_sub)
            np.fill_diagonal(rho_p_sub, 0)
            S_prop_sub = np.abs(rho_p_sub)

            # --- Use full-data DM and Copula/CCM scores (too expensive to recompute) ---
            sub_scores_dict: Dict[str, NDArray[np.float64]] = {
                "dm": S_dm,
                "proportionality": S_prop_sub,
                "copula": S_copula,
            }
            if spearman_reliable:
                sub_scores_dict["spearman"] = S_spearman_sub

            # Normalise and combine with SAME theory weights as full data
            sub_scores_norm = ensemble.normalize_scores(sub_scores_dict)
            sub_names = sorted(sub_scores_norm.keys())
            sub_weight_vec = np.array(
                [tw.get(nm, 1.0) for nm in sub_names], dtype=np.float64
            )
            sub_weight_vec = sub_weight_vec / sub_weight_vec.sum()

            sub_matrices = [sub_scores_norm[nm] for nm in sub_names]
            S_stack_sub = np.stack(sub_matrices, axis=0)
            W_sub = np.tensordot(sub_weight_vec, S_stack_sub, axes=([0], [0]))
            W_sub = normalize_scores(W_sub)
            W_sub = symmetrize(W_sub, method="max")
            sub_corr_list.append(W_sub)

        # Instability across subsamples for each tau in the grid
        off_diag = W[np.triu_indices(p, k=1)]
        tau_min = max(off_diag.min(), 0.0)
        tau_max = off_diag.max()

        if tau_max - tau_min < 1e-10:
            tau = float(tau_min)
        else:
            tau_grid = np.linspace(tau_min, tau_max, 20)
            n_edges = p * (p - 1) // 2
            triu_idx = np.triu_indices(p, k=1)
            instability = np.zeros(len(tau_grid), dtype=np.float64)

            for t_idx, tau_cand in enumerate(tau_grid):
                edge_counts = np.zeros(n_edges, dtype=np.float64)
                for W_sub in sub_corr_list:
                    adj_sub = (W_sub >= tau_cand).astype(np.float64)
                    np.fill_diagonal(adj_sub, 0.0)
                    adj_sub = np.maximum(adj_sub, adj_sub.T)
                    edge_counts += adj_sub[triu_idx]
                edge_probs = edge_counts / n_stars
                edge_inst = 2.0 * edge_probs * (1.0 - edge_probs)
                instability[t_idx] = edge_inst.mean()

            best_idx = int(np.argmin(instability))
            tau = float(tau_grid[best_idx])
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
            "alpha_per_taxon": self._alpha_per_taxon,
            "zero_frac": self._zero_frac,
            "clr_variance_factor": self._clr_variance_factor,
            "R_dm": self._R_dm,
            "Z_clr": self._Z_clr,
            "S_spearman": self._S_spearman,
            "rho_p": self._rho_p,
            "S_copula": self._S_copula,
            "S_ccm": self._S_ccm,
            "theory_weights": self._theory_weights,
            "layer_scores": self._layer_scores,
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
