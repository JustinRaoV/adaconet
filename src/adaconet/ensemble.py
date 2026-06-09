"""
Layer 4 — Diversity-Aware Ensemble for AdaCoNet.

Combines three association signals (DM posterior correlation, Spearman on
CLR, Proportionality) into a single network score using equal-weight
averaging followed by StARS-based threshold selection.

Equal weighting
---------------
All signals receive equal weight (1/K).  No training data or labels are
used to determine weights, ensuring the ensemble is not biased toward
any particular data-generating mechanism.  Pairwise correlation between
signals is reported as a diversity diagnostic.

Score normalisation
------------------
Each signal's off-diagonal entries are min-max normalised to [0, 1].
This preserves each layer's natural score distribution for StARS
thresholding.

StARS threshold selection
-------------------------
Stability Approach to Regularisation Selection (StARS; Liu et al., 2010):

1. For each candidate threshold tau in a grid:
   a. Subsample the data B times (fraction = subsample_rate).
   b. For each subsample, compute the full network and threshold at tau.
   c. Edge instability = 2 * mean(P(e)) * (1 - mean(P(e))) averaged over
      all possible edges, where P(e) is the empirical probability of edge e
      across subsamples.
   d. Total instability D(tau) = sum_e instability(e).

2. Select tau* = argmin_tau D(tau).

References
----------
Lovell, D. et al. (2015). "Proportionality: A Valid Alternative to
Correlation for Relative Data." PLoS Comput Biol.

Liu, H. et al. (2010). "Stability Approach to Regularization Selection
(StARS) for High Dimensional Graphical Models." NeurIPS.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .utils import normalize_scores, symmetrize


class AdaptiveEnsemble:
    """Adaptive ensemble of multiple association score matrices.

    Uses equal-weight averaging (1/K per layer) for principled,
    training-free ensemble combination.  StARS selects the edge
    threshold automatically.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation folds for weight learning.
    n_subsamples : int, default 5
        Number of subsampling iterations for StARS.
    subsample_rate : float, default 0.8
        Fraction of samples retained in each StARS subsample.
    tau_grid_size : int, default 20
        Number of candidate thresholds in the StARS grid.
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_subsamples: int = 5,
        subsample_rate: float = 0.8,
        tau_grid_size: int = 20,
    ) -> None:
        self.n_folds = n_folds
        self.n_subsamples = n_subsamples
        self.subsample_rate = subsample_rate
        self.tau_grid_size = tau_grid_size

        # Learned weights (populated after learn_weights)
        self.weights_: Optional[NDArray[np.float64]] = None
        self.score_names_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Proportionality
    # ------------------------------------------------------------------

    @staticmethod
    def compute_proportionality(
        Z_clr: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Compute the proportionality matrix rho_p from CLR-transformed data.

        rho_p(i, j) = 1 - VLR(i,j) / (var_i + var_j)

        where VLR(i,j) = Var(log(x_i) - log(x_j)) computed across samples.
        Since Z_clr is already log-transformed and centred, we can use it
        directly:

            VLR(i, j) = Var(Z_clr[:, i] - Z_clr[:, j])

        Parameters
        ----------
        Z_clr : ndarray, shape (n_samples, n_taxa)
            CLR-transformed data.

        Returns
        -------
        rho_p : ndarray of float64, shape (p, p)
            Proportionality matrix with values in [-1, 1].
        """
        Z = np.asarray(Z_clr, dtype=np.float64)
        n, p = Z.shape

        var_z = Z.var(axis=0, ddof=1)
        Z_c = Z - Z.mean(axis=0, keepdims=True)
        cov = (Z_c.T @ Z_c) / (n - 1)

        vlr = var_z[:, np.newaxis] + var_z[np.newaxis, :] - 2.0 * cov
        denom = var_z[:, np.newaxis] + var_z[np.newaxis, :]
        denom = np.maximum(denom, 1e-15)

        rho_p = 1.0 - vlr / denom
        np.clip(rho_p, -1.0, 1.0, out=rho_p)
        np.fill_diagonal(rho_p, 0.0)

        return rho_p

    @staticmethod
    def compute_spearman_clr(
        Z_clr: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Compute Spearman rank correlation on CLR-transformed data.

        Vectorised implementation: rank each column, then compute Pearson
        correlation on the ranks via np.corrcoef.

        Parameters
        ----------
        Z_clr : ndarray, shape (n_samples, n_taxa)

        Returns
        -------
        corr : ndarray of float64, shape (p, p)
            Absolute Spearman correlation matrix.
        """
        from scipy.stats import rankdata

        Z = np.asarray(Z_clr, dtype=np.float64)
        n, p = Z.shape

        # Rank each column (Spearman = Pearson on ranks)
        ranked = np.empty_like(Z)
        for j in range(p):
            ranked[:, j] = rankdata(Z[:, j])

        # Pearson correlation on ranks (vectorised via np.corrcoef)
        corr = np.corrcoef(ranked, rowvar=False)
        np.abs(corr, out=corr)
        np.fill_diagonal(corr, 0.0)

        return corr

    # ------------------------------------------------------------------
    # Score normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_scores(
        scores_dict: Dict[str, NDArray[np.floating]],
    ) -> Dict[str, NDArray[np.float64]]:
        """Min-max normalise each score matrix to [0, 1].

        Uses the ``normalize_scores`` utility which zeroes the diagonal
        and scales off-diagonal entries to [0, 1].

        Parameters
        ----------
        scores_dict : dict[str, ndarray]
            Mapping from score name to (p, p) score matrix.

        Returns
        -------
        normed : dict[str, ndarray of float64]
            Normalised score matrices.
        """
        normed = {}
        for name, S in scores_dict.items():
            normed[name] = normalize_scores(S)
        return normed

    @staticmethod
    def rank_normalize_scores(
        scores_dict: Dict[str, NDArray[np.floating]],
    ) -> Dict[str, NDArray[np.float64]]:
        """Rank-normalise each score matrix to uniform [0, 1].

        For each (p, p) score matrix, extracts upper-triangular entries,
        assigns ranks (1 to n_pairs), maps ranks to [0, 1], then fills
        the symmetric matrix (including lower triangle) and zeroes the
        diagonal.

        This ensures each signal contributes equally to the ensemble
        regardless of its original score range or distribution, preventing
        high-variance signals from dominating the average.

        Parameters
        ----------
        scores_dict : dict[str, ndarray]
            Mapping from score name to (p, p) score matrix.

        Returns
        -------
        normed : dict[str, ndarray of float64]
            Rank-normalised score matrices with values in [0, 1].
        """
        from scipy.stats import rankdata

        normed = {}
        for name, S in scores_dict.items():
            S = np.asarray(S, dtype=np.float64)
            p = S.shape[0]
            triu_idx = np.triu_indices(p, k=1)
            vals = np.abs(S[triu_idx])

            # Rank off-diagonal entries → map to [0, 1]
            ranks = rankdata(vals)
            n_pairs = len(ranks)
            if n_pairs > 0:
                ranks_normed = (ranks - 1) / max(n_pairs - 1, 1)
            else:
                ranks_normed = ranks

            # Fill symmetric matrix
            R = np.zeros((p, p), dtype=np.float64)
            R[triu_idx] = ranks_normed
            R = R + R.T  # symmetrise
            np.fill_diagonal(R, 0.0)
            normed[name] = R

        return normed

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def learn_weights(
        self,
        scores_dict: Dict[str, NDArray[np.float64]],
        X: NDArray[np.integer],
        n_folds: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute equal ensemble weights (1/K for each layer).

        Equal weighting is the principled default: no training signal
        is used to determine weights, ensuring the ensemble is not
        biased toward any particular data-generating mechanism.

        Pairwise correlations between normalised layer scores are
        computed as a diversity diagnostic only (not used for weighting).

        Parameters
        ----------
        scores_dict : dict[str, ndarray]
            Score matrices (normalised to [0, 1]).
        X : ndarray, shape (n_samples, n_taxa)
            Original count matrix (unused, kept for API compat).
        n_folds : int, optional
            Unused, kept for API compat.

        Returns
        -------
        weights : ndarray of float64, shape (K,)
            Equal weight vector [1/K, ..., 1/K].
        """
        names = sorted(scores_dict.keys())
        K = len(names)
        self.score_names_ = names
        weights = np.ones(K, dtype=np.float64) / K

        # Diversity diagnostic: pairwise Pearson between layer vectors
        matrices = [scores_dict[name] for name in names]
        p = matrices[0].shape[0]
        triu_idx = np.triu_indices(p, k=1)
        vectors = [m[triu_idx] for m in matrices]

        if K > 1:
            self._diversity_corr = np.corrcoef(vectors)
        else:
            self._diversity_corr = np.array([[1.0]])

        self.weights_ = weights
        return weights

    # ------------------------------------------------------------------
    # Final score computation
    # ------------------------------------------------------------------

    def compute_final_score(
        self,
        scores_dict: Dict[str, NDArray[np.float64]],
        weights: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Compute the ensemble score via weighted arithmetic mean.

        W(i, j) = sum_k w_k * S_k(i, j)

        The arithmetic mean is more forgiving than geometric mean: a single
        low score from one layer does not collapse the result.  This is
        important because different layers capture different aspects of
        the association structure.

        Parameters
        ----------
        scores_dict : dict[str, ndarray]
            Normalised score matrices.
        weights : ndarray, optional
            Weight vector of shape (K,).  If None, uses ``self.weights_``.

        Returns
        -------
        W : ndarray of float64, shape (p, p)
            Ensemble score matrix with values in [0, 1].
        """
        names = sorted(scores_dict.keys())
        K = len(names)

        if weights is None:
            if self.weights_ is not None and len(self.weights_) == K:
                weights = self.weights_
            else:
                weights = np.ones(K, dtype=np.float64) / K

        weights = weights / weights.sum()

        matrices = [scores_dict[name] for name in names]
        S_stack = np.stack(matrices, axis=0)  # (K, p, p)

        # Weighted arithmetic mean
        W = np.tensordot(weights, S_stack, axes=([0], [0]))  # (p, p)

        # Normalise to [0, 1] and symmetrise
        W = normalize_scores(W)
        W = symmetrize(W, method="max")

        return W

    # ------------------------------------------------------------------
    # StARS threshold selection
    # ------------------------------------------------------------------

    def select_threshold_stars(
        self,
        W: NDArray[np.float64],
        X: NDArray[np.integer],
        n_subsamples: Optional[int] = None,
        subsample_rate: Optional[float] = None,
    ) -> float:
        """Select the optimal edge threshold via StARS stability analysis.

        For each subsample, recompute the association matrix using
        vectorised Pearson correlation on CLR-transformed data (fast).
        Then measure edge instability across subsamples for each
        candidate threshold.

        Parameters
        ----------
        W : ndarray, shape (p, p)
            Ensemble score matrix.
        X : ndarray, shape (n_samples, n_taxa)
            Original count matrix (used for subsampling).
        n_subsamples : int, optional
            Override ``self.n_subsamples``.
        subsample_rate : float, optional
            Override ``self.subsample_rate``.

        Returns
        -------
        tau_star : float
            Optimal threshold.
        """
        n_subsamples = (
            n_subsamples if n_subsamples is not None else self.n_subsamples
        )
        subsample_rate = (
            subsample_rate if subsample_rate is not None else self.subsample_rate
        )

        n = X.shape[0]
        p = W.shape[0]
        rng = np.random.default_rng(seed=42)

        # Precompute CLR for full data
        X_f = X.astype(np.float64) + 0.5
        rel_full = X_f / X_f.sum(axis=1, keepdims=True)
        clr_full = np.log(rel_full)
        clr_full = clr_full - clr_full.mean(axis=1, keepdims=True)

        # Compute association matrices for each subsample (vectorised Pearson)
        sub_corr_list = []
        for _ in range(n_subsamples):
            sub_size = max(int(n * subsample_rate), 10)
            sub_idx = rng.choice(n, size=sub_size, replace=False)
            clr_sub = clr_full[sub_idx]

            # Vectorised Pearson correlation (single np.corrcoef call)
            corr = np.corrcoef(clr_sub, rowvar=False)
            np.abs(corr, out=corr)
            np.fill_diagonal(corr, 0.0)
            sub_corr_list.append(corr)

        # Threshold grid from the ensemble score matrix W
        off_diag = W[np.triu_indices(p, k=1)]
        tau_min = max(off_diag.min(), 0.0)
        tau_max = off_diag.max()

        if tau_max - tau_min < 1e-10:
            return float(tau_min)

        tau_grid = np.linspace(tau_min, tau_max, self.tau_grid_size)
        n_edges = p * (p - 1) // 2
        triu_idx = np.triu_indices(p, k=1)

        # For each tau, compute edge instability across subsamples
        instability = np.zeros(self.tau_grid_size, dtype=np.float64)

        for t_idx, tau in enumerate(tau_grid):
            edge_counts = np.zeros(n_edges, dtype=np.float64)
            for corr in sub_corr_list:
                adj = (corr >= tau).astype(np.float64)
                np.fill_diagonal(adj, 0.0)
                adj = np.maximum(adj, adj.T)
                edge_counts += adj[triu_idx]

            edge_probs = edge_counts / n_subsamples
            edge_instability = 2.0 * edge_probs * (1.0 - edge_probs)
            instability[t_idx] = edge_instability.mean()

        best_idx = np.argmin(instability)
        tau_star = float(tau_grid[best_idx])
        return tau_star

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    @staticmethod
    def construct_network(
        W: NDArray[np.float64],
        tau: float,
    ) -> Tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Threshold the ensemble score matrix to produce a network.

        Parameters
        ----------
        W : ndarray, shape (p, p)
            Ensemble score matrix (symmetric, values in [0, 1]).
        tau : float
            Edge threshold.  Pairs with W(i, j) >= tau are connected.

        Returns
        -------
        adjacency : ndarray of bool, shape (p, p)
            Binary adjacency matrix (symmetric, zero diagonal).
        signed_weights : ndarray of float64, shape (p, p)
            Signed weight matrix where edge weights are the ensemble scores
            and non-edges are zero.
        """
        p = W.shape[0]

        # Threshold
        adjacency = W >= tau
        np.fill_diagonal(adjacency, False)

        # Ensure symmetry
        adjacency = adjacency | adjacency.T

        # Signed weight matrix: keep scores for edges, zero elsewhere
        signed_weights = np.where(adjacency, W, 0.0)
        np.fill_diagonal(signed_weights, 0.0)

        return adjacency, signed_weights
