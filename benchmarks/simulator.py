"""Simulated microbial data generator with known ground truth networks."""

import warnings
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


class MicrobialNetworkSimulator:
    """Generates synthetic microbial count data with a known ground truth network.

    The generative model follows:
        1. Sample a ground truth graph G (scale-free or Erdos-Renyi).
        2. Assign signed interaction strengths to edges.
        3. Draw absolute abundances from a latent Gaussian model whose
           covariance structure mirrors G.
        4. Normalize to compositions, inject structural zeros, and draw
           Multinomial counts.
    """

    def __init__(self, n_taxa: int, n_samples: int, seed: int = 42) -> None:
        """Initialize the simulator.

        Args:
            n_taxa: Number of taxa (features / species).
            n_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
        """
        self.n_taxa: int = n_taxa
        self.n_samples: int = n_samples
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scale_free: bool = True,
        density: float = 0.1,
        zero_fraction: float = 0.3,
        overdispersion: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate synthetic count data with a known ground truth adjacency matrix.

        Args:
            scale_free: If True use Barabasi-Albert preferential attachment;
                otherwise use Erdos-Renyi.
            density: Edge density for Erdos-Renyi; controls the *m* parameter
                (edges per new node) for Barabasi-Albert via ``m = density*p/2``.
            zero_fraction: Fraction of entries set to zero as structural zeros.
            overdispersion: Scale parameter for the Gamma overdispersion step.
                Larger values produce heavier tails.

        Returns:
            Dict with keys:
                - ``counts``            : (n, p) int count matrix
                - ``adjacency``         : (p, p) binary symmetric adjacency matrix
                - ``interaction_matrix``: (p, p) signed interaction strength matrix
                - ``compositions``      : (n, p) true relative abundances (pre-sampling)
                - ``network_type``      : ``'scale_free'`` or ``'erdos_renyi'``
        """
        p: int = self.n_taxa
        n: int = self.n_samples
        rng: np.random.Generator = self.rng

        # ------------------------------------------------------------------
        # Step 1 – Ground truth network
        # ------------------------------------------------------------------
        graph_seed: int = int(rng.integers(0, 2**31))

        if scale_free:
            # Barabasi-Albert: m edges per new node.
            # Approximate density ≈ 2m/p  =>  m ≈ density*p/2
            m: int = max(1, int(round(density * p / 2)))
            G: nx.Graph = nx.barabasi_albert_graph(p, m, seed=graph_seed)
            network_type: str = "scale_free"
        else:
            G = nx.erdos_renyi_graph(p, density, seed=graph_seed)
            network_type = "erdos_renyi"

        adjacency: np.ndarray = nx.to_numpy_array(G, dtype=int)

        # ------------------------------------------------------------------
        # Step 2 – Interaction strengths
        # ------------------------------------------------------------------
        interaction_matrix: np.ndarray = self._sample_interaction_strengths(adjacency)

        # ------------------------------------------------------------------
        # Step 3 – Absolute abundances via latent Gaussian model
        # ------------------------------------------------------------------
        d: int = max(1, p // 4)  # latent dimension

        # Loading matrix B (d x p).  Smoothing with (I + alpha*A) makes
        # connected taxa share similar columns, inducing network-structured
        # correlations in the resulting log-abundances.
        alpha: float = 0.5
        B: np.ndarray = rng.standard_normal((d, p)) / np.sqrt(d)
        S: np.ndarray = np.eye(p) + alpha * adjacency
        # Row-normalize S to prevent numerical amplification in dense graphs
        row_norms: np.ndarray = np.clip(S.sum(axis=1), 1e-10, None)
        S = S / row_norms[:, None]
        # Use np.dot to avoid spurious BLAS warnings from the @ operator
        B_smooth: np.ndarray = np.dot(B, S)

        # Latent variables z ~ N(0, I_d), one per sample
        Z: np.ndarray = rng.standard_normal((n, d))

        # log(a) = Z @ B_smooth + noise
        noise_std: float = 0.3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_a: np.ndarray = np.dot(Z, B_smooth) + rng.normal(
                0.0, noise_std, size=(n, p)
            )

        # Per-taxon baseline abundance (some taxa are intrinsically more abundant)
        mean_log_a: np.ndarray = rng.normal(5.0, 1.0, size=p)
        log_a = log_a + mean_log_a[np.newaxis, :]

        # Exponentiate to obtain absolute abundances
        a: np.ndarray = np.exp(log_a)

        # Gamma overdispersion: E[a_new] = a,  Var[a_new] = a * overdispersion
        if overdispersion > 0.0:
            shape: np.ndarray = np.clip(a / overdispersion, 1e-6, None)
            a = rng.gamma(shape, overdispersion)

        # ------------------------------------------------------------------
        # Step 4 – Normalize to compositions
        # ------------------------------------------------------------------
        compositions: np.ndarray = self._normalize_rows(a)

        # ------------------------------------------------------------------
        # Step 5 – Zero inflation (structural zeros)
        # ------------------------------------------------------------------
        if zero_fraction > 0.0:
            zero_mask: np.ndarray = rng.random((n, p)) < zero_fraction
            compositions = compositions * (~zero_mask).astype(float)
            compositions = self._normalize_rows(compositions)

        # ------------------------------------------------------------------
        # Step 6 – Multinomial sampling of counts
        # ------------------------------------------------------------------
        # Library sizes N_i ~ Poisson(10 000)
        library_sizes: np.ndarray = rng.poisson(10_000, size=n)
        counts: np.ndarray = np.zeros((n, p), dtype=int)

        for i in range(n):
            pi: np.ndarray = compositions[i]
            pi_sum: float = pi.sum()
            if pi_sum > 0.0:
                pi = pi / pi_sum
            else:
                pi = np.full(p, 1.0 / p)
            counts[i] = rng.multinomial(library_sizes[i], pi)

        return {
            "counts": counts,
            "adjacency": adjacency,
            "interaction_matrix": interaction_matrix,
            "compositions": compositions,
            "network_type": network_type,
        }

    def generate_multiple_configs(
        self, configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate datasets for multiple (n_samples, n_taxa) configurations.

        Each config dict may contain:
            ``n_samples``, ``n_taxa``, ``scale_free``, ``density``,
            ``zero_fraction``, ``overdispersion``.

        Args:
            configs: List of configuration dicts.

        Returns:
            List of simulation result dicts (same schema as :meth:`generate`),
            each augmented with ``n_samples`` and ``n_taxa`` keys.
        """
        results: List[Dict[str, Any]] = []

        for i, config in enumerate(configs):
            sim = MicrobialNetworkSimulator(
                n_taxa=config["n_taxa"],
                n_samples=config["n_samples"],
                seed=self.seed + i,
            )
            result: Dict[str, Any] = sim.generate(
                scale_free=config.get("scale_free", True),
                density=config.get("density", 0.1),
                zero_fraction=config.get("zero_fraction", 0.3),
                overdispersion=config.get("overdispersion", 1.0),
            )
            result["n_samples"] = config["n_samples"]
            result["n_taxa"] = config["n_taxa"]
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_interaction_strengths(self, adjacency: np.ndarray) -> np.ndarray:
        """Assign signed interaction strengths to edges in *adjacency*.

        Roughly 60 % of edges receive positive strengths ~ N(0.5, 0.2) and the
        remaining 40 % receive negative strengths ~ N(-0.5, 0.2).  The matrix is
        made symmetric.
        """
        p: int = adjacency.shape[0]
        rng: np.random.Generator = self.rng
        interaction_matrix: np.ndarray = np.zeros((p, p), dtype=float)

        upper_row, upper_col = np.triu_indices(p, k=1)
        edge_mask: np.ndarray = adjacency[upper_row, upper_col] == 1
        edge_positions: np.ndarray = np.where(edge_mask)[0]

        if len(edge_positions) == 0:
            return interaction_matrix

        n_edges: int = len(edge_positions)
        n_pos: int = int(round(n_edges * 0.6))

        shuffled: np.ndarray = rng.permutation(n_edges)
        pos_idx: np.ndarray = shuffled[:n_pos]
        neg_idx: np.ndarray = shuffled[n_pos:]

        # Positive interaction strengths
        pos_strengths: np.ndarray = rng.normal(0.5, 0.2, size=len(pos_idx))
        # Negative interaction strengths
        neg_strengths: np.ndarray = rng.normal(-0.5, 0.2, size=len(neg_idx))

        for local_idx, strength in zip(pos_idx, pos_strengths):
            r, c = upper_row[edge_positions[local_idx]], upper_col[edge_positions[local_idx]]
            interaction_matrix[r, c] = strength
            interaction_matrix[c, r] = strength

        for local_idx, strength in zip(neg_idx, neg_strengths):
            r, c = upper_row[edge_positions[local_idx]], upper_col[edge_positions[local_idx]]
            interaction_matrix[r, c] = strength
            interaction_matrix[c, r] = strength

        return interaction_matrix

    @staticmethod
    def _normalize_rows(X: np.ndarray) -> np.ndarray:
        """Row-normalize *X* so each row sums to 1, handling zero-sum rows."""
        row_sums: np.ndarray = X.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-10, None)
        return X / row_sums
