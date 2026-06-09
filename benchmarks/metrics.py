"""Network evaluation metrics for comparing inferred and ground truth graphs."""

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import (
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


class NetworkMetrics:
    """Collection of static methods for evaluating inferred microbial networks.

    All adjacency matrices are assumed to be symmetric ``(p, p)`` binary matrices.
    Score matrices are symmetric ``(p, p)`` matrices of continuous values (higher
    means more likely to be an edge).  Metrics are computed on the **upper
    triangle only** (``k=1``) to avoid double-counting undirected edges.
    """

    # ------------------------------------------------------------------
    # Edge-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def precision_recall_f1(
        true_adj: np.ndarray, pred_adj: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 from binary adjacency matrices.

        Only the strict upper triangle (``k=1``) is used.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.

        Returns:
            ``(precision, recall, f1)`` — each in ``[0, 1]``.  Returns
            ``(0.0, 0.0, 0.0)`` when there are no true edges and no predicted
            edges.
        """
        row, col = np.triu_indices(true_adj.shape[0], k=1)
        y_true: np.ndarray = true_adj[row, col].astype(bool)
        y_pred: np.ndarray = pred_adj[row, col].astype(bool)

        tp: int = int(np.sum(y_true & y_pred))
        fp: int = int(np.sum(~y_true & y_pred))
        fn: int = int(np.sum(y_true & ~y_pred))

        precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1: float = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def auprc(true_adj: np.ndarray, pred_scores: np.ndarray) -> float:
        """Area under the precision-recall curve.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_scores: Continuous edge scores (higher = more likely).

        Returns:
            AUPRC in ``[0, 1]``, or ``float('nan')`` if undefined.
        """
        row, col = np.triu_indices(true_adj.shape[0], k=1)
        y_true: np.ndarray = true_adj[row, col].astype(int)
        y_score: np.ndarray = pred_scores[row, col]

        # Need at least one positive and one negative sample
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    @staticmethod
    def auroc(true_adj: np.ndarray, pred_scores: np.ndarray) -> float:
        """Area under the ROC curve.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_scores: Continuous edge scores (higher = more likely).

        Returns:
            AUROC in ``[0, 1]``, or ``float('nan')`` if undefined.
        """
        row, col = np.triu_indices(true_adj.shape[0], k=1)
        y_true: np.ndarray = true_adj[row, col].astype(int)
        y_score: np.ndarray = pred_scores[row, col]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    # ------------------------------------------------------------------
    # Topology-level metrics
    # ------------------------------------------------------------------

    @staticmethod
    def degree_distribution_similarity(
        true_adj: np.ndarray, pred_adj: np.ndarray
    ) -> float:
        """Kolmogorov-Smirnov statistic between degree distributions.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.

        Returns:
            KS statistic in ``[0, 1]`` (0 = identical distributions).
        """
        true_deg: np.ndarray = true_adj.sum(axis=1).astype(float)
        pred_deg: np.ndarray = pred_adj.sum(axis=1).astype(float)

        if len(true_deg) == 0 or len(pred_deg) == 0:
            return float("nan")

        ks_stat: float = float(sp_stats.ks_2samp(true_deg, pred_deg).statistic)
        return ks_stat

    @staticmethod
    def clustering_coefficient_correlation(
        true_adj: np.ndarray, pred_adj: np.ndarray
    ) -> float:
        """Pearson correlation of per-node clustering coefficients.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.

        Returns:
            Pearson *r* in ``[-1, 1]``, or ``float('nan')`` if undefined.
        """
        G_true: nx.Graph = nx.from_numpy_array(true_adj)
        G_pred: nx.Graph = nx.from_numpy_array(pred_adj)

        cc_true: Dict[int, float] = nx.clustering(G_true)
        cc_pred: Dict[int, float] = nx.clustering(G_pred)

        p: int = true_adj.shape[0]
        vals_true: np.ndarray = np.array([cc_true[i] for i in range(p)])
        vals_pred: np.ndarray = np.array([cc_pred[i] for i in range(p)])

        if vals_true.std() == 0.0 or vals_pred.std() == 0.0:
            return float("nan")

        r: float = float(np.corrcoef(vals_true, vals_pred)[0, 1])
        return r

    @staticmethod
    def modularity(adj: np.ndarray, seed: int = 42) -> float:
        """Newman-Girvan modularity *Q* via Louvain community detection.

        Args:
            adj: Binary adjacency matrix.
            seed: Random seed for Louvain.

        Returns:
            Modularity *Q* in ``[-0.5, 1]``, or ``float('nan')`` for
            degenerate graphs.
        """
        G: nx.Graph = nx.from_numpy_array(adj)

        if G.number_of_edges() == 0:
            return float("nan")

        try:
            communities: list = nx.community.louvain_communities(G, seed=seed)
            q: float = float(nx.community.modularity(G, communities))
            return q
        except Exception:
            return float("nan")

    @staticmethod
    def community_nmi(
        true_adj: np.ndarray, pred_adj: np.ndarray, seed: int = 42
    ) -> float:
        """Normalized mutual information between community assignments.

        Communities are detected with Louvain on each graph independently.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.
            seed: Random seed for Louvain.

        Returns:
            NMI in ``[0, 1]``, or ``float('nan')`` on failure.
        """
        p: int = true_adj.shape[0]

        def _community_labels(adj: np.ndarray) -> Optional[np.ndarray]:
            G: nx.Graph = nx.from_numpy_array(adj)
            if G.number_of_edges() == 0:
                return None
            try:
                communities: list = nx.community.louvain_communities(G, seed=seed)
            except Exception:
                return None
            labels: np.ndarray = np.zeros(p, dtype=int)
            for label_idx, comm in enumerate(communities):
                for node in comm:
                    labels[node] = label_idx
            return labels

        labels_true: Optional[np.ndarray] = _community_labels(true_adj)
        labels_pred: Optional[np.ndarray] = _community_labels(pred_adj)

        if labels_true is None or labels_pred is None:
            return float("nan")

        return float(normalized_mutual_info_score(labels_true, labels_pred))

    @staticmethod
    def hub_recovery_rate(
        true_adj: np.ndarray, pred_adj: np.ndarray, percentile: float = 90.0
    ) -> float:
        """Fraction of true hubs that are also identified as hubs in the predicted graph.

        A *hub* is any node whose degree is at or above the given percentile
        of its graph's degree distribution.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.
            percentile: Degree percentile threshold (default 90).

        Returns:
            Hub recovery rate in ``[0, 1]``, or ``float('nan')`` when there
            are no true hubs.
        """
        true_deg: np.ndarray = true_adj.sum(axis=1)
        pred_deg: np.ndarray = pred_adj.sum(axis=1)

        true_thresh: float = float(np.percentile(true_deg, percentile))
        pred_thresh: float = float(np.percentile(pred_deg, percentile))

        true_hubs: np.ndarray = np.where(true_deg >= true_thresh)[0]
        pred_hubs: set = set(np.where(pred_deg >= pred_thresh)[0].tolist())

        if len(true_hubs) == 0:
            return float("nan")

        recovered: int = sum(1 for h in true_hubs if h in pred_hubs)
        return recovered / len(true_hubs)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        true_adj: np.ndarray,
        pred_adj: np.ndarray,
        pred_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all available metrics and return a flat dict.

        Args:
            true_adj: Ground truth binary adjacency matrix.
            pred_adj: Predicted binary adjacency matrix.
            pred_scores: Optional continuous edge scores.  When *None*, the
                predicted adjacency matrix (cast to float) is used as scores.

        Returns:
            Dict with keys: ``precision``, ``recall``, ``f1``, ``auprc``,
            ``auroc``, ``degree_ks``, ``cluster_corr``, ``modularity_true``,
            ``modularity_pred``, ``community_nmi``, ``hub_recovery``.
        """
        if pred_scores is None:
            pred_scores = pred_adj.astype(float)

        precision, recall, f1 = cls.precision_recall_f1(true_adj, pred_adj)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auprc": cls.auprc(true_adj, pred_scores),
            "auroc": cls.auroc(true_adj, pred_scores),
            "degree_ks": cls.degree_distribution_similarity(true_adj, pred_adj),
            "cluster_corr": cls.clustering_coefficient_correlation(true_adj, pred_adj),
            "modularity_true": cls.modularity(true_adj),
            "modularity_pred": cls.modularity(pred_adj),
            "community_nmi": cls.community_nmi(true_adj, pred_adj),
            "hub_recovery": cls.hub_recovery_rate(true_adj, pred_adj),
        }
