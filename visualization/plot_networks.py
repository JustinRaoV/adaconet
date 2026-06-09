"""Network topology visualization for inferred co-occurrence networks."""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure


# Community colour palette (up to 12 communities)
_COMMUNITY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]


def _adjacency_to_graph(adj: np.ndarray) -> nx.Graph:
    """Convert a weighted adjacency matrix to a NetworkX graph (upper triangle only)."""
    G = nx.Graph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0:
                G.add_edge(i, j, weight=adj[i, j])
    return G


def _community_color_map(G: nx.Graph, seed: int = 42) -> Dict[int, str]:
    """Assign a colour to each node based on Louvain community detection."""
    communities = nx.community.louvain_communities(G, seed=seed)
    node_color: Dict[int, str] = {}
    for idx, comm in enumerate(communities):
        color = _COMMUNITY_COLORS[idx % len(_COMMUNITY_COLORS)]
        for node in comm:
            node_color[node] = color
    # Assign a default colour to isolated nodes not in any community
    for node in G.nodes():
        if node not in node_color:
            node_color[node] = "#cccccc"
    return node_color


def _network_stats_text(G: nx.Graph) -> str:
    """Return a one-line summary string for the graph title."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    try:
        modularity = nx.community.modularity(
            G, nx.community.louvain_communities(G, seed=42)
        )
    except Exception:
        modularity = 0.0
    return f"Nodes={n_nodes}  Edges={n_edges}  Density={density:.3f}  Mod={modularity:.3f}"


def plot_network_comparison(
    true_adj: np.ndarray,
    inferred_adjs: List[np.ndarray],
    method_names: List[str],
    save_path: Optional[str] = None,
) -> Figure:
    """Visualise the ground-truth network alongside inferred networks.

    Parameters
    ----------
    true_adj : np.ndarray
        Binary (or weighted) ground-truth adjacency matrix (P x P).
    inferred_adjs : list of np.ndarray
        List of binary adjacency matrices produced by each method.
    method_names : list of str
        Corresponding method names.
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    n_methods = len(method_names)
    n_cols = 1 + n_methods
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # Compute a single shared layout from the ground-truth graph
    true_graph = _adjacency_to_graph(true_adj)
    pos = nx.spring_layout(true_graph, seed=42)

    panels = [("Ground Truth", true_adj)] + list(zip(method_names, inferred_adjs))

    for ax, (label, adj) in zip(axes, panels):
        G = _adjacency_to_graph(adj)
        node_colors_map = _community_color_map(G if G.number_of_edges() > 0 else true_graph)

        # Node size proportional to degree (min 30, max 300)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [30 + 270 * (degrees[n] / max(max_deg, 1)) for n in G.nodes()]

        node_colors = [node_colors_map[n] for n in G.nodes()]

        # Edge properties
        edges = G.edges(data=True)
        edge_widths = [0.3 + 2.5 * abs(d.get("weight", 1.0)) for _, _, d in edges]
        edge_colors = [
            "#2ca02c" if d.get("weight", 0) >= 0 else "#d62728"
            for _, _, d in edges
        ]

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.85,
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
        )

        stats = _network_stats_text(G)
        ax.set_title(f"{label}\n{stats}", fontsize=11)
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig


def plot_scatter_correlations(
    true_adj: np.ndarray,
    pred_scores: np.ndarray,
    method_name: str,
    save_path: Optional[str] = None,
) -> Figure:
    """Scatter plot of true edge weights vs inferred scores, coloured by edge type.

    Edge types:
    - TP (True Positive):  true > 0 and score above median of non-zero scores
    - FP (False Positive): true == 0 and score above median of non-zero scores
    - FN (False Negative): true > 0 and score below median of non-zero scores

    Parameters
    ----------
    true_adj : np.ndarray
        Ground-truth adjacency matrix.
    pred_scores : np.ndarray
        Predicted continuous score matrix.
    method_name : str
        Name of the method (used in the title).
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    from scipy.stats import pearsonr, spearmanr

    # Extract upper-triangle entries
    idx = np.triu_indices_from(true_adj, k=1)
    y_true = true_adj[idx]
    y_pred = pred_scores[idx]

    # Threshold for classification: median of non-zero predicted scores
    nonzero_scores = y_pred[y_pred != 0]
    threshold = float(np.median(nonzero_scores)) if len(nonzero_scores) > 0 else 0.0

    # Classify edges
    is_tp = (y_true > 0) & (y_pred >= threshold)
    is_fp = (y_true == 0) & (y_pred >= threshold)
    is_fn = (y_true > 0) & (y_pred < threshold)
    is_tn = ~is_tp & ~is_fp & ~is_fn

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot order: TN first (background), then FP, FN, TP on top
    if np.any(is_tn):
        ax.scatter(y_true[is_tn], y_pred[is_tn], c="#cccccc", s=8, alpha=0.4, label="TN", rasterized=True)
    if np.any(is_fp):
        ax.scatter(y_true[is_fp], y_pred[is_fp], c="#ff7f0e", s=12, alpha=0.6, label="FP")
    if np.any(is_fn):
        ax.scatter(y_true[is_fn], y_pred[is_fn], c="#d62728", s=12, alpha=0.6, label="FN")
    if np.any(is_tp):
        ax.scatter(y_true[is_tp], y_pred[is_tp], c="#2ca02c", s=14, alpha=0.7, label="TP")

    # Correlation statistics
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    ax.set_xlabel("True Edge Weight", fontsize=12)
    ax.set_ylabel("Inferred Score", fontsize=12)
    ax.set_title(
        f"{method_name} — Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f}",
        fontsize=13,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    return fig
