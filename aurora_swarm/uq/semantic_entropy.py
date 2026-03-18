"""Cluster-based semantic entropy for Lab 3.

Cluster response embeddings by semantic similarity, then compute
entropy over the cluster distribution: H = -sum p_c log(p_c).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity (embeddings assumed normalized)."""
    # embeddings: (n, d); normalize if not already
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms
    return X @ X.T


def _similarity_to_distance(sim: np.ndarray) -> np.ndarray:
    """Convert similarity in [0,1] or [-1,1] to distance (0 = same, 2 = opposite)."""
    # 1 - sim gives distance in [0, 2] for sim in [-1, 1]
    return 1.0 - np.clip(sim, -1.0, 1.0)


def cluster_embeddings(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.9,
    method: str = "average",
) -> np.ndarray:
    """Assign cluster label per embedding using agglomerative clustering.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, d). One row per response.
    similarity_threshold : float
        Merge clusters if similarity >= this (distance <= 1 - threshold).
        Used to cut the linkage tree.
    method : str
        Linkage method: "average", "single", or "complete".

    Returns
    -------
    np.ndarray
        Integer cluster labels, shape (n,). Cluster ids are 1..C.
    """
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if embeddings.shape[0] == 1:
        return np.array([1], dtype=np.int64)

    sim = _cosine_similarity_matrix(embeddings)
    dist = _similarity_to_distance(sim)
    # Condensed distance matrix (upper triangle, row-major)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    # Cut at distance = 1 - similarity_threshold so that pairs with
    # similarity >= similarity_threshold end up in same cluster
    dist_cut = 1.0 - similarity_threshold
    labels = fcluster(Z, t=dist_cut, criterion="distance")
    return labels.astype(np.int64)


def semantic_entropy(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.9,
    linkage_method: str = "average",
) -> float:
    """Compute semantic entropy over a set of response embeddings.

    Clusters embeddings by semantic similarity, then computes
    H = -sum_c p_c log(p_c) where p_c = n_c / n.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, d). One row per response (n = S samples per prompt).
    similarity_threshold : float
        Merge clusters if cosine similarity >= this (default 0.9).
    linkage_method : str
        Linkage method for hierarchical clustering (default "average").

    Returns
    -------
    float
        Semantic entropy in nats. Returns 0.0 if n <= 1.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 0.0

    labels = cluster_embeddings(
        embeddings,
        similarity_threshold=similarity_threshold,
        method=linkage_method,
    )
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    # H = -sum p_c log(p_c); use 0*log(0)=0
    probs = np.maximum(probs, 1e-12)
    return float(-np.sum(probs * np.log(probs)))
