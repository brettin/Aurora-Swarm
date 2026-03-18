"""Kernel language entropy (KLE) for Lab 3.

Build a kernel matrix from embeddings, normalize to a density matrix,
and compute von Neumann entropy H = -tr(rho log rho).
"""

from __future__ import annotations

import numpy as np


def _cosine_kernel(embeddings: np.ndarray) -> np.ndarray:
    """Gram matrix of cosine similarities (linear kernel on normalized vectors)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms
    K = X @ X.T
    return np.clip(K, -1.0, 1.0)


def _rbf_kernel(embeddings: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """RBF (Gaussian) kernel. Gamma defaults to 1 / n_features."""
    if gamma is None:
        gamma = 1.0 / embeddings.shape[1] if embeddings.shape[1] > 0 else 1.0
    sq_dists = np.sum(embeddings**2, axis=1, keepdims=True) + np.sum(
        embeddings**2, axis=1
    ) - 2 * (embeddings @ embeddings.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-gamma * sq_dists)


def kernel_matrix(
    embeddings: np.ndarray,
    kernel: str = "cosine",
    **kwargs: float,
) -> np.ndarray:
    """Compute kernel matrix from embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, d).
    kernel : str
        "cosine" (default) or "rbf".
    **kwargs
        For "rbf": gamma (optional).

    Returns
    -------
    np.ndarray
        Shape (n, n), symmetric positive semi-definite.
    """
    if kernel == "cosine":
        return _cosine_kernel(embeddings)
    if kernel == "rbf":
        return _rbf_kernel(embeddings, **kwargs)
    raise ValueError(f"Unknown kernel: {kernel}")


def density_matrix_from_kernel(K: np.ndarray) -> np.ndarray:
    """Normalize kernel matrix to density matrix (trace 1, PSD).

    Uses K / trace(K). For PSD K with non-negative entries this yields
    a valid density matrix. For cosine kernel, values can be negative;
    we shift by minimum and then normalize so trace = 1 (or use K^2 for
    PSD guarantee). Standard approach: use K as Gram, then rho = K / tr(K).
    K from cosine is PSD (Gram of normalized vectors).
    """
    K = np.asarray(K, dtype=np.float64)
    # Ensure PSD: if K has negative values (cosine can), use (K + 1) / 2
    # so eigenvalues are in [0, 1], or just use K for cosine (already PSD).
    trace = np.trace(K)
    if trace <= 0:
        trace = np.sum(K)
    if trace <= 0:
        trace = 1.0
    rho = K / trace
    return rho


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-10) -> float:
    """Von Neumann entropy -sum lambda_i log(lambda_i) for eigenvalues of rho.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (trace 1, PSD).
    eps : float
        Clamp eigenvalues below this to avoid log(0).

    Returns
    -------
    float
        Entropy in nats.
    """
    eigs = np.linalg.eigvalsh(rho)
    eigs = np.maximum(eigs, eps)
    eigs = eigs / np.sum(eigs)
    eigs = np.maximum(eigs, eps)
    return float(-np.sum(eigs * np.log(eigs)))


def kernel_language_entropy(
    embeddings: np.ndarray,
    kernel: str = "cosine",
    **kernel_kwargs: float,
) -> float:
    """Compute kernel language entropy (KLE) for a set of response embeddings.

    Builds kernel matrix K from embeddings, normalizes to density matrix,
    and returns von Neumann entropy.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, d). One row per response.
    kernel : str
        "cosine" (default) or "rbf".
    **kernel_kwargs
        Passed to kernel_matrix (e.g. gamma for rbf).

    Returns
    -------
    float
        KLE in nats. Returns 0.0 if n <= 1.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 0.0

    K = kernel_matrix(embeddings, kernel=kernel, **kernel_kwargs)
    rho = density_matrix_from_kernel(K)
    return von_neumann_entropy(rho)
