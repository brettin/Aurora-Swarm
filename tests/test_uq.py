"""Unit tests for aurora_swarm.uq (Lab 3 semantic uncertainty).

Requires optional dependencies: pip install -e ".[uq]"
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from aurora_swarm.uq.semantic_entropy import (
    cluster_embeddings,
    semantic_entropy,
)
from aurora_swarm.uq.kle import (
    kernel_language_entropy,
    kernel_matrix,
    von_neumann_entropy,
    density_matrix_from_kernel,
)
from aurora_swarm.uq.probes import (
    train_probe,
    predict_semantic_entropy,
    save_probe,
    load_probe,
)


# ---------------------------------------------------------------------------
# Semantic entropy
# ---------------------------------------------------------------------------

def test_semantic_entropy_single_sample():
    """Single embedding => entropy 0."""
    emb = np.random.randn(1, 8).astype(np.float64)
    assert semantic_entropy(emb) == 0.0


def test_semantic_entropy_two_equal_clusters():
    """Two tight clusters of equal size => higher entropy than one cluster."""
    # Two clusters: points [0,0,...] and [1,1,...]
    n = 10
    d = 16
    c0 = np.zeros((n, d))
    c0[:, 0] = 1.0
    c1 = np.zeros((n, d))
    c1[:, 1] = 1.0
    embeddings = np.vstack([c0, c1]).astype(np.float64)
    h_two = semantic_entropy(embeddings, similarity_threshold=0.5)
    # All same => one cluster => low entropy
    same = np.tile(embeddings[:1], (2 * n, 1))
    h_one = semantic_entropy(same, similarity_threshold=0.99)
    assert h_two > h_one
    # Two equal clusters => entropy ln(2) in nats
    assert 0.5 < h_two < 1.0


def test_cluster_embeddings_shape():
    """cluster_embeddings returns one label per row."""
    embeddings = np.random.randn(6, 4).astype(np.float64)
    labels = cluster_embeddings(embeddings, similarity_threshold=0.9)
    assert labels.shape == (6,)
    assert np.all(labels >= 1)


# ---------------------------------------------------------------------------
# KLE
# ---------------------------------------------------------------------------

def test_von_neumann_entropy_known():
    """Von Neumann entropy for diagonal [p, 1-p] is -p log p - (1-p) log (1-p)."""
    p = 0.3
    rho = np.diag([p, 1.0 - p]).astype(np.float64)
    h = von_neumann_entropy(rho)
    expected = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    assert abs(h - expected) < 1e-10


def test_kernel_language_entropy_single_sample():
    """Single embedding => KLE 0."""
    emb = np.random.randn(1, 8).astype(np.float64)
    assert kernel_language_entropy(emb) == 0.0


def test_kle_two_samples():
    """Two orthogonal vectors => non-zero entropy."""
    emb = np.eye(2).astype(np.float64)  # 2x2 identity
    h = kernel_language_entropy(emb)
    assert h >= 0.0
    # Cosine kernel on orthonormal => K = I, rho = I/2, eigenvalues 1/2, 1/2 => H = ln(2)
    assert 0.6 < h < 0.7


def test_density_matrix_trace_one():
    """Density matrix from kernel has trace 1."""
    K = np.array([[1.0, 0.5], [0.5, 1.0]])
    rho = density_matrix_from_kernel(K)
    assert abs(np.trace(rho) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def test_train_probe_linear():
    """Train probe on linear relation; predict recovers it."""
    n, d = 20, 5
    np.random.seed(42)
    X = np.random.randn(n, d).astype(np.float64)
    true_coef = np.random.randn(d).astype(np.float64)
    true_intercept = 1.5
    y = X @ true_coef + true_intercept + 0.01 * np.random.randn(n)
    coef, intercept = train_probe(X, y, ridge_alpha=0.01)
    pred = predict_semantic_entropy(X, coef, intercept)
    mse = np.mean((pred - y) ** 2)
    assert mse < 0.1


def test_save_load_probe():
    """Save and load probe; predictions match."""
    coef = np.array([1.0, -0.5, 0.2])
    intercept = 0.3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.npz"
        save_probe(path, coef, intercept, metadata={"version": 1})
        loaded_coef, loaded_intercept, meta = load_probe(path)
        np.testing.assert_array_almost_equal(loaded_coef, coef)
        assert abs(loaded_intercept - intercept) < 1e-10
        assert meta is not None and meta.get("version") == 1
    X = np.random.randn(4, 3)
    pred_orig = predict_semantic_entropy(X, coef, intercept)
    pred_loaded = predict_semantic_entropy(X, loaded_coef, loaded_intercept)
    np.testing.assert_array_almost_equal(pred_orig, pred_loaded)


def test_predict_semantic_entropy_1d():
    """Single row hidden state returns scalar-like array."""
    coef = np.array([0.1, 0.2])
    intercept = 0.0
    x = np.array([1.0, 2.0])
    out = predict_semantic_entropy(x, coef, intercept)
    assert out.shape == (1,) or out.ndim == 1
    assert abs(float(out[0]) - 0.5) < 1e-10
