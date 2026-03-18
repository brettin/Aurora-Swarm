"""Semantic entropy probes for Lab 3.

Train a linear probe on hidden states to predict semantic entropy
(from multi-sample computation), then predict from a single forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def train_probe(
    hidden_states: np.ndarray,
    target_entropy: np.ndarray,
    ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Fit a linear probe: hidden_states -> predicted semantic entropy.

    Uses ridge regression (closed-form). One row per sample; target_entropy
    is the semantic entropy for that prompt/sample set.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_samples, n_features). E.g. one row per training prompt,
        features = flattened last-layer hidden state at EOS.
    target_entropy : np.ndarray
        Shape (n_samples,). Semantic entropy for each prompt (from
        multi-sample semantic_entropy).
    ridge_alpha : float
        L2 regularization strength (default 1.0).

    Returns
    -------
    coef : np.ndarray
        Shape (n_features,).
    intercept : float
        Scalar intercept.
    """
    X = np.asarray(hidden_states, dtype=np.float64)
    y = np.asarray(target_entropy, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("hidden_states and target_entropy must have same length")
    # Add column of ones for intercept
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    # Ridge: (X'X + alpha I)^{-1} X' y
    XtX = X_aug.T @ X_aug
    XtX += ridge_alpha * np.eye(X_aug.shape[1])
    Xty = X_aug.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    intercept = float(beta[0])
    coef = np.asarray(beta[1:], dtype=np.float64)
    return coef, intercept


def predict_semantic_entropy(
    hidden_states: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Predict semantic entropy from hidden states using a trained probe.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n, n_features) or (n_features,). Single or multiple rows.
    coef : np.ndarray
        Shape (n_features,) from train_probe.
    intercept : float
        From train_probe.

    Returns
    -------
    np.ndarray
        Predicted entropy, shape (n,) or scalar if input was 1d.
    """
    X = np.asarray(hidden_states, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return (X @ coef + intercept).ravel()


def save_probe(
    path: str | Path,
    coef: np.ndarray,
    intercept: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save probe coefficients and optional metadata to disk.

    Uses .npz for arrays and a sidecar .json for metadata if provided.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coef=coef, intercept=np.array(intercept))
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_probe(
    path: str | Path,
) -> tuple[np.ndarray, float, dict[str, Any] | None]:
    """Load probe coefficients from disk.

    Returns
    -------
    coef : np.ndarray
    intercept : float
    metadata : dict or None
        From sidecar .json if present.
    """
    path = Path(path)
    data = np.load(path)
    coef = np.asarray(data["coef"])
    intercept = float(data["intercept"])
    meta_path = path.with_suffix(path.suffix + ".json")
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    return coef, intercept, metadata
