"""Lab 3: Train a semantic entropy probe (numpy-only linear ridge).

This helper trains a linear probe to predict semantic entropy from hidden-state
features. It does not extract hidden states; it assumes you already have:

- X: hidden-state feature matrix (.npy) with shape (n_prompts, n_features)
- y: target semantic entropy values (.npy) with shape (n_prompts,)

USAGE:
  python examples/lab3_train_probe.py \\
    --hidden-states X.npy \\
    --targets y.npy \\
    --out probe.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from aurora_swarm.uq.probes import save_probe, train_probe


def print_ts(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hidden-states", type=Path, required=True, help="Path to X.npy (n, d).")
    parser.add_argument("--targets", type=Path, required=True, help="Path to y.npy (n,).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    parser.add_argument("--out", type=Path, required=True, help="Output probe .npz path.")
    args = parser.parse_args()

    X = np.load(args.hidden_states)
    y = np.load(args.targets)
    if X.shape[0] != y.shape[0]:
        print("Error: X rows must match y length.", file=sys.stderr)
        return 1

    coef, intercept = train_probe(X, y, ridge_alpha=args.ridge_alpha)
    meta = {
        "ridge_alpha": args.ridge_alpha,
        "n_prompts": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    save_probe(args.out, coef, intercept, metadata=meta)
    print_ts(f"Saved probe to {args.out}")
    print_ts("Metadata: " + json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

