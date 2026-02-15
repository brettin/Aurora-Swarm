"""Aggregation strategies for agent responses.

Every aggregator silently skips responses with ``success=False`` unless
``include_failures=True`` is passed.
"""

from __future__ import annotations

import json
import statistics as _stats
from collections import Counter
from typing import Any, Callable, Sequence

from aurora_swarm.pool import Response


def _ok(responses: Sequence[Response], include_failures: bool = False) -> list[Response]:
    """Filter to successful responses unless explicitly including failures."""
    if include_failures:
        return list(responses)
    return [r for r in responses if r.success]


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def majority_vote(
    responses: Sequence[Response],
    include_failures: bool = False,
) -> tuple[str, float]:
    """Return ``(winner, confidence)`` where *confidence* is the vote fraction.

    Responses are stripped and compared case-insensitively.
    """
    good = _ok(responses, include_failures)
    if not good:
        return ("", 0.0)
    counts = Counter(r.text.strip().lower() for r in good)
    winner, n = counts.most_common(1)[0]
    return (winner, n / len(good))


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def concat(
    responses: Sequence[Response],
    separator: str = "\n",
    include_failures: bool = False,
) -> str:
    """Join all response texts with *separator*."""
    good = _ok(responses, include_failures)
    return separator.join(r.text for r in good)


# ---------------------------------------------------------------------------
# Quality selection
# ---------------------------------------------------------------------------

def best_of(
    responses: Sequence[Response],
    score_fn: Callable[[Response], float],
    include_failures: bool = False,
) -> Response:
    """Return the single highest-scoring response."""
    good = _ok(responses, include_failures)
    if not good:
        return Response(success=False, text="", error="No responses to select from")
    return max(good, key=score_fn)


def top_k(
    responses: Sequence[Response],
    k: int,
    score_fn: Callable[[Response], float],
    include_failures: bool = False,
) -> list[Response]:
    """Return the *k* highest-scoring responses (descending)."""
    good = _ok(responses, include_failures)
    return sorted(good, key=score_fn, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Structured data
# ---------------------------------------------------------------------------

def structured_merge(
    responses: Sequence[Response],
    include_failures: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Parse each response as JSON and merge into a flat list.

    Returns ``(merged_list, errors)`` where *errors* captures parse
    failures with the agent index and error message.
    """
    good = _ok(responses, include_failures)
    merged: list[Any] = []
    errors: list[dict[str, Any]] = []
    for r in good:
        try:
            obj = json.loads(r.text)
            if isinstance(obj, list):
                merged.extend(obj)
            else:
                merged.append(obj)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append({"agent_index": r.agent_index, "error": str(exc)})
    return merged, errors


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------

def statistics(
    responses: Sequence[Response],
    extract_fn: Callable[[Response], float] | None = None,
    include_failures: bool = False,
) -> dict[str, float]:
    """Compute summary statistics over numeric response values.

    If *extract_fn* is ``None``, response text is converted to float
    directly.

    Returns dict with keys ``mean``, ``std``, ``median``, ``min``, ``max``.
    """
    good = _ok(responses, include_failures)
    if extract_fn is None:
        values = [float(r.text.strip()) for r in good]
    else:
        values = [extract_fn(r) for r in good]

    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": _stats.mean(values),
        "std": _stats.stdev(values) if len(values) > 1 else 0.0,
        "median": _stats.median(values),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def failure_report(responses: Sequence[Response]) -> dict[str, Any]:
    """Return a diagnostic summary of successes and failures.

    Keys: ``total``, ``success_count``, ``failure_count``, ``failures``
    (list of ``{agent_index, error}`` dicts).
    """
    total = len(responses)
    failures = [
        {"agent_index": r.agent_index, "error": r.error}
        for r in responses
        if not r.success
    ]
    return {
        "total": total,
        "success_count": total - len(failures),
        "failure_count": len(failures),
        "failures": failures,
    }
