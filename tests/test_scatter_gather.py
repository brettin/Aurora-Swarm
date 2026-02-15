"""Tests for Pattern 2 — Scatter-Gather."""

import pytest

from aurora_swarm.patterns.scatter_gather import scatter_gather, map_gather


@pytest.mark.asyncio
async def test_scatter_gather(mock_pool):
    """Scatter 4 unique prompts to 4 agents; responses arrive in order."""
    prompts = [f"task-{i}" for i in range(4)]
    responses = await scatter_gather(mock_pool, prompts)

    assert len(responses) == 4
    for i, r in enumerate(responses):
        assert r.success
        assert f"task-{i}" in r.text


@pytest.mark.asyncio
async def test_map_gather(mock_pool):
    """map_gather formats a template with each item and scatters."""
    items = ["alpha", "beta", "gamma", "delta"]
    responses = await map_gather(mock_pool, items, "Process: {item}")

    assert len(responses) == 4
    for item, r in zip(items, responses):
        assert r.success
        assert item in r.text
