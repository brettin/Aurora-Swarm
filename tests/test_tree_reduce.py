"""Tests for Pattern 3 — Hierarchical Tree-Reduce."""

import pytest

from aurora_swarm.patterns.tree_reduce import tree_reduce


@pytest.mark.asyncio
async def test_tree_reduce(mock_pool_8):
    """8 leaf agents with fanin=4 → 2 supervisors → 1 final answer."""
    result = await tree_reduce(
        pool=mock_pool_8,
        prompt="Leaf work",
        reduce_prompt="Summarise level {level}:\n{responses}",
        fanin=4,
    )

    assert result.success
    # The final response is an echo of the last supervisor prompt,
    # which itself contains echoes of earlier work — just verify
    # it's non-empty and came through.
    assert len(result.text) > 0
    # The word "Summarise" should appear because the supervisor echoes it.
    assert "Summarise" in result.text


@pytest.mark.asyncio
async def test_tree_reduce_with_items(mock_pool_8):
    """tree_reduce with explicit items scatters to leaf agents."""
    items = [f"mol-{i}" for i in range(8)]
    result = await tree_reduce(
        pool=mock_pool_8,
        prompt="Evaluate {item}",
        reduce_prompt="Merge level {level}:\n{responses}",
        fanin=4,
        items=items,
    )

    assert result.success
    assert len(result.text) > 0
