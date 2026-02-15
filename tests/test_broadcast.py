"""Tests for Pattern 1 — Broadcast."""

import pytest

from aurora_swarm.patterns.broadcast import broadcast, broadcast_and_reduce


@pytest.mark.asyncio
async def test_broadcast(mock_pool):
    """Broadcast a prompt to 4 agents; expect 4 echo responses."""
    responses = await broadcast(mock_pool, "Say hello")

    assert len(responses) == 4
    for r in responses:
        assert r.success
        assert "Say hello" in r.text


@pytest.mark.asyncio
async def test_broadcast_and_reduce(mock_pool):
    """Broadcast then reduce — should yield a single synthesised response."""
    result = await broadcast_and_reduce(
        mock_pool,
        prompt="Give a fact",
        reduce_prompt="Summarise these responses:\n{responses}",
        reducer_agent_index=0,
    )

    assert result.success
    # The reducer echoes the filled reduce_prompt, which must contain
    # the original agent outputs (each of which contains "Give a fact").
    assert "Give a fact" in result.text
