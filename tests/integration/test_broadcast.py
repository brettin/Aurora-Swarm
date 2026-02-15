"""Integration tests for Pattern 1 — Broadcast against live vLLM endpoints."""

import pytest

from aurora_swarm.patterns.broadcast import broadcast, broadcast_and_reduce


@pytest.mark.asyncio
async def test_broadcast(vllm_pool):
    """Broadcast a prompt to all agents; every agent should respond successfully."""
    responses = await broadcast(vllm_pool, "Respond with one word: hello.")

    assert len(responses) == vllm_pool.size
    for r in responses:
        assert r.success, f"Agent {r.agent_index} failed: {r.error}"
        assert len(r.text.strip()) > 0


@pytest.mark.asyncio
async def test_broadcast_and_reduce(vllm_pool):
    """Broadcast then reduce — should yield a single synthesised response."""
    result = await broadcast_and_reduce(
        vllm_pool,
        prompt="Name one element from the periodic table.",
        reduce_prompt=(
            "Below are responses from several agents. "
            "Summarise them in a single sentence.\n\n{responses}"
        ),
        reducer_agent_index=0,
    )

    assert result.success, f"Reducer failed: {result.error}"
    assert len(result.text.strip()) > 0
