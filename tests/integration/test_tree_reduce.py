"""Integration tests for Pattern 3 — Tree-Reduce against live vLLM endpoints."""

import pytest

from aurora_swarm.patterns.tree_reduce import tree_reduce


@pytest.mark.asyncio
async def test_tree_reduce(vllm_pool):
    """Leaf agents answer, supervisors recursively summarise to a single result."""
    result = await tree_reduce(
        pool=vllm_pool,
        prompt="Name one benefit of renewable energy. Keep it to one sentence.",
        reduce_prompt=(
            "Below are several responses. Merge them into a concise summary "
            "(level {level}):\n\n{responses}"
        ),
        fanin=4,
    )

    assert result.success, f"Tree-reduce failed: {result.error}"
    assert len(result.text.strip()) > 0


@pytest.mark.asyncio
async def test_tree_reduce_with_items(vllm_pool):
    """tree_reduce with explicit items scatters different tasks to leaf agents."""
    molecules = ["water", "methane", "ethanol", "glucose"]
    result = await tree_reduce(
        pool=vllm_pool,
        prompt="Describe the molecular formula of {item} in one sentence.",
        reduce_prompt=(
            "Merge the following molecular descriptions into a single summary "
            "(level {level}):\n\n{responses}"
        ),
        fanin=4,
        items=molecules,
    )

    assert result.success, f"Tree-reduce with items failed: {result.error}"
    assert len(result.text.strip()) > 0
