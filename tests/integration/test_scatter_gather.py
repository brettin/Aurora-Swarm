"""Integration tests for Pattern 2 — Scatter-Gather against live vLLM endpoints."""

import pytest

from aurora_swarm.patterns.scatter_gather import scatter_gather, map_gather


@pytest.mark.asyncio
async def test_scatter_gather(vllm_pool):
    """Scatter unique prompts across agents; responses arrive in input order."""
    prompts = [
        f"What is {n} + {n}? Reply with only the number."
        for n in range(vllm_pool.size)
    ]
    responses = await scatter_gather(vllm_pool, prompts)

    assert len(responses) == len(prompts)
    for i, r in enumerate(responses):
        assert r.success, f"Agent {r.agent_index} failed on prompt {i}: {r.error}"
        assert len(r.text.strip()) > 0


@pytest.mark.asyncio
async def test_map_gather(vllm_pool):
    """map_gather formats a template with each item and scatters."""
    elements = ["hydrogen", "carbon", "oxygen", "nitrogen"]
    responses = await map_gather(
        vllm_pool,
        elements,
        "What is the atomic number of {item}? Reply with only the number.",
    )

    assert len(responses) == len(elements)
    for item, r in zip(elements, responses):
        assert r.success, f"Failed for {item}: {r.error}"
        assert len(r.text.strip()) > 0
