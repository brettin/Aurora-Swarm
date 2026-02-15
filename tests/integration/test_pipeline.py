"""Integration tests for Pattern 5 — Pipeline against live vLLM endpoints."""

import pytest

from aurora_swarm.patterns.pipeline import Stage, run_pipeline, fan_out_fan_in
from aurora_swarm.pool import Response


@pytest.mark.asyncio
async def test_pipeline_two_stages(vllm_pool):
    """Two-stage pipeline: generate ideas then refine them."""
    n = min(2, vllm_pool.size)
    stages = [
        Stage(
            name="brainstorm",
            prompt_template=(
                "Brainstorm one idea about the following topic in one sentence: {input}"
            ),
            n_agents=n,
        ),
        Stage(
            name="refine",
            prompt_template=(
                "Refine the following ideas into a single polished sentence: {input}"
            ),
            n_agents=n,
        ),
    ]

    result = await run_pipeline(
        pool=vllm_pool,
        stages=stages,
        initial_input="improving battery technology",
        reuse_agents=True,
    )

    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.asyncio
async def test_pipeline_with_filter(vllm_pool):
    """Pipeline with an output_filter that keeps only successful responses."""

    def keep_successful(r: Response) -> bool:
        return r.success

    n = min(2, vllm_pool.size)
    stages = [
        Stage(
            name="generate",
            prompt_template="List one fact about: {input}",
            n_agents=n,
            output_filter=keep_successful,
        ),
        Stage(
            name="summarise",
            prompt_template="Summarise the following: {input}",
            n_agents=n,
        ),
    ]

    result = await run_pipeline(vllm_pool, stages, "climate change", reuse_agents=True)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.asyncio
async def test_fan_out_fan_in(vllm_pool):
    """Convenience two-stage: broadcast then collect."""
    result = await fan_out_fan_in(
        vllm_pool,
        prompt="Name one application of machine learning in healthcare.",
        collect_prompt=(
            "The following are responses from multiple agents. "
            "Combine them into a single concise paragraph.\n\n{responses}"
        ),
        n_workers=min(4, vllm_pool.size),
    )

    assert result.success, f"fan_out_fan_in failed: {result.error}"
    assert len(result.text.strip()) > 0
