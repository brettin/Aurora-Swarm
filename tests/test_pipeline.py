"""Tests for Pattern 5 — Pipeline (Multi-Stage DAG)."""

import pytest

from aurora_swarm.patterns.pipeline import Stage, run_pipeline, fan_out_fan_in
from aurora_swarm.pool import Response


@pytest.mark.asyncio
async def test_pipeline_two_stages(mock_pool):
    """Two-stage pipeline: echo → summarise.  Output flows through."""
    stages = [
        Stage(
            name="echo",
            prompt_template="Echo: {input}",
            n_agents=2,
        ),
        Stage(
            name="summarise",
            prompt_template="Summarise: {input}",
            n_agents=2,
        ),
    ]

    result = await run_pipeline(
        pool=mock_pool,
        stages=stages,
        initial_input="hello world",
        reuse_agents=True,
    )

    # result is a string (default transform = concat)
    assert isinstance(result, str)
    # The second stage echoes back a prompt that contains the first
    # stage's output, which itself contains "hello world".
    assert "hello world" in result


@pytest.mark.asyncio
async def test_pipeline_with_filter(mock_pool):
    """output_filter can drop responses between stages."""

    def keep_all(r: Response) -> bool:
        return r.success

    stages = [
        Stage(
            name="generate",
            prompt_template="Generate: {input}",
            n_agents=4,
            output_filter=keep_all,
        ),
        Stage(
            name="refine",
            prompt_template="Refine: {input}",
            n_agents=2,
        ),
    ]

    result = await run_pipeline(mock_pool, stages, "data", reuse_agents=True)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_fan_out_fan_in(mock_pool):
    """Convenience two-stage: broadcast then collect."""
    result = await fan_out_fan_in(
        mock_pool,
        prompt="Analyse X",
        collect_prompt="Collect:\n{responses}",
        n_workers=4,
    )

    assert result.success
    assert "Analyse X" in result.text
