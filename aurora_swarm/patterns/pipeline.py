"""Pattern 5 — Pipeline (Multi-Stage DAG).

Defines a sequence of stages, each served by a pool of agents.  The
output of one stage flows as input to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aurora_swarm.pool import AgentPool, Response


@dataclass
class Stage:
    """One step of a pipeline.

    Attributes
    ----------
    name:
        Human-readable label for the stage.
    prompt_template:
        Must contain ``{input}`` which is replaced with the previous
        stage's output (or the initial input for the first stage).
    n_agents:
        How many agents this stage should use.
    output_transform:
        ``f(responses) -> Any`` — reshapes the list of responses into
        a single value to feed the next stage.  If ``None``, responses
        are concatenated with newlines.
    output_filter:
        ``f(response) -> bool`` — drops responses that return ``False``
        before the transform step.
    """

    name: str
    prompt_template: str
    n_agents: int
    output_transform: Callable[[list[Response]], Any] | None = None
    output_filter: Callable[[Response], bool] | None = None


def _default_transform(responses: list[Response]) -> str:
    """Concatenate successful response texts."""
    return "\n".join(r.text for r in responses if r.success)


async def run_pipeline(
    pool: AgentPool,
    stages: list[Stage],
    initial_input: Any,
    reuse_agents: bool = True,
) -> Any:
    """Execute stages sequentially; the output of each flows to the next.

    Parameters
    ----------
    pool:
        The full agent pool.
    stages:
        Ordered list of pipeline stages.
    initial_input:
        Value substituted into ``{input}`` for the first stage.
    reuse_agents:
        If ``True`` all stages draw agents from the same pool (up to
        ``n_agents``).  If ``False`` the pool is partitioned so each
        stage receives a dedicated, non-overlapping subset.

    Returns
    -------
    Any
        The transformed output of the final stage.
    """
    current_input = initial_input
    offset = 0  # used when partitioning

    for stage in stages:
        # select agents for this stage
        if reuse_agents:
            stage_pool = pool.select(list(range(min(stage.n_agents, pool.size))))
        else:
            end = min(offset + stage.n_agents, pool.size)
            stage_pool = pool.slice(offset, end)
            offset = end

        # build prompts
        prompt = stage.prompt_template.replace("{input}", str(current_input))

        # broadcast the same prompt to all agents in this stage
        responses = await stage_pool.broadcast_prompt(prompt)

        # optional filter
        if stage.output_filter is not None:
            responses = [r for r in responses if stage.output_filter(r)]

        # transform
        transform = stage.output_transform or _default_transform
        current_input = transform(responses)

    return current_input


async def fan_out_fan_in(
    pool: AgentPool,
    prompt: str,
    collect_prompt: str,
    n_workers: int | None = None,
) -> Response:
    """Convenience two-stage pipeline: broadcast then collect.

    Parameters
    ----------
    pool:
        Agent pool.
    prompt:
        Sent to all workers.
    collect_prompt:
        Template with ``{responses}`` placeholder for the collector.
    n_workers:
        How many workers to use (default: all).
    """
    if n_workers is not None:
        worker_pool = pool.select(list(range(min(n_workers, pool.size))))
    else:
        worker_pool = pool

    responses = await worker_pool.broadcast_prompt(prompt)
    combined = "\n---\n".join(r.text for r in responses if r.success)
    filled = collect_prompt.replace("{responses}", combined)
    return await pool.post(0, filled)
