"""Pattern 2 — Scatter-Gather.

Distribute different work items across agents and gather results in
input order.
"""

from __future__ import annotations

from typing import Any

from aurora_swarm.pool import AgentPool, Response


async def scatter_gather(
    pool: AgentPool,
    prompts: list[str],
) -> list[Response]:
    """Send ``prompts[i]`` to ``agent[i % pool.size]``, gather in input order.

    If there are more prompts than agents the work wraps round-robin.
    """
    return await pool.send_all(prompts)


async def map_gather(
    pool: AgentPool,
    items: list[Any],
    prompt_template: str,
) -> list[Response]:
    """Higher-level scatter: format *prompt_template* with each item.

    The template must contain an ``{item}`` placeholder.

    Parameters
    ----------
    pool:
        Agent pool.
    items:
        Work items — each is ``str()``-ified and inserted into the
        template.
    prompt_template:
        Prompt with an ``{item}`` placeholder.
    """
    prompts = [prompt_template.replace("{item}", str(it)) for it in items]
    return await scatter_gather(pool, prompts)
