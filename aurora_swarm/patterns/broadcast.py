"""Pattern 1 — Broadcast.

Send the identical prompt to every agent and collect all responses.
"""

from __future__ import annotations

from aurora_swarm.pool import AgentPool, Response


async def broadcast(pool: AgentPool, prompt: str) -> list[Response]:
    """Send *prompt* to every agent in *pool*, return all responses in order."""
    return await pool.broadcast_prompt(prompt)


async def broadcast_and_reduce(
    pool: AgentPool,
    prompt: str,
    reduce_prompt: str,
    reducer_agent_index: int = 0,
) -> Response:
    """Two-phase broadcast: gather all responses, then reduce with one agent.

    Parameters
    ----------
    pool:
        The agent pool to broadcast to.
    prompt:
        The prompt sent to every agent.
    reduce_prompt:
        A template string containing ``{responses}`` which will be
        replaced with the concatenated agent outputs.
    reducer_agent_index:
        Index of the agent (within *pool*) used for the reduction step.
    """
    responses = await pool.broadcast_prompt(prompt)
    combined = "\n---\n".join(
        r.text for r in responses if r.success
    )
    filled = reduce_prompt.replace("{responses}", combined)
    return await pool.post(reducer_agent_index, filled)
