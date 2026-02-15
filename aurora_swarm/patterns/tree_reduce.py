"""Pattern 3 — Hierarchical Tree-Reduce.

Leaf agents produce initial responses.  Groups of responses are fed to
supervisor agents that summarize them, recursively, until a single
answer remains.
"""

from __future__ import annotations

import math
from typing import Any

from aurora_swarm.pool import AgentPool, Response


async def tree_reduce(
    pool: AgentPool,
    prompt: str,
    reduce_prompt: str,
    fanin: int = 50,
    items: list[Any] | None = None,
) -> Response:
    """Run a hierarchical tree-reduce over *pool*.

    Parameters
    ----------
    pool:
        The agent pool (used for both leaf work and supervisors).
    prompt:
        Leaf-level task.  If *items* is provided the template should
        contain an ``{item}`` placeholder.
    reduce_prompt:
        Supervisor summarisation task.  Must contain ``{responses}`` and
        may contain ``{level}``.
    fanin:
        Number of responses each supervisor handles per group.
    items:
        If given, scatter items across leaf agents (one per agent,
        round-robin).  Otherwise the same *prompt* is broadcast.
    """
    # -- leaf phase ----------------------------------------------------------
    if items is not None:
        leaf_prompts = [prompt.replace("{item}", str(it)) for it in items]
        leaf_responses = await pool.send_all(leaf_prompts)
    else:
        leaf_responses = await pool.broadcast_prompt(prompt)

    # -- reduction phase -----------------------------------------------------
    current: list[str] = [r.text for r in leaf_responses if r.success]
    level = 1

    while len(current) > 1:
        # chunk into groups of *fanin*
        groups: list[list[str]] = []
        for i in range(0, len(current), fanin):
            groups.append(current[i : i + fanin])

        # each group gets sent to one supervisor agent
        supervisor_prompts: list[str] = []
        for group in groups:
            combined = "\n---\n".join(group)
            filled = reduce_prompt.replace("{responses}", combined)
            filled = filled.replace("{level}", str(level))
            supervisor_prompts.append(filled)

        # scatter supervisor work across available agents
        sup_responses = await pool.send_all(supervisor_prompts)
        current = [r.text for r in sup_responses if r.success]
        level += 1

    if not current:
        return Response(success=False, text="", error="All agents failed during reduction")
    return Response(success=True, text=current[0])
