"""Integration tests for Pattern 4 — Blackboard against live vLLM endpoints.

The ``vllm_pool_tagged`` fixture splits endpoints so the first half are
tagged ``role=hypotheses`` and the second half ``role=critiques``.
"""

import pytest

from aurora_swarm.patterns.blackboard import Blackboard


def _prompt_fn(role: str, board: dict[str, list[str]]) -> str:
    """Generate a role-aware prompt based on the current board state."""
    if role == "hypotheses":
        existing = "\n".join(board.get("critiques", [])) or "None yet."
        return (
            "You are a scientist proposing hypotheses. "
            "Here are critiques from the previous round:\n"
            f"{existing}\n\n"
            "Propose a single new hypothesis about antibiotic resistance "
            "in one sentence."
        )
    else:  # critiques
        existing = "\n".join(board.get("hypotheses", [])) or "None yet."
        return (
            "You are a critical reviewer. "
            "Here are hypotheses from the previous round:\n"
            f"{existing}\n\n"
            "Provide a brief one-sentence critique of the most recent hypothesis."
        )


@pytest.mark.asyncio
async def test_blackboard(vllm_pool_tagged):
    """Run 2 rounds; board should accumulate entries from both roles."""
    bb = Blackboard(
        sections=["hypotheses", "critiques"],
        prompt_fn=_prompt_fn,
    )
    final_board = await bb.run(vllm_pool_tagged, max_rounds=2)

    # Each role group runs once per round, so entries should accumulate.
    assert len(final_board["hypotheses"]) > 0, "No hypotheses were produced"
    assert len(final_board["critiques"]) > 0, "No critiques were produced"

    snap = bb.snapshot()
    assert snap["round"] == 2
    assert snap["board"] == final_board


@pytest.mark.asyncio
async def test_blackboard_convergence(vllm_pool_tagged):
    """Convergence function can stop the session early."""
    total_agents = vllm_pool_tagged.size

    def converge_after_one(board: dict[str, list[str]]) -> bool:
        total_entries = sum(len(v) for v in board.values())
        return total_entries >= total_agents

    bb = Blackboard(
        sections=["hypotheses", "critiques"],
        prompt_fn=_prompt_fn,
    )
    await bb.run(vllm_pool_tagged, max_rounds=10, convergence_fn=converge_after_one)

    assert bb.round == 1, f"Expected 1 round but ran {bb.round}"
    assert len(bb.board["hypotheses"]) > 0
    assert len(bb.board["critiques"]) > 0
