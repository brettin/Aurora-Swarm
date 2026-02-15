"""Tests for Pattern 4 — Blackboard (Shared-State Swarm)."""

import pytest

from aurora_swarm.patterns.blackboard import Blackboard


def _prompt_fn(role: str, board: dict[str, list[str]]) -> str:
    """Simple prompt function that shows the role and current board size."""
    total = sum(len(v) for v in board.values())
    return f"You are a {role}. Board has {total} entries. Contribute."


@pytest.mark.asyncio
async def test_blackboard(mock_pool_tagged):
    """2 proposers + 2 critics, 2 rounds — board should accumulate entries."""
    bb = Blackboard(
        sections=["hypotheses", "critiques"],
        prompt_fn=_prompt_fn,
    )
    final_board = await bb.run(mock_pool_tagged, max_rounds=2)

    # 2 proposers × 2 rounds = 4 hypothesis entries
    assert len(final_board["hypotheses"]) == 4
    # 2 critics × 2 rounds = 4 critique entries
    assert len(final_board["critiques"]) == 4

    # snapshot should match
    snap = bb.snapshot()
    assert snap["round"] == 2
    assert snap["board"] == final_board


@pytest.mark.asyncio
async def test_blackboard_convergence(mock_pool_tagged):
    """Convergence function can stop the session early."""

    def converge_after_one(board: dict[str, list[str]]) -> bool:
        return sum(len(v) for v in board.values()) >= 4  # stop after round 1

    bb = Blackboard(
        sections=["hypotheses", "critiques"],
        prompt_fn=_prompt_fn,
    )
    await bb.run(mock_pool_tagged, max_rounds=10, convergence_fn=converge_after_one)

    assert bb.round == 1
    assert len(bb.board["hypotheses"]) == 2
    assert len(bb.board["critiques"]) == 2
