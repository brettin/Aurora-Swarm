"""Pattern 4 — Blackboard (Shared-State Swarm).

Agents collaborate through a shared mutable workspace divided into named
sections.  The session runs in rounds.  Each round every agent reads the
current board, a role-specific prompt function generates a customised
prompt, and all agents respond in parallel.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from aurora_swarm.pool import AgentPool


# Type aliases
BoardState = dict[str, list[str]]
PromptFn = Callable[[str, BoardState], str]
ConvergenceFn = Callable[[BoardState], bool]


class Blackboard:
    """Shared-state workspace for multi-round agent collaboration.

    Parameters
    ----------
    sections:
        Names of the board sections (e.g. ``["hypotheses", "critiques"]``).
    prompt_fn:
        ``prompt_fn(role, board_state) -> str`` — generates the prompt
        that an agent with the given *role* should receive, given the
        current board contents.
    """

    def __init__(
        self,
        sections: list[str],
        prompt_fn: PromptFn,
    ) -> None:
        self._board: BoardState = {section: [] for section in sections}
        self._prompt_fn = prompt_fn
        self._round = 0

    # -- public API ----------------------------------------------------------

    @property
    def board(self) -> BoardState:
        """Current board state (mutable reference)."""
        return self._board

    @property
    def round(self) -> int:
        """Number of completed rounds."""
        return self._round

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable deep copy of the board state."""
        return {
            "round": self._round,
            "board": copy.deepcopy(self._board),
        }

    async def run(
        self,
        pool: AgentPool,
        max_rounds: int = 10,
        convergence_fn: ConvergenceFn | None = None,
    ) -> BoardState:
        """Execute rounds until *max_rounds* or convergence.

        Agent roles are determined by the ``role`` tag on each endpoint.
        Agents whose ``role`` matches a board section contribute to that
        section.  Agents with no ``role`` tag or a role not in the board
        sections are skipped.

        Parameters
        ----------
        pool:
            Agent pool with role-tagged endpoints.
        max_rounds:
            Upper bound on the number of rounds.
        convergence_fn:
            Optional ``convergence_fn(board_state) -> bool``.  If it
            returns ``True`` after a round the session stops early.

        Returns
        -------
        BoardState
            The final board.
        """
        sections = list(self._board.keys())

        for _ in range(max_rounds):
            for section in sections:
                sub = pool.by_tag("role", section)
                if sub.size == 0:
                    continue

                prompt = self._prompt_fn(section, self._board)
                responses = await sub.broadcast_prompt(prompt)

                for r in responses:
                    if r.success:
                        self._board[section].append(r.text)

            self._round += 1

            if convergence_fn is not None and convergence_fn(self._board):
                break

        return self._board
