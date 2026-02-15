"""VLLMPool — AgentPool subclass for vLLM OpenAI-compatible endpoints.

vLLM exposes an OpenAI-compatible chat completions API at
``/v1/chat/completions``.  This pool overrides :meth:`post` to speak
that protocol instead of the simpler ``/generate`` endpoint used by
the base :class:`AgentPool`.
"""

from __future__ import annotations

import aiohttp

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool, Response


class VLLMPool(AgentPool):
    """Agent pool that communicates via vLLM's OpenAI-compatible API.

    Parameters
    ----------
    endpoints:
        Agent endpoints (host + port where vLLM is listening).
    model:
        Model identifier passed in the ``"model"`` field of every
        request (e.g. ``"openai/gpt-oss-120b"``).
    max_tokens:
        Maximum tokens to generate per request.
    concurrency:
        Maximum number of in-flight requests.
    connector_limit:
        Maximum TCP connections in the aiohttp pool.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: list[AgentEndpoint],
        model: str = "openai/gpt-oss-120b",
        max_tokens: int = 512,
        concurrency: int = 512,
        connector_limit: int = 1024,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            endpoints,
            concurrency=concurrency,
            connector_limit=connector_limit,
            timeout=timeout,
        )
        self._model = model
        self._max_tokens = max_tokens

    # -- core request (OpenAI chat completions) ------------------------------

    async def post(self, agent_index: int, prompt: str) -> Response:
        """Send *prompt* via the OpenAI chat-completions API on the agent.

        The prompt is wrapped as a single ``user`` message.
        """
        ep = self._endpoints[agent_index]
        session = await self._get_session()
        async with self._semaphore:
            try:
                async with session.post(
                    f"{ep.url}/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self._max_tokens,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    data = await resp.json()
                    message = data["choices"][0]["message"]
                    text = message.get("content") or message.get("reasoning_content") or ""
                    return Response(
                        success=True,
                        text=text,
                        agent_index=agent_index,
                    )
            except Exception as exc:
                return Response(
                    success=False,
                    text="",
                    error=str(exc),
                    agent_index=agent_index,
                )

    # -- sub-pool override ---------------------------------------------------

    def _sub_pool(self, endpoints: list[AgentEndpoint]) -> "VLLMPool":
        """Create a child VLLMPool sharing concurrency settings."""
        child = VLLMPool.__new__(VLLMPool)
        child._endpoints = endpoints
        child._concurrency = self._concurrency
        child._connector_limit = self._connector_limit
        child._timeout = self._timeout
        child._semaphore = self._semaphore
        child._session = self._session
        child._model = self._model
        child._max_tokens = self._max_tokens
        return child
