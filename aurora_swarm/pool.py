"""AgentPool — async connection pool for Aurora agent endpoints.

Provides semaphore-throttled, pooled HTTP access to 1000–4000 LLM agent
instances.  Every public pattern function in this package takes an
``AgentPool`` as its first argument.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Sequence

import aiohttp

from aurora_swarm.hostfile import AgentEndpoint


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class Response:
    """Result of a single agent call."""

    success: bool
    text: str
    error: str | None = None
    agent_index: int = -1


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------


class AgentPool:
    """Async pool of agent HTTP endpoints with concurrency control.

    Parameters
    ----------
    endpoints:
        Agent endpoints — either :class:`AgentEndpoint` objects or plain
        ``(host, port)`` tuples (tags will be empty).
    concurrency:
        Maximum number of in-flight requests (asyncio semaphore size).
    connector_limit:
        Maximum number of TCP connections in the aiohttp pool.
    timeout:
        Per-request timeout in seconds.
    proxy_url:
        Optional URL of the reverse proxy. When set, requests are routed
        through the proxy using path-prefix routing instead of connecting
        directly to agent endpoints. Can also be set via the
        ``AURORA_SWARM_PROXY_URL`` environment variable (explicit parameter
        takes priority).
    """

    def __init__(
        self,
        endpoints: Sequence[AgentEndpoint | tuple[str, int]],
        concurrency: int = 512,
        connector_limit: int = 1024,
        timeout: float = 120.0,
        proxy_url: str | None = None,
    ) -> None:
        self._endpoints: list[AgentEndpoint] = []
        for ep in endpoints:
            if isinstance(ep, AgentEndpoint):
                self._endpoints.append(ep)
            else:
                host, port = ep
                self._endpoints.append(AgentEndpoint(host=host, port=port))

        self._concurrency = concurrency
        self._connector_limit = connector_limit
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._session: aiohttp.ClientSession | None = None

        # Proxy configuration: explicit parameter > environment variable
        self._proxy_url: str | None = proxy_url or os.environ.get(
            "AURORA_SWARM_PROXY_URL"
        )

        # Global indices track each endpoint's position in the top-level pool,
        # which is needed for proxy path-prefix routing (e.g. /agent/{index}).
        self._global_indices: list[int] = list(range(len(self._endpoints)))

    # -- lifecycle -----------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=self._connector_limit)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AgentPool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- properties ----------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of agents in the pool."""
        return len(self._endpoints)

    @property
    def endpoints(self) -> list[AgentEndpoint]:
        return list(self._endpoints)

    @property
    def proxy_url(self) -> str | None:
        """The proxy URL if configured, otherwise None."""
        return self._proxy_url

    # -- proxy-aware URL helper ----------------------------------------------

    def _agent_base_url(self, agent_index: int) -> str:
        """Get the base URL for the given agent, routing through proxy if configured.

        Args:
            agent_index: The local index within this pool (0 to size-1).

        Returns:
            Base URL string. When proxy is configured, returns
            ``{proxy_url}/agent/{global_index}``. Otherwise returns
            the endpoint's direct URL.
        """
        if self._proxy_url is None:
            return self._endpoints[agent_index].url
        return (
            f"{self._proxy_url.rstrip('/')}/agent/{self._global_indices[agent_index]}"
        )

    # -- selectors -----------------------------------------------------------

    def by_tag(self, key: str, value: str) -> "AgentPool":
        """Return a sub-pool of agents whose tag *key* equals *value*."""
        filtered: list[AgentEndpoint] = []
        indices: list[int] = []
        for i, ep in enumerate(self._endpoints):
            if ep.tags.get(key) == value:
                filtered.append(ep)
                indices.append(self._global_indices[i])
        return self._sub_pool(filtered, global_indices=indices)

    def sample(self, n: int) -> "AgentPool":
        """Return a sub-pool of *n* randomly chosen agents."""
        pool_size = len(self._endpoints)
        chosen_local = random.sample(range(pool_size), min(n, pool_size))
        endpoints = [self._endpoints[i] for i in chosen_local]
        indices = [self._global_indices[i] for i in chosen_local]
        return self._sub_pool(endpoints, global_indices=indices)

    def select(self, indices: Sequence[int]) -> "AgentPool":
        """Return a sub-pool with agents at the given indices."""
        selected = [self._endpoints[i] for i in indices]
        global_idx = [self._global_indices[i] for i in indices]
        return self._sub_pool(selected, global_indices=global_idx)

    def slice(self, start: int, stop: int) -> "AgentPool":
        """Return a sub-pool from index *start* to *stop*."""
        return self._sub_pool(
            self._endpoints[start:stop],
            global_indices=self._global_indices[start:stop],
        )

    def _sub_pool(
        self,
        endpoints: list[AgentEndpoint],
        global_indices: list[int] | None = None,
    ) -> "AgentPool":
        """Create a child pool sharing concurrency settings.

        Args:
            endpoints: The endpoints for the child pool.
            global_indices: Global index mapping for proxy routing. If None,
                defaults to ``list(range(len(endpoints)))``.
        """
        child = AgentPool.__new__(AgentPool)
        child._endpoints = endpoints
        child._concurrency = self._concurrency
        child._connector_limit = self._connector_limit
        child._timeout = self._timeout
        child._semaphore = self._semaphore  # share parent semaphore
        child._session = self._session  # share parent session
        child._proxy_url = self._proxy_url
        child._global_indices = (
            global_indices
            if global_indices is not None
            else list(range(len(endpoints)))
        )
        return child

    # -- core request --------------------------------------------------------

    async def post(
        self, agent_index: int, prompt: str, max_tokens: int | None = None
    ) -> Response:
        """Send *prompt* to the agent at *agent_index* and return its response.

        The call is throttled by the pool-wide semaphore so that at most
        ``concurrency`` requests are in flight at once.

        Parameters
        ----------
        agent_index:
            Index of the agent to send the prompt to.
        prompt:
            The prompt text.
        max_tokens:
            Optional maximum tokens to generate. Ignored by base AgentPool
            (only used by VLLMPool and subclasses).
        """
        session = await self._get_session()
        url = f"{self._agent_base_url(agent_index)}/generate"
        async with self._semaphore:
            try:
                async with session.post(
                    url,
                    json={"prompt": prompt},
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    data = await resp.json()
                    return Response(
                        success=True,
                        text=data.get("response", data.get("text", "")),
                        agent_index=agent_index,
                    )
            except Exception as exc:
                return Response(
                    success=False,
                    text="",
                    error=str(exc),
                    agent_index=agent_index,
                )

    async def send_all(self, prompts: list[str]) -> list[Response]:
        """Send ``prompts[i]`` to ``agent[i % size]`` concurrently.

        Returns responses in *input* order (i.e. ``results[i]``
        corresponds to ``prompts[i]``).
        """
        tasks = [self.post(i % self.size, prompt) for i, prompt in enumerate(prompts)]
        return list(await asyncio.gather(*tasks))

    async def broadcast_prompt(self, prompt: str) -> list[Response]:
        """Send the same *prompt* to every agent in the pool."""
        tasks = [self.post(i, prompt) for i in range(self.size)]
        return list(await asyncio.gather(*tasks))
