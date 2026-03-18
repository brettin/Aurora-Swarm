"""EmbeddingPool — async pool for OpenAI-compatible /v1/embeddings endpoints.

Provides scatter-gather over embedding servers with the same hostfile/selector
API as AgentPool, so parse_hostfile + by_tag work the same way.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from aurora_swarm.hostfile import AgentEndpoint

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# EmbeddingResponse
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResponse:
    """Result of a single embedding request."""

    success: bool
    embedding: list[float] | None
    error: str | None = None
    agent_index: int = -1


# ---------------------------------------------------------------------------
# EmbeddingPool
# ---------------------------------------------------------------------------

def _resolve_endpoints(
    endpoints: Sequence[AgentEndpoint | tuple[str, int]],
) -> list[AgentEndpoint]:
    """Normalize to list of AgentEndpoint (same as AgentPool)."""
    out: list[AgentEndpoint] = []
    for ep in endpoints:
        if isinstance(ep, AgentEndpoint):
            out.append(ep)
        else:
            host, port = ep
            out.append(AgentEndpoint(host=host, port=port))
    return out


class EmbeddingPool:
    """Async pool of embedding endpoints (OpenAI-compatible /v1/embeddings).

    Parameters
    ----------
    endpoints:
        Embedding endpoints — either :class:`AgentEndpoint` objects or
        ``(host, port)`` tuples (tags will be empty).
    model:
        Embedding model id (e.g. sentence-transformers/all-MiniLM-L6-v2).
    concurrency:
        Maximum number of in-flight requests (asyncio semaphore size).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: Sequence[AgentEndpoint | tuple[str, int]],
        model: str,
        concurrency: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self._endpoints = _resolve_endpoints(endpoints)
        self._model = model
        self._concurrency = concurrency
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._clients: list[AsyncOpenAI] | None = None

    async def _get_clients(self) -> list[AsyncOpenAI]:
        """Create and cache one AsyncOpenAI per endpoint."""
        if self._clients is not None:
            return self._clients
        from openai import AsyncOpenAI
        self._clients = [
            AsyncOpenAI(
                base_url=f"{ep.url}/v1",
                api_key="EMPTY",
                timeout=self._timeout,
            )
            for ep in self._endpoints
        ]
        return self._clients

    @property
    def size(self) -> int:
        """Number of endpoints in the pool."""
        return len(self._endpoints)

    @property
    def timeout(self) -> float:
        """Per-request timeout in seconds."""
        return self._timeout

    @property
    def endpoints(self) -> list[AgentEndpoint]:
        return list(self._endpoints)

    def by_tag(self, key: str, value: str) -> EmbeddingPool:
        """Return a sub-pool of endpoints whose tag *key* equals *value*."""
        filtered = [ep for ep in self._endpoints if ep.tags.get(key) == value]
        return self._sub_pool(filtered)

    def sample(self, n: int) -> EmbeddingPool:
        """Return a sub-pool of *n* randomly chosen endpoints."""
        chosen = random.sample(self._endpoints, min(n, len(self._endpoints)))
        return self._sub_pool(chosen)

    def select(self, indices: Sequence[int]) -> EmbeddingPool:
        """Return a sub-pool with endpoints at the given indices."""
        selected = [self._endpoints[i] for i in indices]
        return self._sub_pool(selected)

    def slice(self, start: int, stop: int) -> EmbeddingPool:
        """Return a sub-pool from index *start* to *stop*."""
        return self._sub_pool(self._endpoints[start:stop])

    def _sub_pool(self, endpoints: list[AgentEndpoint]) -> EmbeddingPool:
        """Create a child pool with filtered endpoints (own clients when used)."""
        child = EmbeddingPool.__new__(EmbeddingPool)
        child._endpoints = endpoints
        child._model = self._model
        child._concurrency = self._concurrency
        child._timeout = self._timeout
        child._semaphore = self._semaphore
        child._clients = None
        return child

    async def embed_one(self, agent_index: int, text: str) -> EmbeddingResponse:
        """Request embedding for *text* from the endpoint at *agent_index*."""
        if not self._endpoints or agent_index < 0 or agent_index >= len(self._endpoints):
            return EmbeddingResponse(
                success=False,
                embedding=None,
                error="Invalid agent_index",
                agent_index=agent_index,
            )
        clients = await self._get_clients()
        async with self._semaphore:
            try:
                r = await clients[agent_index].embeddings.create(
                    model=self._model,
                    input=[text],
                )
                if r.data and len(r.data) > 0:
                    return EmbeddingResponse(
                        success=True,
                        embedding=list(r.data[0].embedding),
                        agent_index=agent_index,
                    )
                return EmbeddingResponse(
                    success=False,
                    embedding=None,
                    error="Empty response data",
                    agent_index=agent_index,
                )
            except Exception as exc:
                return EmbeddingResponse(
                    success=False,
                    embedding=None,
                    error=str(exc),
                    agent_index=agent_index,
                )

    async def embed_all(self, texts: list[str]) -> list[EmbeddingResponse]:
        """Scatter *texts* across endpoints round-robin; return responses in input order."""
        n = self.size
        if n == 0:
            return [
                EmbeddingResponse(success=False, embedding=None, error="Empty pool", agent_index=-1)
                for _ in texts
            ]
        tasks = [
            self.embed_one(i % n, text)
            for i, text in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def close(self) -> None:
        """Release resources. AsyncOpenAI clients do not require explicit close."""
        self._clients = None

    async def __aenter__(self) -> EmbeddingPool:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
