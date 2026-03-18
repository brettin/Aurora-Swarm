"""Unit tests for EmbeddingPool and scatter_gather_embeddings."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aurora_swarm.embedding_pool import EmbeddingPool, EmbeddingResponse
from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.patterns.embedding import scatter_gather_embeddings


# ---------------------------------------------------------------------------
# EmbeddingPool: mock OpenAI client
# ---------------------------------------------------------------------------

def _make_mock_client(agent_index: int):
    """One AsyncOpenAI-like client that returns embedding [agent_index, 0.0, 0.0]."""
    client = MagicMock()
    async def create(model=None, input=None):
        return MagicMock(
            data=[MagicMock(embedding=[float(agent_index), 0.0, 0.0])]
        )
    client.embeddings.create = AsyncMock(side_effect=create)
    return client


@pytest_asyncio.fixture
async def mock_embed_pool():
    """EmbeddingPool with mocked AsyncOpenAI; 4 endpoints, each returns distinct embedding."""
    endpoints = [
        AgentEndpoint(host="127.0.0.1", port=8000 + i)
        for i in range(4)
    ]
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.side_effect = [_make_mock_client(i) for i in range(4)]
        pool = EmbeddingPool(
            endpoints,
            model="test-model",
            concurrency=8,
            timeout=10.0,
        )
        yield pool
        await pool.close()


@pytest.mark.asyncio
async def test_embed_all_returns_in_order(mock_embed_pool):
    """embed_all returns one EmbeddingResponse per text in input order."""
    texts = ["a", "b", "c", "d", "e", "f"]
    results = await mock_embed_pool.embed_all(texts)

    assert len(results) == len(texts)
    for i, r in enumerate(results):
        assert r.success
        assert r.embedding is not None
        assert r.agent_index == (i % mock_embed_pool.size)
        # Round-robin: text i goes to agent i % size; our mock returns embedding [agent_index, 0, 0]
        assert r.embedding[0] == float(i % mock_embed_pool.size)


@pytest.mark.asyncio
async def test_embed_all_round_robin(mock_embed_pool):
    """Verify round-robin: first text to agent 0, second to agent 1, etc."""
    texts = ["one", "two", "three", "four", "five"]
    results = await mock_embed_pool.embed_all(texts)

    expected_agents = [0, 1, 2, 3, 0]
    for i, r in enumerate(results):
        assert r.agent_index == expected_agents[i]


@pytest.mark.asyncio
async def test_embed_pool_by_tag():
    """by_tag returns a sub-pool with only matching endpoints."""
    endpoints = [
        AgentEndpoint(host="h", port=1, tags={"role": "embed"}),
        AgentEndpoint(host="h", port=2, tags={"role": "llm"}),
        AgentEndpoint(host="h", port=3, tags={"role": "embed"}),
    ]
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.side_effect = [_make_mock_client(i) for i in range(3)]
        pool = EmbeddingPool(endpoints, model="m", concurrency=4)
        sub = pool.by_tag("role", "embed")
        assert sub.size == 2
        assert pool.size == 3
        await pool.close()


@pytest.mark.asyncio
async def test_embed_pool_select():
    """select(indices) returns a sub-pool with endpoints at those indices."""
    endpoints = [AgentEndpoint(host="h", port=8000 + i) for i in range(5)]
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.side_effect = [_make_mock_client(i) for i in range(5)]
        pool = EmbeddingPool(endpoints, model="m", concurrency=4)
        sub = pool.select([0, 2, 4])
        assert sub.size == 3
        await pool.close()


@pytest.mark.asyncio
async def test_embed_pool_slice():
    """slice(start, stop) returns a sub-pool of that range."""
    endpoints = [AgentEndpoint(host="h", port=8000 + i) for i in range(5)]
    with patch("openai.AsyncOpenAI") as MockOpenAI:
        MockOpenAI.side_effect = [_make_mock_client(i) for i in range(5)]
        pool = EmbeddingPool(endpoints, model="m", concurrency=4)
        sub = pool.slice(1, 4)
        assert sub.size == 3
        await pool.close()


@pytest.mark.asyncio
async def test_embed_one_invalid_agent_index(mock_embed_pool):
    """embed_one with invalid agent_index returns failure response."""
    r = await mock_embed_pool.embed_one(99, "x")
    assert not r.success
    assert r.embedding is None
    assert r.agent_index == 99
    assert "Invalid" in (r.error or "")


# ---------------------------------------------------------------------------
# scatter_gather_embeddings pattern
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scatter_gather_embeddings_delegates_to_embed_all():
    """scatter_gather_embeddings returns same order and count as embed_pool.embed_all."""
    mock_pool = MagicMock(spec=EmbeddingPool)
    fixed_results = [
        EmbeddingResponse(success=True, embedding=[1.0, 2.0], agent_index=0),
        EmbeddingResponse(success=True, embedding=[3.0, 4.0], agent_index=1),
        EmbeddingResponse(success=False, embedding=None, error="err", agent_index=0),
    ]
    mock_pool.embed_all = AsyncMock(return_value=fixed_results)

    texts = ["a", "b", "c"]
    results = await scatter_gather_embeddings(mock_pool, texts)

    assert results == fixed_results
    assert len(results) == 3
    mock_pool.embed_all.assert_called_once_with(texts)
