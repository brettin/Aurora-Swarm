"""Shared fixtures — mock agent HTTP servers and pool factory."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp import web

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool


# ---------------------------------------------------------------------------
# Mock agent server
# ---------------------------------------------------------------------------

async def _handle_generate(request: web.Request) -> web.Response:
    """Echo the prompt back as the response — simplest possible agent."""
    data = await request.json()
    prompt = data.get("prompt", "")
    return web.json_response({"response": f"echo: {prompt}"})


async def _start_mock_server(host: str, port: int) -> tuple[web.AppRunner, int]:
    """Start a single mock agent server and return (runner, actual_port)."""
    app = web.Application()
    app.router.add_post("/generate", _handle_generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Resolve actual port (useful if port=0 for OS-assigned)
    actual_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, actual_port


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def mock_pool() -> AsyncIterator[AgentPool]:
    """Yield an AgentPool backed by 4 local echo-servers."""
    n_agents = 4
    base_port = 0  # let the OS pick free ports
    runners: list[web.AppRunner] = []
    endpoints: list[AgentEndpoint] = []

    for _ in range(n_agents):
        runner, port = await _start_mock_server("127.0.0.1", base_port)
        runners.append(runner)
        endpoints.append(AgentEndpoint(host="127.0.0.1", port=port))

    pool = AgentPool(endpoints, concurrency=16, connector_limit=32)
    try:
        yield pool
    finally:
        await pool.close()
        for runner in runners:
            await runner.cleanup()


@pytest_asyncio.fixture
async def mock_pool_8() -> AsyncIterator[AgentPool]:
    """Yield an AgentPool backed by 8 local echo-servers."""
    n_agents = 8
    runners: list[web.AppRunner] = []
    endpoints: list[AgentEndpoint] = []

    for _ in range(n_agents):
        runner, port = await _start_mock_server("127.0.0.1", 0)
        runners.append(runner)
        endpoints.append(AgentEndpoint(host="127.0.0.1", port=port))

    pool = AgentPool(endpoints, concurrency=16, connector_limit=32)
    try:
        yield pool
    finally:
        await pool.close()
        for runner in runners:
            await runner.cleanup()


@pytest_asyncio.fixture
async def mock_pool_tagged() -> AsyncIterator[AgentPool]:
    """Yield an AgentPool with 4 agents: 2 tagged role=hypotheses, 2 role=critiques."""
    runners: list[web.AppRunner] = []
    endpoints: list[AgentEndpoint] = []

    for i in range(4):
        runner, port = await _start_mock_server("127.0.0.1", 0)
        runners.append(runner)
        role = "hypotheses" if i < 2 else "critiques"
        endpoints.append(AgentEndpoint(host="127.0.0.1", port=port, tags={"role": role}))

    pool = AgentPool(endpoints, concurrency=16, connector_limit=32)
    try:
        yield pool
    finally:
        await pool.close()
        for runner in runners:
            await runner.cleanup()
