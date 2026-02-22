"""Shared fixtures — mock agent HTTP servers and pool factory."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp import web

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool
from aurora_swarm.vllm_pool import VLLMPool


def pytest_addoption(parser):
    """Register the ``--hostfile`` option for integration tests (used by tests/integration/conftest.py)."""
    parser.addoption(
        "--hostfile",
        action="store",
        default=None,
        help="Path to the tab-separated vLLM hostfile for integration tests (hostname<TAB>port).",
    )


# ---------------------------------------------------------------------------
# Mock agent server
# ---------------------------------------------------------------------------

async def _handle_generate(request: web.Request) -> web.Response:
    """Echo the prompt back as the response — simplest possible agent."""
    data = await request.json()
    prompt = data.get("prompt", "")
    return web.json_response({"response": f"echo: {prompt}"})


async def _handle_completions(request: web.Request) -> web.Response:
    """Mock vLLM completions endpoint — supports batch prompts."""
    data = await request.json()
    
    # Handle chat completions format (messages) and completions format (prompt)
    if "messages" in data:
        # Chat completions format
        messages = data["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            prompts = [messages[0].get("content", "")]
        else:
            prompts = [""]
        
        # Return chat completion format
        choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": f"completion: {p}",
                },
                "index": 0,
                "finish_reason": "stop",
            }
            for p in prompts
        ]
        
        return web.json_response({
            "id": "mock-chat-completion-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": choices,
        })
    else:
        # Completions format
        prompt = data.get("prompt")
        
        # Handle both single prompt (string) and batch prompts (list)
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, list):
            prompts = prompt
        else:
            return web.json_response({"error": "Invalid prompt type"}, status=400)
        
        # Create choices - one per prompt
        choices = [
            {
                "text": f"completion: {p}",
                "index": i,
                "finish_reason": "stop",
            }
            for i, p in enumerate(prompts)
        ]
        
        return web.json_response({
            "id": "mock-completion-id",
            "object": "text_completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": choices,
        })


async def _handle_models(request: web.Request) -> web.Response:
    """Mock vLLM /v1/models endpoint."""
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": 1234567890,
                "owned_by": "test",
                "max_model_len": 8192,
            }
        ],
    })


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


async def _start_mock_vllm_server(host: str, port: int) -> tuple[web.AppRunner, int]:
    """Start a mock vLLM server with OpenAI-compatible endpoints."""
    app = web.Application()
    app.router.add_post("/v1/completions", _handle_completions)
    app.router.add_post("/v1/chat/completions", _handle_completions)  # Reuse handler
    app.router.add_get("/v1/models", _handle_models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
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


@pytest_asyncio.fixture
async def mock_vllm_pool() -> AsyncIterator[VLLMPool]:
    """Yield a VLLMPool backed by 4 local mock vLLM servers."""
    n_agents = 4
    runners: list[web.AppRunner] = []
    endpoints: list[AgentEndpoint] = []

    for _ in range(n_agents):
        runner, port = await _start_mock_vllm_server("127.0.0.1", 0)
        runners.append(runner)
        endpoints.append(AgentEndpoint(host="127.0.0.1", port=port))

    pool = VLLMPool(
        endpoints,
        model="mock-model",
        max_tokens=100,
        concurrency=16,
        connector_limit=32,
        use_batch=True,
    )
    try:
        yield pool
    finally:
        await pool.close()
        for runner in runners:
            await runner.cleanup()
