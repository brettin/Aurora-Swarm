"""Tests for the Aurora-Swarm HTTP reverse proxy (aurora_swarm.proxy)."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from aurora_swarm.hostfile import AgentEndpoint


# ---------------------------------------------------------------------------
# Helper: build a proxy app wired to the given endpoints
# ---------------------------------------------------------------------------


def _create_proxy_app(
    endpoints: list[AgentEndpoint],
    timeout: float = 30.0,
    connector_limit: int = 64,
) -> web.Application:
    """Create a proxy ``web.Application`` without calling ``run_proxy``.

    This mirrors the setup logic in :func:`aurora_swarm.proxy.run_proxy` but
    avoids blocking on ``web.run_app``.
    """
    from aurora_swarm.proxy import (
        _health_handler,
        _on_cleanup,
        _on_startup,
        _proxy_handler,
        _status_handler,
    )

    app = web.Application()
    app["endpoints"] = endpoints
    app["connector_limit"] = connector_limit
    app["default_timeout"] = timeout
    app["listen_host"] = "127.0.0.1"
    app["listen_port"] = 0

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    app.router.add_get("/health", _health_handler)
    app.router.add_get("/status", _status_handler)
    app.router.add_route("*", "/agent/{index}/{path:.*}", _proxy_handler)

    return app


# ---------------------------------------------------------------------------
# Mock downstream agent
# ---------------------------------------------------------------------------


async def _mock_generate(request: web.Request) -> web.Response:
    """Echo the prompt back as the response."""
    data = await request.json()
    prompt = data.get("prompt", "")
    return web.json_response({"response": f"echo: {prompt}"})


async def _mock_models(request: web.Request) -> web.Response:
    """Return a fake /v1/models payload."""
    qs = request.query_string
    return web.json_response(
        {
            "data": [{"id": "test-model", "max_model_len": 4096}],
            "query_string": qs,
        }
    )


async def _mock_headers_echo(request: web.Request) -> web.Response:
    """Echo back all received headers as JSON."""
    headers = dict(request.headers)
    return web.json_response({"headers": headers})


async def _mock_error(request: web.Request) -> web.Response:
    """Return a 503 error."""
    return web.json_response({"error": "service unavailable"}, status=503)


def _build_mock_agent_app() -> web.Application:
    """Build a mock downstream agent application."""
    app = web.Application()
    app.router.add_post("/generate", _mock_generate)
    app.router.add_get("/v1/models", _mock_models)
    app.router.add_get("/headers", _mock_headers_echo)
    app.router.add_post("/error", _mock_error)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def downstream_server():
    """Start a mock downstream agent server and yield its (host, port)."""
    app = _build_mock_agent_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield ("127.0.0.1", port)
    await runner.cleanup()


@pytest.fixture
async def proxy_client(downstream_server):
    """Create a test client for the proxy app backed by one downstream agent."""
    host, port = downstream_server
    endpoints = [AgentEndpoint(host=host, port=port)]
    app = _create_proxy_app(endpoints)
    async with TestClient(TestServer(app)) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(proxy_client):
    """GET /health returns 200 and {"status": "ok"}."""
    resp = await proxy_client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_status_endpoint(proxy_client):
    """GET /status returns 200 with agents count and uptime_seconds."""
    resp = await proxy_client.get("/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["agents"] == 1
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], float)
    assert "endpoints" in data
    assert len(data["endpoints"]) == 1


@pytest.mark.asyncio
async def test_proxy_forward_post(proxy_client):
    """POST /agent/0/generate correctly forwards to downstream agent."""
    resp = await proxy_client.post(
        "/agent/0/generate",
        json={"prompt": "hello world"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["response"] == "echo: hello world"


@pytest.mark.asyncio
async def test_proxy_forward_get(proxy_client):
    """GET /agent/0/v1/models correctly forwards to downstream agent."""
    resp = await proxy_client.get("/agent/0/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert "data" in data
    assert data["data"][0]["id"] == "test-model"


@pytest.mark.asyncio
async def test_proxy_forward_query_string(proxy_client):
    """Request with query string is correctly forwarded."""
    resp = await proxy_client.get("/agent/0/v1/models?format=json&limit=10")
    assert resp.status == 200
    data = await resp.json()
    assert data["query_string"] == "format=json&limit=10"


@pytest.mark.asyncio
async def test_proxy_forward_headers(proxy_client):
    """Request headers are correctly forwarded (excluding hop-by-hop headers)."""
    resp = await proxy_client.get(
        "/agent/0/headers",
        headers={
            "X-Custom-Header": "test-value",
            "Authorization": "Bearer token123",
        },
    )
    assert resp.status == 200
    data = await resp.json()
    received_headers = data["headers"]
    # Custom headers should be forwarded
    assert received_headers.get("X-Custom-Header") == "test-value"
    assert received_headers.get("Authorization") == "Bearer token123"


@pytest.mark.asyncio
async def test_proxy_invalid_index(proxy_client):
    """Index out of range returns 400."""
    resp = await proxy_client.get("/agent/99/v1/models")
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "out of range" in data["error"]


@pytest.mark.asyncio
async def test_proxy_negative_index(proxy_client):
    """Negative index returns 400."""
    resp = await proxy_client.get("/agent/-1/v1/models")
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_proxy_downstream_error(proxy_client):
    """Downstream error status code is transparently forwarded."""
    resp = await proxy_client.post("/agent/0/error", json={})
    assert resp.status == 503
    body = await resp.read()
    data = json.loads(body)
    assert data["error"] == "service unavailable"
