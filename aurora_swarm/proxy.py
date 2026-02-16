"""HTTP reverse proxy for Aurora-Swarm agent endpoints.

Deploys on an HPC login node to multiplex all agent endpoints behind a
single TCP port, allowing users to reach compute-node agents via one
SSH tunnel.

Routes::

    GET  /health              → health check
    GET  /status              → agent list & uptime
    *    /agent/{index}/{path} → forward to endpoints[index]
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from aurora_swarm.hostfile import parse_hostfile

if TYPE_CHECKING:
    from aurora_swarm.hostfile import AgentEndpoint

logger = logging.getLogger("aurora_swarm.proxy")

# Hop-by-hop headers that MUST NOT be forwarded between client and
# downstream (RFC 2616 §13.5.1 + common proxy practice).
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Additional response headers that the proxy must strip because aiohttp
# handles its own transfer coding for the streamed response.
_STRIP_RESPONSE_HEADERS = frozenset(
    {
        "content-encoding",
        "transfer-encoding",
        "content-length",
    }
)


# ------------------------------------------------------------------
# Route handlers
# ------------------------------------------------------------------


async def _health_handler(request: web.Request) -> web.Response:
    """Return a simple health-check payload."""
    return web.json_response({"status": "ok"})


async def _status_handler(request: web.Request) -> web.Response:
    """Return proxy status including agent list and uptime."""
    endpoints: list[AgentEndpoint] = request.app["endpoints"]
    start_time: float = request.app["start_time"]
    uptime = time.monotonic() - start_time

    agent_list = [
        {
            "index": i,
            "host": ep.host,
            "port": ep.port,
            "tags": dict(ep.tags),
        }
        for i, ep in enumerate(endpoints)
    ]

    return web.json_response(
        {
            "agents": len(endpoints),
            "uptime_seconds": round(uptime, 2),
            "endpoints": agent_list,
        }
    )


async def _proxy_handler(request: web.Request) -> web.StreamResponse:
    """Forward an incoming request to the appropriate downstream agent.

    URL pattern: ``/agent/{index}/{path:.*}``

    The handler:
    1. Validates the agent *index*.
    2. Constructs the downstream URL (including query string).
    3. Forwards method, headers (minus hop-by-hop), and body.
    4. Streams the downstream response back to the client.
    """
    endpoints: list[AgentEndpoint] = request.app["endpoints"]
    session: ClientSession = request.app["client_session"]
    default_timeout: float = request.app["default_timeout"]

    # --- extract & validate index ---
    try:
        index = int(request.match_info["index"])
    except (KeyError, ValueError):
        return web.json_response({"error": "invalid agent index"}, status=400)

    if index < 0 or index >= len(endpoints):
        return web.json_response(
            {"error": (f"agent index {index} out of range [0, {len(endpoints)})")},
            status=400,
        )

    ep = endpoints[index]
    path = request.match_info.get("path", "")

    # --- build downstream URL ---
    downstream_url = f"http://{ep.host}:{ep.port}/{path}"
    if request.query_string:
        downstream_url += f"?{request.query_string}"

    # --- determine timeout ---
    timeout_value = default_timeout
    x_timeout = request.headers.get("X-Timeout")
    if x_timeout is not None:
        try:
            timeout_value = float(x_timeout)
        except ValueError:
            pass  # fall back to default

    timeout = ClientTimeout(total=timeout_value)

    # --- prepare forwarded headers ---
    forward_headers: dict[str, str] = {}
    for name, value in request.headers.items():
        lower = name.lower()
        if lower in _HOP_BY_HOP_HEADERS:
            continue
        # X-Timeout is consumed by the proxy; do not forward.
        if lower == "x-timeout":
            continue
        forward_headers[name] = value

    # --- read request body ---
    body = await request.read()

    # --- forward request ---
    try:
        downstream_resp = await session.request(
            method=request.method,
            url=downstream_url,
            headers=forward_headers,
            data=body,
            timeout=timeout,
            allow_redirects=False,
        )
    except TimeoutError:
        logger.error(
            "Timeout after %.0fs forwarding %s /agent/%d/%s to %s",
            timeout_value,
            request.method,
            index,
            path,
            downstream_url,
        )
        return web.json_response(
            {"error": f"upstream timeout after {timeout_value}s"},
            status=504,
        )
    except OSError as exc:
        logger.error(
            "Connection error forwarding %s /agent/%d/%s to %s: %s",
            request.method,
            index,
            path,
            downstream_url,
            exc,
        )
        return web.json_response(
            {"error": f"cannot connect to {ep.host}:{ep.port}"},
            status=502,
        )
    except Exception:
        logger.exception(
            "Unexpected error forwarding %s /agent/%d/%s to %s",
            request.method,
            index,
            path,
            downstream_url,
        )
        return web.json_response({"error": "internal proxy error"}, status=500)

    # --- stream response back ---
    try:
        response = web.StreamResponse(status=downstream_resp.status)

        # Copy response headers, excluding those the proxy must strip.
        skip = _HOP_BY_HOP_HEADERS | _STRIP_RESPONSE_HEADERS
        for name, value in downstream_resp.headers.items():
            if name.lower() not in skip:
                response.headers[name] = value

        await response.prepare(request)

        async for chunk in downstream_resp.content.iter_any():
            await response.write(chunk)

        await response.write_eof()

        logger.debug(
            "%s /agent/%d/%s -> %s %d",
            request.method,
            index,
            path,
            downstream_url,
            downstream_resp.status,
        )

        return response
    finally:
        downstream_resp.release()


# ------------------------------------------------------------------
# Application lifecycle
# ------------------------------------------------------------------


async def _on_startup(app: web.Application) -> None:
    """Create the shared ``ClientSession`` used for downstream requests."""
    connector = TCPConnector(limit=app["connector_limit"])
    app["client_session"] = ClientSession(connector=connector)
    app["start_time"] = time.monotonic()
    logger.info(
        "Proxy started — listening on %s:%s with %d agent(s)",
        app["listen_host"],
        app["listen_port"],
        len(app["endpoints"]),
    )


async def _on_cleanup(app: web.Application) -> None:
    """Close the shared ``ClientSession``."""
    session: ClientSession | None = app.get("client_session")
    if session is not None:
        await session.close()
    logger.info("Proxy shut down")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def run_proxy(
    hostfile: str,
    host: str = "0.0.0.0",
    port: int = 9090,
    connector_limit: int = 1024,
    timeout: float = 300.0,
) -> None:
    """Start the Aurora-Swarm HTTP reverse proxy.

    Args:
        hostfile: Path to the hostfile listing agent endpoints.
        host: Address to bind the proxy server to.
        port: TCP port to listen on.
        connector_limit: Maximum number of simultaneous outbound
            connections managed by the ``TCPConnector``.
        timeout: Default upstream request timeout in seconds.  Can be
            overridden per-request via the ``X-Timeout`` header.
    """
    endpoints = parse_hostfile(hostfile)
    if not endpoints:
        raise SystemExit("No endpoints found in hostfile — nothing to proxy")

    app = web.Application()

    # Store configuration on the app dict so handlers can access it.
    app["endpoints"] = endpoints
    app["connector_limit"] = connector_limit
    app["default_timeout"] = timeout
    app["listen_host"] = host
    app["listen_port"] = port

    # Register lifecycle hooks.
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # Register routes.
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/status", _status_handler)
    app.router.add_route("*", "/agent/{index}/{path:.*}", _proxy_handler)

    web.run_app(app, host=host, port=port, print=None)
