"""Tests for AgentPool proxy mode and _global_indices tracking."""

from __future__ import annotations


from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_endpoints(n: int, tag_fn=None) -> list[AgentEndpoint]:
    """Create *n* dummy AgentEndpoint objects (no real server needed).

    Args:
        n: Number of endpoints to create.
        tag_fn: Optional callable ``(index) -> dict`` to assign tags.
    """
    eps: list[AgentEndpoint] = []
    for i in range(n):
        tags = tag_fn(i) if tag_fn is not None else {}
        eps.append(AgentEndpoint(host=f"host{i}", port=8000 + i, tags=tags))
    return eps


# ---------------------------------------------------------------------------
# Tests: _agent_base_url
# ---------------------------------------------------------------------------


def test_agent_base_url_no_proxy():
    """Without proxy, _agent_base_url returns the direct endpoint URL."""
    endpoints = _make_endpoints(3)
    pool = AgentPool(endpoints, proxy_url=None)
    assert pool._agent_base_url(0) == "http://host0:8000"
    assert pool._agent_base_url(1) == "http://host1:8001"
    assert pool._agent_base_url(2) == "http://host2:8002"


def test_agent_base_url_with_proxy():
    """With proxy, _agent_base_url returns proxy URL with /agent/{index}."""
    endpoints = _make_endpoints(3)
    pool = AgentPool(endpoints, proxy_url="http://login-node:9090")
    assert pool._agent_base_url(0) == "http://login-node:9090/agent/0"
    assert pool._agent_base_url(1) == "http://login-node:9090/agent/1"
    assert pool._agent_base_url(2) == "http://login-node:9090/agent/2"


def test_agent_base_url_with_proxy_trailing_slash():
    """Trailing slash on proxy_url is stripped correctly."""
    endpoints = _make_endpoints(2)
    pool = AgentPool(endpoints, proxy_url="http://login-node:9090/")
    assert pool._agent_base_url(0) == "http://login-node:9090/agent/0"
    assert pool._agent_base_url(1) == "http://login-node:9090/agent/1"


# ---------------------------------------------------------------------------
# Tests: _global_indices initialisation
# ---------------------------------------------------------------------------


def test_global_indices_init():
    """On init, _global_indices is [0, 1, 2, ...]."""
    endpoints = _make_endpoints(5)
    pool = AgentPool(endpoints)
    assert pool._global_indices == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Tests: _global_indices after sub-pool operations
# ---------------------------------------------------------------------------


def test_global_indices_slice():
    """slice() preserves the correct global indices."""
    endpoints = _make_endpoints(6)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    sub = pool.slice(2, 5)
    assert sub._global_indices == [2, 3, 4]
    assert sub.size == 3
    # Verify proxy URL uses global indices
    assert sub._agent_base_url(0) == "http://proxy:9090/agent/2"
    assert sub._agent_base_url(1) == "http://proxy:9090/agent/3"
    assert sub._agent_base_url(2) == "http://proxy:9090/agent/4"


def test_global_indices_select():
    """select() preserves the correct global indices."""
    endpoints = _make_endpoints(6)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    sub = pool.select([1, 3, 5])
    assert sub._global_indices == [1, 3, 5]
    assert sub.size == 3
    assert sub._agent_base_url(0) == "http://proxy:9090/agent/1"
    assert sub._agent_base_url(1) == "http://proxy:9090/agent/3"
    assert sub._agent_base_url(2) == "http://proxy:9090/agent/5"


def test_global_indices_by_tag():
    """by_tag() preserves the correct global indices."""

    def tag_fn(i):
        return {"role": "worker" if i % 2 == 0 else "critic"}

    endpoints = _make_endpoints(6, tag_fn=tag_fn)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    workers = pool.by_tag("role", "worker")
    # Indices 0, 2, 4 have role=worker
    assert workers._global_indices == [0, 2, 4]
    assert workers.size == 3
    assert workers._agent_base_url(0) == "http://proxy:9090/agent/0"
    assert workers._agent_base_url(1) == "http://proxy:9090/agent/2"
    assert workers._agent_base_url(2) == "http://proxy:9090/agent/4"


def test_global_indices_sample():
    """sample() returns global indices that are a subset of the parent's."""
    endpoints = _make_endpoints(10)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    sub = pool.sample(4)
    assert sub.size == 4
    assert len(sub._global_indices) == 4
    # All global indices must come from the parent's range
    for idx in sub._global_indices:
        assert idx in pool._global_indices


def test_global_indices_chained_operations():
    """Chained sub-pool operations preserve correct global indices."""
    endpoints = _make_endpoints(10)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    # slice first, then select within the slice
    sliced = pool.slice(2, 8)  # global indices [2, 3, 4, 5, 6, 7]
    selected = sliced.select([0, 2, 4])  # local 0,2,4 -> global 2,4,6
    assert selected._global_indices == [2, 4, 6]
    assert selected._agent_base_url(0) == "http://proxy:9090/agent/2"
    assert selected._agent_base_url(1) == "http://proxy:9090/agent/4"
    assert selected._agent_base_url(2) == "http://proxy:9090/agent/6"


# ---------------------------------------------------------------------------
# Tests: proxy_url from environment variable
# ---------------------------------------------------------------------------


def test_proxy_url_from_env(monkeypatch):
    """proxy_url is read from AURORA_SWARM_PROXY_URL env var."""
    monkeypatch.setenv("AURORA_SWARM_PROXY_URL", "http://env-proxy:9090")
    endpoints = _make_endpoints(2)
    pool = AgentPool(endpoints)
    assert pool.proxy_url == "http://env-proxy:9090"
    assert pool._agent_base_url(0) == "http://env-proxy:9090/agent/0"


def test_proxy_url_param_overrides_env(monkeypatch):
    """Explicit proxy_url parameter takes priority over env var."""
    monkeypatch.setenv("AURORA_SWARM_PROXY_URL", "http://env-proxy:9090")
    endpoints = _make_endpoints(2)
    pool = AgentPool(endpoints, proxy_url="http://param-proxy:8080")
    assert pool.proxy_url == "http://param-proxy:8080"
    assert pool._agent_base_url(0) == "http://param-proxy:8080/agent/0"


def test_proxy_url_none_when_unset(monkeypatch):
    """proxy_url is None when neither parameter nor env var is set."""
    monkeypatch.delenv("AURORA_SWARM_PROXY_URL", raising=False)
    endpoints = _make_endpoints(2)
    pool = AgentPool(endpoints)
    assert pool.proxy_url is None


# ---------------------------------------------------------------------------
# Tests: sub-pool inherits proxy_url
# ---------------------------------------------------------------------------


def test_sub_pool_inherits_proxy_url():
    """Child pools inherit the parent's proxy_url."""
    endpoints = _make_endpoints(4)
    pool = AgentPool(endpoints, proxy_url="http://proxy:9090")
    sub = pool.slice(0, 2)
    assert sub.proxy_url == "http://proxy:9090"


def test_sub_pool_inherits_no_proxy():
    """Child pools inherit None proxy_url from parent."""
    endpoints = _make_endpoints(4)
    pool = AgentPool(endpoints, proxy_url=None)
    sub = pool.slice(0, 2)
    assert sub.proxy_url is None
