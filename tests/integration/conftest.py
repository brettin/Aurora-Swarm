"""Shared fixtures for integration tests against live vLLM endpoints.

Hostfile location (checked in order):

1. ``pytest --hostfile=/path/to/hostfile``
2. ``AURORA_SWARM_HOSTFILE`` environment variable
3. ``SCRIPT_DIR/hostfile`` (same directory as this conftest)

The hostfile contains one agent per line in tab-separated format::

    hostname1	8000
    hostname2	8001

Blank lines and lines starting with ``#`` are ignored.

If the resolved hostfile does not exist the entire integration suite
is skipped automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.vllm_pool import VLLMPool

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Hostfile resolution
# ---------------------------------------------------------------------------

def _resolve_hostfile(config) -> Path:
    """Return the hostfile path from CLI option, env var, or default."""
    # 1. pytest CLI: --hostfile=/some/path
    cli_value = config.getoption("--hostfile", default=None)
    if cli_value:
        return Path(cli_value).resolve()

    # 2. Environment variable
    env_value = os.environ.get("AURORA_SWARM_HOSTFILE")
    if env_value:
        return Path(env_value).resolve()

    # 3. Default: same directory as this file
    return SCRIPT_DIR / "hostfile"


def pytest_addoption(parser):
    """Register the ``--hostfile`` command-line option."""
    parser.addoption(
        "--hostfile",
        action="store",
        default=None,
        help="Path to the tab-separated vLLM hostfile (hostname<TAB>port).",
    )


# ---------------------------------------------------------------------------
# Hostfile parser (tab-separated: hostname <TAB> port)
# ---------------------------------------------------------------------------

def _parse_vllm_hostfile(path: Path) -> list[AgentEndpoint]:
    """Parse a tab-separated hostfile and return AgentEndpoint objects."""
    endpoints: list[AgentEndpoint] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            host = parts[0].strip()
            port = int(parts[1].strip())
            endpoints.append(AgentEndpoint(host=host, port=port))
    return endpoints


# ---------------------------------------------------------------------------
# Skip-if-no-hostfile logic
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Auto-skip every test in this directory when the hostfile is absent."""
    hostfile = _resolve_hostfile(config)
    if not hostfile.exists():
        skip = pytest.mark.skip(reason=f"hostfile not found at {hostfile}")
        for item in items:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hostfile_path(request) -> Path:
    """Resolved hostfile path available to all integration tests."""
    return _resolve_hostfile(request.config)


@pytest_asyncio.fixture
async def vllm_pool(hostfile_path) -> AsyncIterator[VLLMPool]:
    """Yield a VLLMPool backed by all endpoints in the hostfile."""
    endpoints = _parse_vllm_hostfile(hostfile_path)
    pool = VLLMPool(
        endpoints,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        concurrency=64,
        connector_limit=128,
        timeout=300.0,
    )
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def vllm_pool_tagged(hostfile_path) -> AsyncIterator[VLLMPool]:
    """Yield a VLLMPool where the first half of agents are tagged
    ``role=hypotheses`` and the second half ``role=critiques``.

    Requires at least 2 agents in the hostfile.
    """
    endpoints = _parse_vllm_hostfile(hostfile_path)
    mid = len(endpoints) // 2
    tagged: list[AgentEndpoint] = []
    for i, ep in enumerate(endpoints):
        role = "hypotheses" if i < mid else "critiques"
        tagged.append(AgentEndpoint(host=ep.host, port=ep.port, tags={"role": role}))

    pool = VLLMPool(
        tagged,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        concurrency=64,
        connector_limit=128,
        timeout=300.0,
    )
    try:
        yield pool
    finally:
        await pool.close()
