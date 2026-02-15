"""Hostfile parser for Aurora agent endpoints.

Hostfile format — one agent per line::

    host1:8000 node=aurora-0001 role=worker
    host2:8000 node=aurora-0002 role=critic

Blank lines and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AgentEndpoint:
    """A single agent's network address plus optional metadata tags."""

    host: str
    port: int
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def parse_hostfile(path: str | Path) -> list[AgentEndpoint]:
    """Parse a hostfile and return a list of :class:`AgentEndpoint` objects.

    Parameters
    ----------
    path:
        Path to the hostfile.

    Returns
    -------
    list[AgentEndpoint]
        Parsed endpoints in file order.
    """
    endpoints: list[AgentEndpoint] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            host_port = parts[0]
            if ":" in host_port:
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 8000  # default
            tags: dict[str, str] = {}
            for token in parts[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    tags[key] = value
            endpoints.append(AgentEndpoint(host=host, port=port, tags=tags))
    return endpoints
