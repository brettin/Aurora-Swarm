"""Hostfile parser for Aurora agent endpoints.

Hostfile format — one agent per line (tab-delimited)::

    host1\t8000\tnode=aurora-0001\trole=worker
    host2\t8000\tnode=aurora-0002\trole=critic

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
            parts = line.split('\t')
            host = parts[0]
            # Port is in second column, defaults to 8000 if not provided
            if len(parts) > 1 and parts[1].isdigit():
                port = int(parts[1])
                tag_start = 2
            else:
                port = 8000
                tag_start = 1
            # Parse optional tags from remaining columns
            tags: dict[str, str] = {}
            for token in parts[tag_start:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    tags[key] = value
            endpoints.append(AgentEndpoint(host=host, port=port, tags=tags))
    return endpoints
