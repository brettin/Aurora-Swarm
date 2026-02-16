"""CLI entry point for the Aurora-Swarm HTTP reverse proxy.

Usage::

    aurora-swarm-proxy --hostfile /path/to/agents.txt [OPTIONS]

The hostfile path can also be supplied via the ``AURORA_SWARM_HOSTFILE``
environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``aurora-swarm-proxy``."""
    parser = argparse.ArgumentParser(
        prog="aurora-swarm-proxy",
        description=(
            "HTTP reverse proxy that multiplexes Aurora-Swarm agent "
            "endpoints behind a single TCP port."
        ),
    )
    parser.add_argument(
        "--hostfile",
        "-f",
        default=os.environ.get("AURORA_SWARM_HOSTFILE"),
        help=(
            "Path to the hostfile listing agent endpoints. "
            "Falls back to the AURORA_SWARM_HOSTFILE environment variable."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to bind the proxy server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9090,
        help="TCP port to listen on (default: 9090).",
    )
    parser.add_argument(
        "--connector-limit",
        type=int,
        default=1024,
        help=(
            "Maximum number of simultaneous outbound connections "
            "to downstream agents (default: 1024)."
        ),
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=300.0,
        help=(
            "Default upstream request timeout in seconds (default: 300). "
            "Can be overridden per-request via the X-Timeout header."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and start the reverse proxy."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.hostfile is None:
        parser.error(
            "the --hostfile/-f argument is required (or set AURORA_SWARM_HOSTFILE)"
        )

    # Configure root logger so that both aurora_swarm.proxy and aiohttp
    # messages are visible.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Import here to avoid pulling in aiohttp when the module is merely
    # imported (e.g. during tests or IDE introspection).
    from aurora_swarm.proxy import run_proxy

    run_proxy(
        hostfile=args.hostfile,
        host=args.host,
        port=args.port,
        connector_limit=args.connector_limit,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
