"""Test whether gpt-oss-120b (or given model) supports the embeddings API.

Uses the scatter-gather pattern: scatter several input texts across agents
round-robin, then gather embedding results in input order. Prints success/failure
and a short summary (e.g. embedding dimension) per input.

USAGE:
    python examples/test_embeddings_sg.py --hostfile agents.txt
    python examples/test_embeddings_sg.py --hostfile agents.txt --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

from aurora_swarm import EmbeddingPool, parse_hostfile
from aurora_swarm.patterns import scatter_gather_embeddings


def print_ts(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


async def print_raw_embeddings_response(
    url: str,
    model: str,
    input_text: str,
    timeout: float,
) -> None:
    """POST to /v1/embeddings and print raw status, headers, and body."""
    payload = {"model": model, "input": [input_text]}
    print_ts("Raw request: POST " + url)
    print_ts("Request body: " + json.dumps(payload, indent=2))
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            body = await resp.text()
            print_ts("Raw server response:")
            print(f"  Status: {resp.status} {resp.reason}", file=sys.stderr)
            for k, v in resp.headers.items():
                print(f"  {k}: {v}", file=sys.stderr)
            print_ts("Response body:")
            try:
                parsed = json.loads(body)
                print(json.dumps(parsed, indent=2), file=sys.stderr)
                err_msg = (parsed.get("error") or {}).get("message") or ""
                if "does not exist" in err_msg or resp.status == 404:
                    print_ts("Hint: Query the server with GET /v1/models to see available model IDs.")
            except Exception:
                print(body, file=sys.stderr)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        help="Path to hostfile (default: AURORA_SWARM_HOSTFILE)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent requests (default: 16)",
    )
    parser.add_argument(
        "--role",
        default="embed",
        help="Filter endpoints by role tag (default: embed). Use empty string to disable.",
    )
    args = parser.parse_args()

    hostfile = args.hostfile
    if hostfile is None:
        hostfile = os.environ.get("AURORA_SWARM_HOSTFILE")
        if hostfile:
            hostfile = Path(hostfile)
    if not hostfile or not Path(hostfile).exists():
        print("Error: Need --hostfile or AURORA_SWARM_HOSTFILE pointing to a file.", file=sys.stderr)
        return 1

    hostfile_path = Path(hostfile).resolve()
    endpoints = parse_hostfile(hostfile_path)
    if args.role:
        endpoints = [ep for ep in endpoints if ep.tags.get("role") == args.role]
        if not endpoints:
            print(f"Error: No endpoints in hostfile with role={args.role!r}.", file=sys.stderr)
            return 1
    if not endpoints:
        print("Error: No endpoints in hostfile.", file=sys.stderr)
        return 1

    print_ts("=" * 60)
    print_ts("Embeddings test (scatter-gather)")
    print_ts("=" * 60)
    print_ts(f"Hostfile: {hostfile_path}, endpoints: {len(endpoints)}, model: {args.model}")

    embed_pool = EmbeddingPool(
        endpoints,
        model=args.model,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )

    # Small test inputs scattered across agents
    test_texts = [
        "Hello world.",
        "Embeddings test.",
        "Scatter-gather pattern.",
    ]

    # Probe first endpoint to show raw server response
    first_url = f"{endpoints[0].url}/v1/embeddings"
    await print_raw_embeddings_response(
        first_url, args.model, test_texts[0], args.timeout
    )
    print_ts("-" * 60)

    print_ts(f"Scattering {len(test_texts)} texts across {embed_pool.size} agents...")
    async with embed_pool:
        results = await scatter_gather_embeddings(embed_pool, test_texts)

    ok = sum(1 for r in results if r.success)
    print_ts(f"Gathered: {ok}/{len(results)} succeeded.")
    print()

    for i, r in enumerate(results):
        text = test_texts[i] if i < len(test_texts) else ""
        status = "OK" if r.success else "FAIL"
        dim = len(r.embedding) if r.embedding else 0
        preview = (text[:40] + "…") if len(text) > 40 else text
        print(f"  [{i}] {status} (agent {r.agent_index}) {preview!r}")
        if r.success and r.embedding:
            print(f"       dimension={dim}, first 3 values: {r.embedding[:3]}")
        if r.error:
            print(f"       error: {r.error}")

    print_ts("=" * 60)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
