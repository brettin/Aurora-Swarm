"""Broadcast pattern example with aggregation strategies.

This script broadcasts the same prompt to all agents except one (configurable
by index), then demonstrates two aggregation strategies on the collected
responses: majority_vote (categorical consensus) and concat (combined text).

USAGE EXAMPLES:
---------------

Using a hostfile:
    python examples/broadcast_aggregators.py --hostfile agents.txt

Using environment variable:
    export AURORA_SWARM_HOSTFILE=/path/to/agents.txt
    python examples/broadcast_aggregators.py

Broadcast to all except agent index 2 with a custom prompt:
    python examples/broadcast_aggregators.py --hostfile agents.txt \\
        --exclude-index 2 --prompt "Reply with one word: left or right."

HOSTFILE FORMAT:
----------------
One endpoint per line (see aurora_swarm.hostfile documentation):
    hostname1:8000
    hostname2:8001
    hostname3:8000 node=worker-01
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.aggregators import concat, failure_report, majority_vote
from aurora_swarm.patterns.broadcast import broadcast


def print_with_timestamp(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


async def main() -> int:
    """Main entry point."""
    default_prompt = (
        "Answer with one word: yes or no. Is the sky blue?"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        help="Path to hostfile with agent endpoints (default: AURORA_SWARM_HOSTFILE env var)",
    )
    parser.add_argument(
        "--exclude-index",
        type=int,
        default=0,
        help="Agent index to exclude from the broadcast (default: 0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help=f"Prompt to broadcast (default: {default_prompt!r})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name (default: meta-llama/Llama-3.1-70B-Instruct)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Maximum concurrent requests (default: 64)",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failure report to stderr",
    )
    parser.add_argument(
        "--max-concat-chars",
        type=int,
        default=2000,
        help="Max characters to print for concat output (default: 2000)",
    )
    args = parser.parse_args()

    hostfile_path = args.hostfile
    if hostfile_path is None:
        hostfile_env = os.environ.get("AURORA_SWARM_HOSTFILE")
        if hostfile_env:
            hostfile_path = Path(hostfile_env)
        else:
            print(
                "Error: No hostfile specified. Use --hostfile or set AURORA_SWARM_HOSTFILE",
                file=sys.stderr,
            )
            print("\nExample hostfile format:", file=sys.stderr)
            print("  hostname1:8000", file=sys.stderr)
            print("  hostname2:8001", file=sys.stderr)
            return 1

    if not hostfile_path.exists():
        print(f"Error: Hostfile not found: {hostfile_path}", file=sys.stderr)
        return 1

    print_with_timestamp("=" * 60)
    print_with_timestamp("Broadcast + Aggregators Example")
    print_with_timestamp("=" * 60)

    endpoints = parse_hostfile(hostfile_path)
    print_with_timestamp(f"Loaded {len(endpoints)} endpoints from {hostfile_path}")

    if args.exclude_index < 0 or args.exclude_index >= len(endpoints):
        print(
            f"Error: --exclude-index must be in [0, {len(endpoints) - 1}]",
            file=sys.stderr,
        )
        return 1

    indices = [i for i in range(len(endpoints)) if i != args.exclude_index]
    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    sub_pool = None
    try:
        sub_pool = pool.select(indices)
        print_with_timestamp(
            f"Broadcasting to {sub_pool.size} agents (excluding index {args.exclude_index})"
        )
        prompt_preview = args.prompt[:80] + "..." if len(args.prompt) > 80 else args.prompt
        print_with_timestamp(f"Prompt: {prompt_preview}")
        responses = await broadcast(sub_pool, args.prompt)
    finally:
        # Sub-pool may have created the aiohttp session on first use; close it so we don't leave
        # the session/connector open (parent pool's session may still be None).
        if sub_pool is not None:
            await sub_pool.close()
        await pool.close()
        await asyncio.sleep(0)  # yield so aiohttp cleanup can run
    print_with_timestamp(f"Received {len(responses)} responses")

    # Aggregation strategy 1: majority vote
    winner, confidence = majority_vote(responses)
    print("\nMajority vote:")
    print(f"  Winner: {winner!r}")
    print(f"  Confidence: {confidence:.2f}")

    # If no successful responses, show a short note so user knows why aggregators are empty
    report = failure_report(responses)
    if report["success_count"] == 0 and report["failure_count"] > 0:
        print_with_timestamp(f"  (No successful responses; {report['failure_count']} failure(s). Use --show-failures for details.)")

    # Aggregation strategy 2: concat
    combined = concat(responses, separator=" | ")
    if len(combined) > args.max_concat_chars:
        combined_display = combined[: args.max_concat_chars] + "..."
    else:
        combined_display = combined
    print("\nAll responses (concat):")
    print(f"  {combined_display}")

    if args.show_failures:
        report = failure_report(responses)
        print_with_timestamp("\nFailure report:")
        print_with_timestamp(f"  Total: {report['total']}, Success: {report['success_count']}, Failures: {report['failure_count']}")
        for f in report["failures"]:
            print_with_timestamp(f"  Agent {f['agent_index']}: {f['error']}")

    print_with_timestamp("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
