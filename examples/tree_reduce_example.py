"""Tree-reduce pattern example — hierarchical summarization.

Leaf agents produce initial responses. Groups of responses are fed to
supervisor agents that summarize them recursively until a single answer
remains.

Two modes:
  - Broadcast: Same prompt sent to all agents; responses are reduced.
  - Items: Different prompts (with {item} placeholder) scattered across
    agents; responses are reduced.

USAGE EXAMPLES:
---------------

Using a hostfile (broadcast mode):
    python examples/tree_reduce_example.py --hostfile agents.txt

Using environment variable:
    export AURORA_SWARM_HOSTFILE=/path/to/agents.txt
    python examples/tree_reduce_example.py

Items mode (scatter different tasks, then reduce):
    python examples/tree_reduce_example.py --hostfile agents.txt \\
        --items "water,methane,ethanol,glucose" \\
        --prompt "Describe the molecular formula of {item} in one sentence."

Custom reduce prompt and fanin:
    python examples/tree_reduce_example.py --hostfile agents.txt \\
        --reduce-prompt "Merge into summary (level {level}):\\n{responses}" \\
        --fanin 4

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
from aurora_swarm.patterns.tree_reduce import tree_reduce


def print_with_timestamp(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


async def main() -> int:
    """Main entry point."""
    default_prompt = (
        "Name one benefit of renewable energy. Keep it to one sentence."
    )
    default_reduce_prompt = (
        "Below are several responses. Merge them into a concise summary "
        "(level {level}):\n\n{responses}"
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
        "--prompt",
        type=str,
        default=default_prompt,
        help=f"Leaf-level prompt (default: {default_prompt!r})",
    )
    parser.add_argument(
        "--reduce-prompt",
        type=str,
        default=default_reduce_prompt,
        help="Supervisor summarization prompt with {responses} and optionally {level} (default: merge summary)",
    )
    parser.add_argument(
        "--items",
        type=str,
        default=None,
        metavar="CSV",
        help="Comma-separated items for scatter mode; prompt must contain {item} placeholder",
    )
    parser.add_argument(
        "--fanin",
        type=int,
        default=4,
        help="Number of responses each supervisor handles per group (default: 4)",
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
        help="Print failure details to stderr if reduction fails",
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

    if "{responses}" not in args.reduce_prompt:
        print(
            "Error: --reduce-prompt must contain {responses} placeholder",
            file=sys.stderr,
        )
        return 1

    items_list = None
    if args.items is not None:
        items_list = [s.strip() for s in args.items.split(",") if s.strip()]
        if not items_list:
            print("Error: --items must contain at least one item", file=sys.stderr)
            return 1
        if "{item}" not in args.prompt:
            print(
                "Error: --prompt must contain {item} placeholder when using --items",
                file=sys.stderr,
            )
            return 1

    print_with_timestamp("=" * 60)
    print_with_timestamp("Tree-Reduce Example")
    print_with_timestamp("=" * 60)

    endpoints = parse_hostfile(hostfile_path)
    print_with_timestamp(f"Loaded {len(endpoints)} endpoints from {hostfile_path}")

    pool = VLLMPool(
        endpoints,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )

    try:
        if items_list is not None:
            print_with_timestamp(
                f"Scatter mode: {len(items_list)} items, fanin={args.fanin}"
            )
            prompt_preview = args.prompt[:60] + "..." if len(args.prompt) > 60 else args.prompt
            print_with_timestamp(f"Prompt template: {prompt_preview}")
        else:
            print_with_timestamp(f"Broadcast mode: fanin={args.fanin}")
            prompt_preview = args.prompt[:80] + "..." if len(args.prompt) > 80 else args.prompt
            print_with_timestamp(f"Prompt: {prompt_preview}")

        result = await tree_reduce(
            pool=pool,
            prompt=args.prompt,
            reduce_prompt=args.reduce_prompt,
            fanin=args.fanin,
            items=items_list,
        )
    finally:
        await pool.close()
        await asyncio.sleep(0)

    if not result.success:
        print_with_timestamp(f"Tree-reduce failed: {result.error}")
        if args.show_failures and result.error:
            print_with_timestamp(f"  Error details: {result.error}")
        print_with_timestamp("=" * 60)
        return 1

    print("\nFinal result:")
    print("-" * 40)
    print(result.text)
    print("-" * 40)

    print_with_timestamp("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
