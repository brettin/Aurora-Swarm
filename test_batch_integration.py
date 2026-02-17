#!/usr/bin/env python
"""Integration test for batch prompting with real vLLM endpoints.

Usage:
    python test_batch_integration.py /path/to/hostfile
"""

import asyncio
import sys
import time
from pathlib import Path

from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather


async def test_batch_vs_non_batch(hostfile_path: Path, num_prompts: int = 20):
    """Compare batch vs non-batch performance."""
    
    print(f"Loading endpoints from {hostfile_path}")
    endpoints = parse_hostfile(hostfile_path)
    print(f"Found {len(endpoints)} endpoints")
    
    # Create test prompts
    prompts = [
        f"What is the capital of country number {i}? Answer in one word."
        for i in range(num_prompts)
    ]
    
    print(f"\nTesting with {num_prompts} prompts")
    print("=" * 60)
    
    # Test 1: With batching enabled (default)
    print("\n1. WITH BATCH MODE (use_batch=True)")
    print("-" * 60)
    async with VLLMPool(
        endpoints,
        model="openai/gpt-oss-120b",
        max_tokens=50,
        use_batch=True,
        concurrency=64,
        timeout=120.0,
    ) as pool:
        start = time.time()
        responses_batched = await scatter_gather(pool, prompts)
        elapsed_batched = time.time() - start
        
        success_count = sum(1 for r in responses_batched if r.success)
        print(f"Completed: {success_count}/{len(prompts)} successful")
        print(f"Time: {elapsed_batched:.2f}s")
        print(f"Throughput: {num_prompts/elapsed_batched:.2f} prompts/sec")
        
        # Show first 3 responses
        print("\nFirst 3 responses:")
        for i, resp in enumerate(responses_batched[:3]):
            if resp.success:
                text = resp.text[:100].replace('\n', ' ')
                print(f"  [{i}] {text}...")
            else:
                print(f"  [{i}] ERROR: {resp.error}")
    
    # Test 2: Without batching
    print("\n2. WITHOUT BATCH MODE (use_batch=False)")
    print("-" * 60)
    async with VLLMPool(
        endpoints,
        model="openai/gpt-oss-120b",
        max_tokens=50,
        use_batch=False,  # Disable batching
        concurrency=64,
        timeout=120.0,
    ) as pool:
        start = time.time()
        responses_non_batched = await scatter_gather(pool, prompts)
        elapsed_non_batched = time.time() - start
        
        success_count = sum(1 for r in responses_non_batched if r.success)
        print(f"Completed: {success_count}/{len(prompts)} successful")
        print(f"Time: {elapsed_non_batched:.2f}s")
        print(f"Throughput: {num_prompts/elapsed_non_batched:.2f} prompts/sec")
        
        # Show first 3 responses
        print("\nFirst 3 responses:")
        for i, resp in enumerate(responses_non_batched[:3]):
            if resp.success:
                text = resp.text[:100].replace('\n', ' ')
                print(f"  [{i}] {text}...")
            else:
                print(f"  [{i}] ERROR: {resp.error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Batch mode:     {elapsed_batched:.2f}s ({num_prompts/elapsed_batched:.2f} prompts/sec)")
    print(f"Non-batch mode: {elapsed_non_batched:.2f}s ({num_prompts/elapsed_non_batched:.2f} prompts/sec)")
    
    if elapsed_batched < elapsed_non_batched:
        speedup = elapsed_non_batched / elapsed_batched
        print(f"\nBatch mode is {speedup:.2f}x faster!")
    else:
        slowdown = elapsed_batched / elapsed_non_batched
        print(f"\nNote: Batch mode was {slowdown:.2f}x slower (may indicate overhead for small batches)")
    
    print("\n✓ Integration test completed successfully")


async def main():
    import os
    
    # Try to get hostfile from arguments, then environment variable
    if len(sys.argv) >= 2:
        hostfile_path = Path(sys.argv[1])
    elif "AURORA_SWARM_HOSTFILE" in os.environ:
        hostfile_path = Path(os.environ["AURORA_SWARM_HOSTFILE"])
        print(f"Using hostfile from AURORA_SWARM_HOSTFILE: {hostfile_path}")
    else:
        print("Error: Hostfile path required")
        print()
        print("Usage:")
        print("  python test_batch_integration.py /path/to/hostfile")
        print()
        print("Or set environment variable:")
        print("  export AURORA_SWARM_HOSTFILE=/path/to/hostfile")
        print("  python test_batch_integration.py")
        sys.exit(1)
    
    if not hostfile_path.exists():
        print(f"Error: Hostfile not found: {hostfile_path}")
        sys.exit(1)
    
    await test_batch_vs_non_batch(hostfile_path, num_prompts=20)


if __name__ == "__main__":
    asyncio.run(main())
