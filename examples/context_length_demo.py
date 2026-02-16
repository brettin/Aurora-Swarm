"""Example demonstrating context length configuration features.

This script shows how to:
1. Configure max_tokens via environment variables
2. Use explicit configuration parameters
3. Let dynamic sizing handle varying prompt lengths
4. Use aggregation presets for reduce operations
"""

import asyncio
import os
from aurora_swarm import VLLMPool, AgentEndpoint


async def demo_explicit_config():
    """Demo: Explicit configuration parameters."""
    print("=== Demo 1: Explicit Configuration ===\n")
    
    # Create endpoints (replace with your actual vLLM hosts)
    endpoints = [
        AgentEndpoint("localhost", 8000),
        AgentEndpoint("localhost", 8001),
    ]
    
    # Explicit configuration
    pool = VLLMPool(
        endpoints,
        model="openai/gpt-oss-120b",
        max_tokens=1024,                # Default for simple prompts
        max_tokens_aggregation=2048,    # For aggregation steps
        model_max_context=131072,       # Skip API query (optional)
        buffer=512,                     # Safety margin
    )
    
    print(f"Default max_tokens: {pool._max_tokens}")
    print(f"Aggregation max_tokens: {pool._max_tokens_aggregation}")
    print(f"Model max context: {pool._model_max_context}")
    print(f"Buffer: {pool._buffer}\n")
    
    await pool.close()


async def demo_env_config():
    """Demo: Environment variable configuration."""
    print("=== Demo 2: Environment Variable Configuration ===\n")
    
    # Set environment variables
    os.environ["AURORA_SWARM_MAX_TOKENS"] = "2048"
    os.environ["AURORA_SWARM_MAX_TOKENS_AGGREGATION"] = "4096"
    os.environ["AURORA_SWARM_MODEL_MAX_CONTEXT"] = "131072"
    
    endpoints = [AgentEndpoint("localhost", 8000)]
    pool = VLLMPool(endpoints)
    
    print(f"Max tokens (from env): {pool._max_tokens}")
    print(f"Aggregation max tokens (from env): {pool._max_tokens_aggregation}")
    print(f"Model context (from env): {pool._model_max_context}\n")
    
    # Clean up env vars
    del os.environ["AURORA_SWARM_MAX_TOKENS"]
    del os.environ["AURORA_SWARM_MAX_TOKENS_AGGREGATION"]
    del os.environ["AURORA_SWARM_MODEL_MAX_CONTEXT"]
    
    await pool.close()


async def demo_dynamic_sizing():
    """Demo: Dynamic sizing based on prompt length."""
    print("=== Demo 3: Dynamic Sizing ===\n")
    
    endpoints = [AgentEndpoint("localhost", 8000)]
    pool = VLLMPool(
        endpoints,
        max_tokens=1024,
        model_max_context=131072,
    )
    
    # Short prompt - uses default max_tokens (capped by model context)
    short_prompt = "Hello, how are you?"
    print(f"Short prompt ({len(short_prompt)} chars):")
    print(f"  Estimated tokens: {len(short_prompt) // 4}")
    print(f"  Will use: min(1024, 131072 - {len(short_prompt) // 4} - 512)")
    print(f"  = min(1024, {131072 - len(short_prompt) // 4 - 512}) = 1024\n")
    
    # Very long prompt - dynamic sizing reduces max_tokens
    long_prompt = "Context: " + "x" * 500000  # ~500KB
    print(f"Long prompt ({len(long_prompt)} chars):")
    print(f"  Estimated tokens: {len(long_prompt) // 4}")
    print(f"  Will use: min(1024, 131072 - {len(long_prompt) // 4} - 512)")
    print(f"  = min(1024, {131072 - len(long_prompt) // 4 - 512})")
    print(f"  = {min(1024, max(128, 131072 - len(long_prompt) // 4 - 512))}\n")
    
    await pool.close()


async def demo_aggregation_pattern():
    """Demo: Using aggregation presets in patterns."""
    print("=== Demo 4: Aggregation Pattern ===\n")
    
    endpoints = [AgentEndpoint("localhost", 8000)]
    pool = VLLMPool(
        endpoints,
        max_tokens=1024,
        max_tokens_aggregation=2048,
    )
    
    print("In broadcast_and_reduce:")
    print(f"  - Broadcast step uses: {pool._max_tokens} tokens (default)")
    print(f"  - Reduce step uses: {pool._max_tokens_aggregation} tokens (aggregation)")
    print("\nThis ensures the reduce step has enough capacity for:")
    print("  - Concatenated responses from all agents")
    print("  - Reasoning overhead for synthesis")
    print("  - Final summarized output\n")
    
    await pool.close()


async def main():
    """Run all demos."""
    await demo_explicit_config()
    await demo_env_config()
    await demo_dynamic_sizing()
    await demo_aggregation_pattern()
    
    print("=== Summary ===\n")
    print("Context length management features:")
    print("  1. Environment variables: AURORA_SWARM_MAX_TOKENS, etc.")
    print("  2. Explicit parameters: max_tokens, max_tokens_aggregation")
    print("  3. Dynamic sizing: Adapts to prompt length automatically")
    print("  4. Aggregation presets: Larger budgets for reduce steps")
    print("  5. Per-request override: pool.post(..., max_tokens=N)")


if __name__ == "__main__":
    asyncio.run(main())
