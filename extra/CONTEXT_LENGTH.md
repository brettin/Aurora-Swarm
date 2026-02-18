# Context Length Management

This document describes the context length configuration and dynamic sizing features in Aurora Swarm.

## Overview

Aurora Swarm implements a hybrid approach combining **config-based** and **dynamic** context length management:

1. **Config-based**: Set default and aggregation-specific token limits via environment variables or constructor parameters
2. **Dynamic sizing**: Automatically adjust `max_tokens` based on prompt length and model capacity to prevent truncation
3. **Aggregation presets**: Use larger token budgets for reduce/aggregation steps where prompts grow

## Features

### 1. Environment Variable Configuration

Configure token limits globally via environment variables:

```bash
export AURORA_SWARM_MAX_TOKENS=1024              # Default max tokens
export AURORA_SWARM_MAX_TOKENS_AGGREGATION=2048  # For aggregation/reduce steps
export AURORA_SWARM_MODEL_MAX_CONTEXT=131072     # Model's max context (optional)
```

### 2. Explicit Configuration

Set limits programmatically when creating a pool:

```python
from aurora_swarm import VLLMPool, AgentEndpoint

pool = VLLMPool(
    endpoints=[AgentEndpoint("host1", 8000)],
    model="openai/gpt-oss-120b",
    max_tokens=1024,                    # Default for simple prompts
    max_tokens_aggregation=2048,        # For aggregation steps
    model_max_context=131072,           # Optional: skip API query
    buffer=512,                         # Safety margin (default)
)
```

**Priority**: Explicit parameters > Environment variables > Defaults

### 3. Dynamic Sizing

`VLLMPool.post()` automatically computes appropriate `max_tokens` when not explicitly provided:

```python
# Formula
max_tokens = min(
    configured_max_tokens,
    max(128, model_max_context - estimated_prompt_tokens - buffer)
)
```

**How it works:**
1. Fetches model's max context length from vLLM `/v1/models` endpoint (cached after first call)
2. Estimates prompt tokens using `len(prompt) // 4` heuristic
3. Computes safe `max_tokens` that won't exceed model capacity
4. Respects configured cap as upper bound

**Benefits:**
- Prevents "context length exceeded" errors
- Automatically adapts to varying prompt sizes
- Accounts for reasoning model overhead via buffer

### 4. Aggregation Presets

Patterns that concatenate responses (reduce steps) use `max_tokens_aggregation`:

| Pattern | Leaf/Simple | Aggregation |
|---------|-------------|-------------|
| `broadcast_and_reduce` | `max_tokens` | `max_tokens_aggregation` (reduce step) |
| `fan_out_fan_in` | `max_tokens` | `max_tokens_aggregation` (collector) |
| `run_pipeline` | `max_tokens` (stage 1) | Dynamic sizing (stages 2+) |
| `tree_reduce` | `max_tokens` (leaves) | Dynamic sizing (supervisors) |

**Implementation**: Patterns call `pool.post(..., max_tokens=pool._max_tokens_aggregation)` for reduce steps.

### 5. Per-Request Override

Explicitly set `max_tokens` for individual requests:

```python
response = await pool.post(
    agent_index=0,
    prompt="Your prompt here",
    max_tokens=512  # Override default/dynamic sizing
)
```

## Examples

### Simple Usage (Dynamic Sizing)

```python
import asyncio
from aurora_swarm import VLLMPool, AgentEndpoint

async def main():
    pool = VLLMPool([AgentEndpoint("host1", 8000)])
    
    # Short prompt - uses default max_tokens
    response = await pool.post(0, "Hello")
    
    # Long prompt - dynamic sizing reduces max_tokens automatically
    long_prompt = "Context: " + "x" * 100000
    response = await pool.post(0, long_prompt)
    
    await pool.close()

asyncio.run(main())
```

### Using Aggregation Patterns

```python
from aurora_swarm.patterns.broadcast import broadcast_and_reduce

async def main():
    pool = VLLMPool(
        endpoints=[...],
        max_tokens=1024,                # Broadcast step
        max_tokens_aggregation=2048,    # Reduce step
    )
    
    result = await broadcast_and_reduce(
        pool,
        prompt="What are benefits of renewable energy?",
        reduce_prompt="Summarize these responses:\n{responses}",
    )
    
    await pool.close()
```

### Environment-Based Configuration

```bash
# Set environment variables
export AURORA_SWARM_MAX_TOKENS=2048
export AURORA_SWARM_MAX_TOKENS_AGGREGATION=4096

# Run your script
python your_script.py
```

```python
# In your_script.py
pool = VLLMPool(endpoints)  # Uses env vars automatically
```

## Implementation Details

### VLLMPool Changes

**New Parameters:**
- `max_tokens: int | None = None` - Default max tokens (None uses env or 512)
- `max_tokens_aggregation: int | None = None` - For aggregation (None = 2× max_tokens)
- `model_max_context: int | None = None` - Model's max context (None fetches from API)
- `buffer: int = 512` - Safety margin for dynamic sizing

**New Methods:**
- `async def _get_model_max_context() -> int` - Fetch and cache model metadata

**Modified Methods:**
- `async def post(..., max_tokens: int | None = None)` - Accepts per-request override

### AgentPool Changes

**Modified Methods:**
- `async def post(..., max_tokens: int | None = None)` - Signature change for consistency (parameter ignored by base class)

### Pattern Changes

**Modified Functions:**
- `broadcast_and_reduce()` - Passes `max_tokens_aggregation` to reduce step
- `fan_out_fan_in()` - Passes `max_tokens_aggregation` to collector

**Note**: Other patterns (`run_pipeline`, `tree_reduce`, `blackboard`) rely on dynamic sizing automatically since they call `post()` internally.

## Testing

Run the test suite:

```bash
# Unit tests (no vLLM servers required)
pytest tests/test_vllm_pool_config.py -v

# All unit tests
pytest tests/ -v --ignore=tests/integration

# Integration tests (requires vLLM servers + hostfile)
pytest tests/integration/ -v --hostfile=/path/to/hostfile
```

Run the demonstration script:

```bash
python examples/context_length_demo.py
```

## Backward Compatibility

All changes are backward compatible:
- Existing code without `max_tokens` parameters continues to work
- Default behavior unchanged (512 tokens for simple prompts)
- Tests pass without modification
- No breaking API changes

## Reasoning Model Support

The implementation specifically addresses reasoning models (like `gpt-oss-120b`) that:
- Return `content: null` when reasoning tokens exhaust the budget
- Populate `reasoning_content` with chain-of-thought tokens
- Require larger token budgets for complex tasks

**Mitigations:**
1. **Fallback chain**: `content or reasoning_content or ""`
2. **Buffer**: 512-token safety margin for reasoning overhead
3. **Aggregation presets**: 2× larger budget for reduce steps
4. **Dynamic sizing**: Adapts to long prompts automatically

## Future Enhancements

Potential improvements:
1. **Accurate token counting**: Use `tiktoken` instead of char÷4 heuristic
2. **Per-pattern hints**: Pattern-specific default `max_tokens`
3. **Adaptive buffer**: Learn optimal buffer from usage patterns
4. **Token usage tracking**: Log and analyze token consumption
5. **Config file support**: YAML/TOML config in addition to env vars

## References

- vLLM OpenAI API: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- OpenAI Chat Completions: https://platform.openai.com/docs/api-reference/chat
- Implementation Plan: `.cursor/plans/context_length_config_*.plan.md`
