# Batch Prompting - Quick Reference

## Quick Start

```python
from aurora_swarm import VLLMPool, parse_hostfile
from aurora_swarm.patterns.scatter_gather import scatter_gather

endpoints = parse_hostfile("agents.hostfile")
async with VLLMPool(endpoints, model="openai/gpt-oss-120b") as pool:
    prompts = [f"Task {i}" for i in range(1000)]
    responses = await scatter_gather(pool, prompts)  # Automatic batching!
```

## Key Benefits

| Metric | Without Batch | With Batch | Improvement |
|--------|---------------|------------|-------------|
| HTTP Requests (1000 prompts, 4 agents) | 1,000 | 4 | **250× fewer** |
| HTTP Requests (10,000 prompts, 100 agents) | 10,000 | 100 | **100× fewer** |
| Measured Speedup (20 prompts, 4 agents) | baseline | 1.12× faster | **12% faster** |

## Configuration

### Enable Batching (default)
```python
pool = VLLMPool(endpoints, model="model-name", use_batch=True)
```

### Disable Batching
```python
pool = VLLMPool(endpoints, model="model-name", use_batch=False)
```

### Manual Batch Control
```python
# Send batch to specific agent
responses = await pool.post_batch(agent_index=0, prompts=["p1", "p2", "p3"])

# Use send_all_batched directly
responses = await pool.send_all_batched(prompts, max_tokens=512)
```

## Patterns That Use Batching

All automatically use batching when `VLLMPool` is used:

- ✅ `scatter_gather()` - Distributes prompts with batching
- ✅ `map_gather()` - Uses scatter_gather internally
- ✅ `tree_reduce()` - Batches both leaf and supervisor prompts

## API Endpoints

| Mode | Endpoint | Accepts | Use Case |
|------|----------|---------|----------|
| **Batch** | `/v1/completions` | `prompt: str \| list[str]` | High throughput |
| **Non-batch** | `/v1/chat/completions` | `messages: list[dict]` | Single requests |

## Testing

```bash
# Unit tests
pytest tests/test_vllm_pool.py -v

# Integration tests (requires running vLLM)
pytest tests/integration/ --hostfile=/path/to/hostfile -v

# Performance comparison
python test_batch_integration.py /path/to/hostfile
```

## Common Commands

```bash
# Setup
module load frameworks
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm

# Run with real endpoints
export AURORA_SWARM_HOSTFILE=/path/to/hostfile
python examples/scatter_gather_coli.py /path/to/data/ \
    --num-files 10 --output results.txt
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection errors | Verify vLLM endpoints: `curl http://host:port/v1/completions` |
| Empty responses | Increase `max_tokens` or check prompt formatting |
| Timeout errors | Increase timeout: `VLLMPool(endpoints, timeout=600.0)` |
| Chat format issues | Manually format: `prompt = "<\|user\|>\\n{text}\\n<\|assistant\|>\\n"` |

## Documentation Links

- **[BATCH_PROMPTING.md](BATCH_PROMPTING.md)** - Full implementation guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete summary
- **[docs/batch_prompting.rst](docs/batch_prompting.rst)** - Sphinx documentation
- **[README.md](README.md)** - Project overview

## Key Methods

```python
# VLLMPool methods
await pool.post_batch(agent_index, prompts, max_tokens=None)
await pool.send_all_batched(prompts, max_tokens=None)

# Base AgentPool (fallback)
await pool.send_all_batched(prompts)  # Delegates to send_all()
```

## Environment Variables

```bash
export AURORA_SWARM_HOSTFILE=/path/to/hostfile
export AURORA_SWARM_MAX_TOKENS=1024
export AURORA_SWARM_MAX_TOKENS_AGGREGATION=2048
```

## Performance Tips

1. **Use VLLMPool** - Base AgentPool doesn't support batching
2. **Batch size matters** - More prompts per agent = fewer requests
3. **Monitor timeouts** - Large batches may need longer timeouts
4. **Check model support** - Verify `/v1/completions` endpoint works
5. **Format prompts** - For chat models, add templates manually if needed

## Backward Compatibility

✅ All existing code works unchanged  
✅ Can disable with `use_batch=False`  
✅ Patterns automatically detect batch support  
✅ Non-VLLMPool pools unaffected
