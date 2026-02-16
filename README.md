# Aurora Swarm


Communication patterns for orchestrating large-scale LLM agent swarms on Aurora.

Aurora Swarm provides an async Python library for coordinating thousands of LLM agent endpoints using common distributed communication patterns — broadcast, scatter-gather, tree-reduce, blackboard, and multi-stage pipelines. It manages pooled HTTP connections with semaphore-based concurrency control so you can safely drive 1,000–4,000+ agents from a single orchestrator process.

**API reference (GitHub Pages):** [https://brettin.github.io/Aurora-Swarm/](https://brettin.github.io/Aurora-Swarm/)

---

## Getting Started

### Prerequisites

- Access to a system with [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- Python 3.11 or later
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Aurora-Swarm.git
cd Aurora-Swarm
```

### 2. Create and activate a Conda environment

**On HPC (e.g. Aurora):** load the frameworks module and activate the project conda env before any `pip install`:

```bash
module load frameworks
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm
```

**Otherwise**, create and activate a local env:

```bash
module load frameworks
conda create -n aurora-swarm python=3.11 -y
conda activate aurora-swarm
```

### 3. Install the package

Install in editable (development) mode so that local changes are picked up immediately:

```bash
pip install -e .
```

This pulls in the core runtime dependency (`aiohttp>=3.9`).

### 4. Install development / test dependencies

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `pytest-asyncio`, and everything needed to run the test suite.

### 5. Verify the installation

```bash
python -c "import aurora_swarm; print('aurora_swarm imported successfully')"
```

### 6. Run the tests

```bash
pytest -v
```

The tests spin up lightweight mock HTTP servers on localhost so no external agents are needed.

### 7. Build and view the documentation (optional)

Published docs are at **https://brettin.github.io/Aurora-Swarm/** (built and deployed from `main` via GitHub Actions). In repo **Settings → Pages**, set "Build and deployment" source to **GitHub Actions** if it is not already.

To build and view locally (on HPC, load the environment first):

```bash
module load frameworks
conda activate /lus/flare/projects/ModCon/brettin/conda_envs/swarm
pip install -e ".[docs]"
cd docs && make html && cd ..   # output in docs/_build/
# then open docs/_build/index.html in a browser
```

---

## Project Structure

```
Aurora-Swarm/
├── aurora_swarm/
│   ├── __init__.py            # Public API re-exports
│   ├── hostfile.py            # Hostfile parser (AgentEndpoint)
│   ├── pool.py                # AgentPool — async connection pool
│   ├── aggregators.py         # Response aggregation strategies
│   └── patterns/
│       ├── __init__.py
│       ├── broadcast.py       # Pattern 1 — Broadcast
│       ├── scatter_gather.py  # Pattern 2 — Scatter-Gather
│       ├── tree_reduce.py     # Pattern 3 — Tree-Reduce
│       ├── blackboard.py      # Pattern 4 — Blackboard (shared-state)
│       └── pipeline.py        # Pattern 5 — Pipeline (multi-stage DAG)
├── tests/
│   ├── conftest.py            # Shared fixtures (mock servers, pools)
│   ├── test_broadcast.py
│   ├── test_scatter_gather.py
│   ├── test_tree_reduce.py
│   ├── test_blackboard.py
│   ├── test_pipeline.py
│   └── integration/          # Integration tests
│       ├── test_broadcast.py
│       ├── test_scatter_gather.py
│       ├── test_tree_reduce.py
│       ├── test_blackboard.py
│       └── test_pipeline.py
├── docs/
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst               # Landing page
│   ├── api.rst                 # API reference (autodoc)
│   ├── Makefile                # make html
│   └── _build/                 # make html output
└── pyproject.toml
```

---

## Quick Tour

### Hostfile

Agents are listed in a plain-text hostfile, one per line:

```
host1:8000 node=aurora-0001 role=worker
host2:8000 node=aurora-0002 role=critic
# blank lines and comments are ignored
```

Parse it with:

```python
from aurora_swarm import parse_hostfile

endpoints = parse_hostfile("agents.hostfile")
```

### AgentPool

`AgentPool` wraps a list of endpoints with pooled HTTP sessions and semaphore-based concurrency control:

```python
from aurora_swarm import AgentPool

async with AgentPool(endpoints, concurrency=512) as pool:
    response = await pool.post(0, "Hello, agent!")
    print(response.text)
```

### Communication Patterns

| Pattern | Description |
|---------|-------------|
| **Broadcast** | Send the same prompt to every agent and collect all responses. |
| **Scatter-Gather** | Distribute different prompts across agents round-robin and gather results in input order. |
| **Tree-Reduce** | Hierarchical map-reduce — leaf agents produce answers, supervisors recursively summarize groups. |
| **Blackboard** | Agents collaborate through a shared mutable workspace in iterative rounds until convergence. |
| **Pipeline** | Multi-stage DAG where the output of one stage feeds the next. |

### Aggregators

Built-in aggregation helpers for collected responses:

- `majority_vote` — categorical consensus with confidence score
- `concat` — join response texts
- `best_of` / `top_k` — quality-ranked selection via a custom score function
- `structured_merge` — parse JSON responses and merge into a flat list
- `statistics` — numeric summary (mean, std, median, min, max)
- `failure_report` — diagnostic summary of successes and failures

---

## Example: Broadcast and Reduce

```python
import asyncio
from aurora_swarm import AgentPool, parse_hostfile
from aurora_swarm.patterns.broadcast import broadcast_and_reduce

async def main():
    endpoints = parse_hostfile("agents.hostfile")
    async with AgentPool(endpoints) as pool:
        result = await broadcast_and_reduce(
            pool,
            prompt="Propose a hypothesis for why X happens.",
            reduce_prompt="Summarize these hypotheses:\n{responses}",
        )
        print(result.text)

asyncio.run(main())
```

---

## Configuration

### Context Length Management

Aurora Swarm intelligently manages `max_tokens` for different types of operations:

**Dynamic Sizing** — Automatically adjusts `max_tokens` based on prompt length to prevent truncation:
- Estimates prompt tokens (using a chars÷4 heuristic)
- Queries vLLM's `/v1/models` endpoint to get the model's max context length
- Computes: `max_tokens = min(cap, model_max - prompt_tokens - buffer)`

**Aggregation Presets** — Uses larger token budgets for reduce/aggregation steps where prompts grow:
- `max_tokens` (default): for simple broadcasts and leaf prompts
- `max_tokens_aggregation`: for reduce steps (defaults to 2× `max_tokens`)

### Environment Variables

Configure context length via environment variables:

```bash
export AURORA_SWARM_MAX_TOKENS=1024              # Default max tokens
export AURORA_SWARM_MAX_TOKENS_AGGREGATION=2048  # Aggregation/reduce steps
export AURORA_SWARM_MODEL_MAX_CONTEXT=131072     # Model's max context (optional)
```

### VLLMPool Parameters

```python
from aurora_swarm import VLLMPool, AgentEndpoint

pool = VLLMPool(
    endpoints=[AgentEndpoint("host1", 8000), AgentEndpoint("host2", 8000)],
    model="openai/gpt-oss-120b",
    max_tokens=1024,                    # Default for simple prompts
    max_tokens_aggregation=2048,        # For aggregation steps
    model_max_context=131072,           # Optional: skip API query
    buffer=512,                         # Safety margin for reasoning overhead
    concurrency=512,
    connector_limit=1024,
    timeout=300.0,
)
```

**Per-request override:**

```python
# Explicitly set max_tokens for a specific call
response = await pool.post(agent_index=0, prompt="...", max_tokens=512)
```

---

## Environment Summary

| Item | Value |
|------|-------|
| Python | >= 3.11 |
| Core dependency | `aiohttp >= 3.9` |
| Dev dependencies | `pytest >= 8.0`, `pytest-asyncio >= 0.23` |
| Docs (optional) | `pip install -e ".[docs]"` — Sphinx API reference |
| Build backend | `setuptools >= 68.0` |
