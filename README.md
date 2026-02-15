# Aurora Swarm

Communication patterns for orchestrating large-scale LLM agent swarms on Aurora.

Aurora Swarm provides an async Python library for coordinating thousands of LLM agent endpoints using common distributed communication patterns — broadcast, scatter-gather, tree-reduce, blackboard, and multi-stage pipelines. It manages pooled HTTP connections with semaphore-based concurrency control so you can safely drive 1,000–4,000+ agents from a single orchestrator process.

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

```bash
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
│   └── test_pipeline.py
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

## Environment Summary

| Item | Value |
|------|-------|
| Python | >= 3.11 |
| Core dependency | `aiohttp >= 3.9` |
| Dev dependencies | `pytest >= 8.0`, `pytest-asyncio >= 0.23` |
| Build backend | `setuptools >= 68.0` |
