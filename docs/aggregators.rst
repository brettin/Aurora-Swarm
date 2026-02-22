Aggregators
===========

Aggregators combine a list of agent :class:`~aurora_swarm.pool.Response` objects (for example from :func:`~aurora_swarm.patterns.broadcast.broadcast`) into a single result. They are useful after broadcast or scatter-gather when you need a single answer or summary from many responses.

Unless you pass ``include_failures=True``, every aggregator silently skips responses with ``success=False``.

Categorical
-----------

:func:`~aurora_swarm.aggregators.majority_vote` — Returns ``(winner, confidence)`` where *confidence* is the fraction of votes for the winner. Response text is stripped and compared case-insensitively.

**When to use:** Short categorical answers (e.g. yes/no, class labels, single-word choices). Use when you want consensus and a confidence score.

Text
----

:func:`~aurora_swarm.aggregators.concat` — Joins all response texts with a configurable *separator* (default newline).

**When to use:** You need one combined string of all answers (e.g. for logging, for feeding into a reduce prompt, or for downstream processing).

Quality selection
-----------------

:func:`~aurora_swarm.aggregators.best_of` — Returns the single :class:`~aurora_swarm.pool.Response` with the highest ``score_fn(response)`` value. You must provide a callable that maps each response to a float (e.g. length, a quality metric).

:func:`~aurora_swarm.aggregators.top_k` — Returns the *k* highest-scoring responses in descending order.

**When to use:** When you have a scoring function (e.g. length, readability, model score) and want the best one or the top *k*.

Structured data
---------------

:func:`~aurora_swarm.aggregators.structured_merge` — Parses each response text as JSON and merges into a flat list. JSON arrays are flattened; other values are appended. Returns ``(merged_list, errors)`` where *errors* is a list of ``{agent_index, error}`` for parse failures.

**When to use:** Agents return JSON (e.g. lists of items); you want one merged list and optional error reporting.

Numeric
-------

:func:`~aurora_swarm.aggregators.statistics` — Computes summary statistics over numeric values. If *extract_fn* is ``None``, response text is converted to float. Returns a dict with keys ``mean``, ``std``, ``median``, ``min``, ``max``.

**When to use:** Responses are numbers (or you have an *extract_fn* to get a number from each response); you want mean, std, median, min, max.

Diagnostics
-----------

:func:`~aurora_swarm.aggregators.failure_report` — Returns a dict with ``total``, ``success_count``, ``failure_count``, and ``failures`` (list of ``{agent_index, error}``). Does not filter by success; use to inspect which agents failed and why.

Usage with broadcast
--------------------

To broadcast the same prompt to a subset of agents (e.g. all except one) and then aggregate in Python:

1. Build a sub-pool with :meth:`~aurora_swarm.pool.AgentPool.select` (e.g. exclude one index).
2. Call :func:`~aurora_swarm.patterns.broadcast.broadcast` with the sub-pool and prompt.
3. Apply one or more aggregators to the returned list of responses.

Example (minimal)::

   from aurora_swarm import VLLMPool, parse_hostfile
   from aurora_swarm.patterns.broadcast import broadcast
   from aurora_swarm.aggregators import majority_vote, concat

   endpoints = parse_hostfile("agents.hostfile")
   async with VLLMPool(endpoints, model="...") as pool:
       exclude_index = 0
       indices = [i for i in range(pool.size) if i != exclude_index]
       sub_pool = pool.select(indices)
       responses = await broadcast(sub_pool, "Answer with one word: yes or no. Is the sky blue?")
   winner, confidence = majority_vote(responses)
   combined = concat(responses, separator=" | ")

A full runnable example that broadcasts to all hosts except one and prints both ``majority_vote`` and ``concat`` results is in the repo: ``examples/broadcast_aggregators.py``. Launch it with ``examples/broadcast_aggregators.sh <hostfile>`` (see the script and docstring for options).
