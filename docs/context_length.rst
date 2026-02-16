Context length configuration
=============================

Aurora Swarm supports both **config-based** and **dynamic** context length management when using :class:`~aurora_swarm.vllm_pool.VLLMPool`.

Config-based
------------

Set default and aggregation token limits via environment variables or constructor parameters:

**Environment variables:**

- ``AURORA_SWARM_MAX_TOKENS`` — default max tokens (e.g. 512)
- ``AURORA_SWARM_MAX_TOKENS_AGGREGATION`` — for aggregation/reduce steps (e.g. 2048)
- ``AURORA_SWARM_MODEL_MAX_CONTEXT`` — model's max context length (optional; if unset, fetched from vLLM ``/v1/models``)

**Constructor (priority over env):**

.. code-block:: python

   from aurora_swarm import VLLMPool, AgentEndpoint

   pool = VLLMPool(
       endpoints=[AgentEndpoint("host1", 8000)],
       model="openai/gpt-oss-120b",
       max_tokens=1024,
       max_tokens_aggregation=2048,
       model_max_context=131072,  # optional; skip API query
       buffer=512,
   )

**Per-request override:** pass ``max_tokens`` to :meth:`~aurora_swarm.vllm_pool.VLLMPool.post` for a single call.

Dynamic sizing
--------------

When ``max_tokens`` is not passed to :meth:`~aurora_swarm.vllm_pool.VLLMPool.post`, the pool computes a safe value:

- Fetches the model's max context from vLLM ``/v1/models`` (cached)
- Estimates prompt tokens (e.g. ``len(prompt) // 4``)
- Uses ``min(configured_cap, model_max - prompt_est - buffer)`` so responses are not truncated

Aggregation steps in patterns (e.g. :func:`~aurora_swarm.patterns.broadcast.broadcast_and_reduce` reduce step, pipeline stages 2+) use ``max_tokens_aggregation`` when the pool provides it.

Full details, examples, and reasoning-model notes are in the repo: ``CONTEXT_LENGTH.md``.
