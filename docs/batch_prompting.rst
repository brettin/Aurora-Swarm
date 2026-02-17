Batch Prompting
===============

Overview
--------

Batch prompting is a performance optimization that dramatically reduces HTTP request overhead by sending multiple prompts to each agent in a single request. This feature uses the OpenAI Python client's completions API, which supports sending a list of prompts that are processed together.

Performance Impact
------------------

Request Reduction
~~~~~~~~~~~~~~~~~

**Without batching:**
   10,000 prompts with 100 agents = 10,000 HTTP requests (100 per agent)

**With batching:**
   10,000 prompts with 100 agents = 100 HTTP requests (1 per agent)

**Result: 100× reduction in HTTP requests**

Measured Performance
~~~~~~~~~~~~~~~~~~~~

Testing with 20 prompts on 4 vLLM endpoints showed:

- **Batch mode:** 2.76s (7.24 prompts/sec) using 4 HTTP requests
- **Non-batch mode:** 3.09s (6.46 prompts/sec) using 20 HTTP requests
- **Improvement:** 12% faster, 80% fewer requests

At scale (1,000 prompts, 4 agents), batch mode uses only 4 HTTP requests versus 1,000 for non-batch mode.

Usage
-----

Basic Example
~~~~~~~~~~~~~

Batch prompting is enabled by default in :class:`VLLMPool`:

.. code-block:: python

   from aurora_swarm import VLLMPool, parse_hostfile
   from aurora_swarm.patterns.scatter_gather import scatter_gather

   # Load endpoints
   endpoints = parse_hostfile("agents.hostfile")

   # Create pool with batch mode enabled (default)
   async with VLLMPool(
       endpoints,
       model="openai/gpt-oss-120b",
       max_tokens=1024,
       use_batch=True,  # Default, can omit
   ) as pool:
       # Generate many prompts
       prompts = [f"Analyze gene {i}" for i in range(10000)]
       
       # scatter_gather automatically uses batch API
       responses = await scatter_gather(pool, prompts)

Disable Batching
~~~~~~~~~~~~~~~~

For debugging or compatibility, you can disable batching:

.. code-block:: python

   pool = VLLMPool(
       endpoints,
       model="openai/gpt-oss-120b",
       use_batch=False,  # Falls back to individual requests
   )

Manual Batch Control
~~~~~~~~~~~~~~~~~~~~

You can also manually control batching:

.. code-block:: python

   # Send batch to specific agent
   prompts_for_agent_0 = ["prompt1", "prompt2", "prompt3"]
   responses = await pool.post_batch(0, prompts_for_agent_0)

   # Or use send_all_batched directly
   all_prompts = [f"task-{i}" for i in range(100)]
   responses = await pool.send_all_batched(all_prompts, max_tokens=512)

Implementation Details
----------------------

API Endpoints
~~~~~~~~~~~~~

**Batch mode** uses the ``/v1/completions`` endpoint:
   - Accepts ``prompt`` as ``str`` or ``list[str]``
   - Returns one choice per prompt
   - Prompts sent as raw text

**Non-batch mode** uses the ``/v1/chat/completions`` endpoint:
   - Wraps prompts in ``{"role": "user", "content": prompt}``
   - vLLM handles chat template formatting
   - One message per request

Architecture
~~~~~~~~~~~~

When you call ``scatter_gather(pool, prompts)``:

1. ``scatter_gather`` calls ``pool.send_all_batched(prompts)``
2. Prompts are grouped by target agent using round-robin (``i % pool.size``)
3. For each agent, ``post_batch(agent_idx, prompts_for_agent)`` sends one request
4. ``AsyncOpenAI.completions.create(prompt=list_of_prompts)`` processes the batch
5. Responses are reconstructed in original input order

Key Methods
~~~~~~~~~~~

.. py:method:: VLLMPool.post_batch(agent_index, prompts, max_tokens=None)
   :async:

   Send multiple prompts to one agent in a single request.

   :param int agent_index: Index of the agent to send prompts to
   :param list[str] prompts: List of prompts to send in one batch
   :param int max_tokens: Optional override for max tokens
   :return: List of Response objects, one per prompt
   :rtype: list[Response]

.. py:method:: VLLMPool.send_all_batched(prompts, max_tokens=None)
   :async:

   Distribute prompts across all agents with batching.

   Automatically groups prompts by target agent and sends one batched
   request per agent. Returns responses in the same order as input prompts.

   :param list[str] prompts: List of prompts to send
   :param int max_tokens: Optional override for max tokens
   :return: Responses in input order
   :rtype: list[Response]

Pattern Integration
~~~~~~~~~~~~~~~~~~~

The following patterns automatically use batching when available:

- ``scatter_gather()`` - Distributes prompts with batching
- ``map_gather()`` - Uses scatter_gather internally
- ``tree_reduce()`` - Batches both leaf and supervisor prompts

Backward Compatibility
----------------------

The batch prompting feature is fully backward compatible:

✓ All existing tests pass
✓ ``AgentPool.send_all()`` unchanged
✓ Non-VLLMPool usage unchanged  
✓ ``post()`` method unchanged (uses chat completions)
✓ Can disable batching with ``use_batch=False``
✓ Pattern APIs unchanged (batching is transparent)

Chat Template Handling
----------------------

The completions API sends prompts as raw text. For instruction-tuned models that expect specific chat formatting, you may need to format prompts before sending:

.. code-block:: python

   # Example: Add chat template manually
   prompt = "<|user|>\nAnalyze gene ABC123\n<|assistant|>\n"
   prompts = [prompt]  # Send formatted prompt

This ensures the model receives the expected format even when using the completions endpoint.

Troubleshooting
---------------

Connection Errors
~~~~~~~~~~~~~~~~~

If batch requests fail with connection errors, verify your vLLM endpoints support ``/v1/completions``:

.. code-block:: bash

   curl -X POST http://hostname:port/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"model-name","prompt":"test","max_tokens":10}'

Empty Responses
~~~~~~~~~~~~~~~

If responses are empty, the model may need:

- More specific prompts
- Higher ``max_tokens`` value
- Different prompt formatting (e.g., chat template)

.. code-block:: python

   responses = await pool.send_all_batched(prompts, max_tokens=2048)

Timeout Errors
~~~~~~~~~~~~~~

For large batches, increase the timeout:

.. code-block:: python

   pool = VLLMPool(endpoints, timeout=600.0, use_batch=True)

Testing
-------

Unit tests with mock endpoints:

.. code-block:: bash

   pytest tests/test_vllm_pool.py -v

Integration tests with real vLLM endpoints:

.. code-block:: bash

   pytest tests/integration/ --hostfile=/path/to/hostfile -v

Performance comparison:

.. code-block:: bash

   python test_batch_integration.py /path/to/hostfile

See Also
--------

- :doc:`api` - Full API reference including VLLMPool
- :doc:`context_length` - Context length configuration
- ``BATCH_PROMPTING.md`` - Detailed implementation documentation
