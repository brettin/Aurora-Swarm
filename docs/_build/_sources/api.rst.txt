API Reference
=============

Core
----

Package (aurora_swarm)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aurora_swarm
   :members: AgentEndpoint, AgentPool, Response, parse_hostfile
   :undoc-members:
   :show-inheritance:

Hostfile
~~~~~~~~

.. autoclass:: aurora_swarm.hostfile.AgentEndpoint
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: aurora_swarm.hostfile.parse_hostfile

Pool
~~~~

.. autoclass:: aurora_swarm.pool.Response
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aurora_swarm.pool.AgentPool
   :members:
   :undoc-members:
   :show-inheritance:

Aggregators
~~~~~~~~~~~

.. automodule:: aurora_swarm.aggregators
   :members: majority_vote, concat, best_of, top_k, structured_merge, statistics, failure_report
   :undoc-members:
   :show-inheritance:

Communication patterns
----------------------

Broadcast
~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.broadcast.broadcast
.. autofunction:: aurora_swarm.patterns.broadcast.broadcast_and_reduce

Scatter-Gather
~~~~~~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.scatter_gather.scatter_gather
.. autofunction:: aurora_swarm.patterns.scatter_gather.map_gather

Tree-Reduce
~~~~~~~~~~~

.. autofunction:: aurora_swarm.patterns.tree_reduce.tree_reduce

Blackboard
~~~~~~~~~~

.. autoclass:: aurora_swarm.patterns.blackboard.Blackboard
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
~~~~~~~~

.. autoclass:: aurora_swarm.patterns.pipeline.Stage
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: name, prompt_template, n_agents, output_transform, output_filter

.. autofunction:: aurora_swarm.patterns.pipeline.run_pipeline
