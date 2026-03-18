"""Scatter-gather pattern for embedding pools.

Distribute texts across embedding endpoints and gather EmbeddingResponses
in input order.
"""

from __future__ import annotations

from aurora_swarm.embedding_pool import EmbeddingPool, EmbeddingResponse


async def scatter_gather_embeddings(
    embed_pool: EmbeddingPool,
    texts: list[str],
) -> list[EmbeddingResponse]:
    """Send ``texts[i]`` to ``endpoint[i % pool.size]``, gather in input order.

    Parameters
    ----------
    embed_pool:
        Embedding pool (e.g. from parse_hostfile + by_tag).
    texts:
        Texts to embed.

    Returns
    -------
    list[EmbeddingResponse]
        One response per text, in same order as *texts*.
    """
    return await embed_pool.embed_all(texts)
