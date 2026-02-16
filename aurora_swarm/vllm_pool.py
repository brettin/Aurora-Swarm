"""VLLMPool — AgentPool subclass for vLLM OpenAI-compatible endpoints.

vLLM exposes an OpenAI-compatible chat completions API at
``/v1/chat/completions``.  This pool overrides :meth:`post` to speak
that protocol instead of the simpler ``/generate`` endpoint used by
the base :class:`AgentPool`.
"""

from __future__ import annotations

import os
import aiohttp

from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.pool import AgentPool, Response


class VLLMPool(AgentPool):
    """Agent pool that communicates via vLLM's OpenAI-compatible API.

    Parameters
    ----------
    endpoints:
        Agent endpoints (host + port where vLLM is listening).
    model:
        Model identifier passed in the ``"model"`` field of every
        request (e.g. ``"openai/gpt-oss-120b"``).
    max_tokens:
        Maximum tokens to generate per request (default context).
        Can be overridden via ``AURORA_SWARM_MAX_TOKENS`` env var.
    max_tokens_aggregation:
        Maximum tokens for aggregation/reduce steps (larger prompts).
        Can be overridden via ``AURORA_SWARM_MAX_TOKENS_AGGREGATION`` env var.
        Defaults to 2 * max_tokens if not specified.
    model_max_context:
        Model's maximum context length. If None, will be fetched from
        vLLM's ``/v1/models`` endpoint on first request. Can be overridden
        via ``AURORA_SWARM_MODEL_MAX_CONTEXT`` env var.
    buffer:
        Safety margin (in tokens) for dynamic sizing to account for
        reasoning overhead. Defaults to 512.
    concurrency:
        Maximum number of in-flight requests.
    connector_limit:
        Maximum TCP connections in the aiohttp pool.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoints: list[AgentEndpoint],
        model: str = "openai/gpt-oss-120b",
        max_tokens: int | None = None,
        max_tokens_aggregation: int | None = None,
        model_max_context: int | None = None,
        buffer: int = 512,
        concurrency: int = 512,
        connector_limit: int = 1024,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            endpoints,
            concurrency=concurrency,
            connector_limit=connector_limit,
            timeout=timeout,
        )
        self._model = model
        
        # Load from environment with fallbacks
        self._max_tokens = (
            max_tokens
            or int(os.environ.get("AURORA_SWARM_MAX_TOKENS", "512"))
        )
        self._max_tokens_aggregation = (
            max_tokens_aggregation
            or int(os.environ.get("AURORA_SWARM_MAX_TOKENS_AGGREGATION", str(self._max_tokens * 2)))
        )
        self._model_max_context = (
            model_max_context
            or (int(os.environ["AURORA_SWARM_MODEL_MAX_CONTEXT"]) if "AURORA_SWARM_MODEL_MAX_CONTEXT" in os.environ else None)
        )
        self._buffer = buffer
        self._model_max_context_cached: int | None = None

    # -- model metadata -------------------------------------------------------

    async def _get_model_max_context(self) -> int:
        """Fetch the model's max context length from vLLM /v1/models endpoint.
        
        Cached after first call. Returns a sensible default if fetch fails.
        """
        # Return cached value if available
        if self._model_max_context_cached is not None:
            return self._model_max_context_cached
        
        # Return explicitly configured value
        if self._model_max_context is not None:
            self._model_max_context_cached = self._model_max_context
            return self._model_max_context
        
        # Fetch from vLLM API
        try:
            ep = self._endpoints[0]
            session = await self._get_session()
            async with session.get(
                f"{ep.url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                data = await resp.json()
                # Find our model in the list
                for model_info in data.get("data", []):
                    if model_info.get("id") == self._model:
                        max_len = model_info.get("max_model_len")
                        if max_len:
                            self._model_max_context_cached = max_len
                            return max_len
        except Exception:
            pass  # Fall back to default
        
        # Default fallback (131072 is common for many models)
        self._model_max_context_cached = 131072
        return self._model_max_context_cached

    # -- core request (OpenAI chat completions) ------------------------------

    async def post(self, agent_index: int, prompt: str, max_tokens: int | None = None) -> Response:
        """Send *prompt* via the OpenAI chat-completions API on the agent.

        The prompt is wrapped as a single ``user`` message.

        Parameters
        ----------
        agent_index:
            Index of the agent to send the prompt to.
        prompt:
            The prompt text.
        max_tokens:
            Optional override for max tokens. If None, uses dynamic sizing
            based on prompt length and model context limit.
        """
        ep = self._endpoints[agent_index]
        session = await self._get_session()
        
        # Compute max_tokens dynamically if not explicitly provided
        if max_tokens is None:
            # Get model's max context length
            model_max = await self._get_model_max_context()
            
            # Estimate prompt tokens (rough heuristic: 1 token ≈ 4 chars)
            prompt_est = len(prompt) // 4
            
            # Dynamic sizing: never exceed model capacity
            # Use default max_tokens as the preferred cap
            tokens = min(
                self._max_tokens,
                max(128, model_max - prompt_est - self._buffer)
            )
        else:
            tokens = max_tokens
        
        async with self._semaphore:
            try:
                async with session.post(
                    f"{ep.url}/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": tokens,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    data = await resp.json()
                    
                    # Check for error response
                    if resp.status != 200:
                        error_msg = data.get("error", {}).get("message", f"HTTP {resp.status}")
                        return Response(
                            success=False,
                            text="",
                            error=f"API error: {error_msg}",
                            agent_index=agent_index,
                        )
                    
                    # Check for expected response structure
                    if "choices" not in data or not data["choices"]:
                        return Response(
                            success=False,
                            text="",
                            error=f"Invalid response structure: {list(data.keys())}",
                            agent_index=agent_index,
                        )
                    
                    message = data["choices"][0]["message"]
                    text = message.get("content") or message.get("reasoning_content") or ""
                    return Response(
                        success=True,
                        text=text,
                        agent_index=agent_index,
                    )
            except Exception as exc:
                return Response(
                    success=False,
                    text="",
                    error=f"{type(exc).__name__}: {str(exc)}",
                    agent_index=agent_index,
                )

    # -- sub-pool override ---------------------------------------------------

    def _sub_pool(self, endpoints: list[AgentEndpoint]) -> "VLLMPool":
        """Create a child VLLMPool sharing concurrency settings."""
        child = VLLMPool.__new__(VLLMPool)
        child._endpoints = endpoints
        child._concurrency = self._concurrency
        child._connector_limit = self._connector_limit
        child._timeout = self._timeout
        child._semaphore = self._semaphore
        child._session = self._session
        child._model = self._model
        child._max_tokens = self._max_tokens
        child._max_tokens_aggregation = self._max_tokens_aggregation
        child._model_max_context = self._model_max_context
        child._buffer = self._buffer
        child._model_max_context_cached = self._model_max_context_cached
        return child
