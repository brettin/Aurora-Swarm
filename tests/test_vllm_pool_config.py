"""Unit tests for VLLMPool configuration and dynamic sizing."""

import os
import pytest
from aurora_swarm.hostfile import AgentEndpoint
from aurora_swarm.vllm_pool import VLLMPool


def test_vllm_pool_default_config():
    """Test VLLMPool with default configuration."""
    endpoints = [AgentEndpoint("host1", 8000)]
    pool = VLLMPool(endpoints)
    
    assert pool._max_tokens == 512
    assert pool._max_tokens_aggregation == 1024  # 2x default
    assert pool._model == "openai/gpt-oss-120b"
    assert pool._buffer == 512


def test_vllm_pool_explicit_config():
    """Test VLLMPool with explicit configuration."""
    endpoints = [AgentEndpoint("host1", 8000)]
    pool = VLLMPool(
        endpoints,
        max_tokens=1024,
        max_tokens_aggregation=2048,
        model_max_context=131072,
        buffer=256,
    )
    
    assert pool._max_tokens == 1024
    assert pool._max_tokens_aggregation == 2048
    assert pool._model_max_context == 131072
    assert pool._buffer == 256


def test_vllm_pool_env_config(monkeypatch):
    """Test VLLMPool with environment variable configuration."""
    monkeypatch.setenv("AURORA_SWARM_MAX_TOKENS", "2048")
    monkeypatch.setenv("AURORA_SWARM_MAX_TOKENS_AGGREGATION", "4096")
    monkeypatch.setenv("AURORA_SWARM_MODEL_MAX_CONTEXT", "200000")
    
    endpoints = [AgentEndpoint("host1", 8000)]
    pool = VLLMPool(endpoints)
    
    assert pool._max_tokens == 2048
    assert pool._max_tokens_aggregation == 4096
    assert pool._model_max_context == 200000


def test_vllm_pool_explicit_overrides_env(monkeypatch):
    """Test that explicit config overrides environment variables."""
    monkeypatch.setenv("AURORA_SWARM_MAX_TOKENS", "2048")
    
    endpoints = [AgentEndpoint("host1", 8000)]
    pool = VLLMPool(endpoints, max_tokens=1024)
    
    assert pool._max_tokens == 1024


def test_vllm_pool_subpool_inherits_config():
    """Test that sub-pools inherit configuration."""
    endpoints = [
        AgentEndpoint("host1", 8000),
        AgentEndpoint("host2", 8000),
        AgentEndpoint("host3", 8000),
    ]
    pool = VLLMPool(
        endpoints,
        max_tokens=1024,
        max_tokens_aggregation=2048,
        buffer=256,
    )
    
    # Create a sub-pool
    sub_pool = pool.select([0, 1])
    
    assert sub_pool._max_tokens == 1024
    assert sub_pool._max_tokens_aggregation == 2048
    assert sub_pool._buffer == 256
    assert len(sub_pool._endpoints) == 2


@pytest.mark.asyncio
async def test_vllm_pool_post_signature():
    """Test that post() accepts max_tokens parameter."""
    endpoints = [AgentEndpoint("host1", 8000)]
    pool = VLLMPool(endpoints)
    
    # Verify the method signature includes max_tokens
    import inspect
    sig = inspect.signature(pool.post)
    assert "max_tokens" in sig.parameters
    assert sig.parameters["max_tokens"].default is None
