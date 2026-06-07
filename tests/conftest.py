import pytest
import numpy as np

from kv_cache_tier.config import ModelConfig, SystemConfig, TierConfig
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache

@pytest.fixture
def model_config():
    """Small model config for fast tests."""
    return ModelConfig(
        num_layers=4,
        num_heads=4,
        head_dim=64,
        block_size=16,
        dtype="float16"
    )

@pytest.fixture
def system_config(tmp_path):
    """System config with tiny tiers for testing eviction."""
    cfg = SystemConfig.default()
    cfg.model = ModelConfig(num_layers=4, num_heads=4, head_dim=64, block_size=16)
    
    # Tiny capacities
    cfg.tiers = TierConfig(
        hot_capacity_mb=1,  # 1MB
        warm_capacity_mb=10, # 10MB
        cold_capacity_mb=0,  # Unlimited
        warm_storage_path=str(tmp_path / "warm"),
        cold_storage_path=str(tmp_path / "cold"),
        cold_backend="local"
    )
    return cfg

@pytest.fixture
def sample_kv_data(model_config):
    """Generate 32 tokens of random KV data."""
    return generate_random_kv_cache(model_config, 32)
