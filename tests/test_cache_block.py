import pytest
import numpy as np
from kv_cache_tier.core.cache_block import CacheBlockTable
from kv_cache_tier.utils.tensor_utils import kv_caches_equal

def test_cache_block_creation(model_config, sample_kv_data):
    table = CacheBlockTable.from_kv_dict(sample_kv_data, model_config.block_size)
    assert len(table.blocks) == 2  # 32 tokens / 16 block_size
    assert table.total_tokens == 32
    assert table.validate(model_config)

def test_block_table_from_kv_dict(model_config, sample_kv_data):
    table = CacheBlockTable.from_kv_dict(sample_kv_data, model_config.block_size)
    reconstructed = table.to_kv_dict()
    assert kv_caches_equal(sample_kv_data, reconstructed)
