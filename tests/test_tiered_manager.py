import pytest
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import kv_caches_equal

def test_save_and_load_round_trip(system_config, sample_kv_data):
    manager = TieredCacheManager(system_config)
    session_id = "test_sess_1"
    
    manager.save(session_id, "user_1", sample_kv_data)
    loaded = manager.load(session_id)
    
    assert loaded is not None
    assert kv_caches_equal(sample_kv_data, loaded)
    
def test_cache_miss_returns_none(system_config):
    manager = TieredCacheManager(system_config)
    assert manager.load("nonexistent") is None
    
def test_hot_tier_eviction_to_warm(system_config, model_config, sample_kv_data):
    # We initialized tensors with np.zeros to speed up benchmarks, which compresses extremely well.
    # Disable compression so they actually take up space (130KB each) and exceed 1MB.
    system_config.serialization.compression = "none"
    system_config.tiers.hot_capacity_mb = 1

    manager = TieredCacheManager(system_config)
    
    for i in range(20):
        manager.save(f"sess_{i}", "user_1", sample_kv_data)
        
    stats = manager.stats()
    assert stats["usage"]["warm"].entry_count > 0
    assert stats["usage"]["hot"].entry_count > 0
    
    # We should still be able to load the oldest one (from warm tier)
    loaded = manager.load("sess_0")
    assert loaded is not None
    
def test_demote_and_promote(system_config, sample_kv_data):
    manager = TieredCacheManager(system_config)
    manager.save("sess_1", "user_1", sample_kv_data)
    
    assert manager.index.get("sess_1").tier == "hot"
    
    manager.demote("sess_1")
    assert manager.index.get("sess_1").tier == "warm"
    
    manager.promote("sess_1", "hot")
    assert manager.index.get("sess_1").tier == "hot"

def test_delete(system_config, sample_kv_data):
    manager = TieredCacheManager(system_config)
    manager.save("sess_1", "u1", sample_kv_data)
    manager.delete("sess_1")
    assert manager.load("sess_1") is None
