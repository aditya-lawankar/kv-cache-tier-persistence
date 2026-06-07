import pytest
from kv_cache_tier.tiers import HotTier, WarmTier, ColdTier
from kv_cache_tier.core.cache_metadata import CacheEntry
import time

def dummy_entry(tier="hot"):
    return CacheEntry(
        session_id="dummy",
        user_id="user1",
        model_config_hash="abc",
        created_at=time.time(),
        last_accessed=time.time(),
        token_count=10,
        num_blocks=1,
        tier=tier,
        size_bytes=100
    )

def test_hot_tier_put_get():
    tier = HotTier(capacity_bytes=1024)
    entry = dummy_entry()
    
    tier.put("k1", b"data", entry)
    assert tier.contains("k1")
    assert tier.get("k1") == b"data"
    
def test_hot_tier_capacity_limit():
    tier = HotTier(capacity_bytes=10)
    entry = dummy_entry()
    
    with pytest.raises(MemoryError):
        tier.put("k1", b"way_too_much_data_for_this_tier", entry)

def test_warm_tier_put_get(tmp_path):
    tier = WarmTier(storage_path=str(tmp_path), capacity_bytes=1024)
    entry = dummy_entry("warm")
    
    tier.put("k1", b"data", entry)
    assert tier.contains("k1")
    assert tier.get("k1") == b"data"
    
def test_warm_tier_delete(tmp_path):
    tier = WarmTier(storage_path=str(tmp_path), capacity_bytes=1024)
    tier.put("k1", b"data", dummy_entry())
    assert tier.delete("k1")
    assert not tier.contains("k1")
    
def test_cold_tier_local_backend(tmp_path):
    tier = ColdTier.create("local", storage_path=str(tmp_path))
    tier.put("k1", b"data", dummy_entry("cold"))
    assert tier.get("k1") == b"data"
