import pytest
import time
from kv_cache_tier.eviction import LRUEvictionPolicy, TTLEvictionPolicy, PredictiveEvictionPolicy
from kv_cache_tier.core.cache_metadata import CacheEntry

def dummy_entry(sid, last_access=None, count=1, tokens=10):
    return CacheEntry(
        session_id=sid,
        user_id="u1",
        model_config_hash="h",
        created_at=time.time(),
        last_accessed=last_access or time.time(),
        token_count=tokens,
        num_blocks=1,
        tier="hot",
        size_bytes=100,
        access_count=count
    )

def test_lru_ordering():
    lru = LRUEvictionPolicy()
    entries = {"s1": dummy_entry("s1"), "s2": dummy_entry("s2")}
    lru.on_insert("s1", entries["s1"])
    lru.on_insert("s2", entries["s2"])
    
    # s1 was inserted first, so it should be evicted first
    assert lru.select_victim(entries) == "s1"
    
def test_lru_access_updates_order():
    lru = LRUEvictionPolicy()
    entries = {"s1": dummy_entry("s1"), "s2": dummy_entry("s2")}
    lru.on_insert("s1", entries["s1"])
    lru.on_insert("s2", entries["s2"])
    
    # Access s1, now s2 is least recently used
    lru.on_access("s1", entries["s1"])
    assert lru.select_victim(entries) == "s2"

def test_ttl_expired_entry():
    ttl = TTLEvictionPolicy({"hot": 10})
    now = time.time()
    
    entries = {
        "s1": dummy_entry("s1", last_access=now),
        "s2": dummy_entry("s2", last_access=now - 20)  # Expired (20 > 10)
    }
    
    assert ttl.should_evict("s2", entries["s2"])
    assert not ttl.should_evict("s1", entries["s1"])
    assert ttl.select_victim(entries) == "s2"

def test_predictive_scoring():
    pred = PredictiveEvictionPolicy()
    now = time.time()
    
    entries = {
        # High freq, recent
        "s1": dummy_entry("s1", last_access=now, count=10, tokens=100),
        # Low freq, old
        "s2": dummy_entry("s2", last_access=now - 3600, count=1, tokens=10)
    }
    
    scores = pred.get_scores(entries)
    assert scores["s1"] > scores["s2"]
    
    # Select victim should pick the lowest score
    assert pred.select_victim(entries) == "s2"
