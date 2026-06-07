import pytest
from kv_cache_tier.config import SystemConfig, ModelConfig, TierConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.core.session import SessionManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache, kv_caches_equal

def test_full_lifecycle(tmp_path):
    """End-to-end integration test simulating a user session."""
    
    config = SystemConfig.default()
    config.model = ModelConfig(num_layers=2, num_heads=2, head_dim=32, block_size=16)
    config.tiers = TierConfig(
        hot_capacity_mb=10,
        warm_capacity_mb=50,
        cold_capacity_mb=0,
        warm_storage_path=str(tmp_path / "warm"),
        cold_storage_path=str(tmp_path / "cold"),
        cold_backend="local"
    )
    
    manager = TieredCacheManager(config)
    
    # Hook for session end
    def on_end(sid, info):
        # Generate dummy KV data to simulate what vLLM would provide
        kv = generate_random_kv_cache(config.model, info.token_count)
        manager.save(sid, info.user_id, kv)
        # Store for verification
        on_end.saved_kv = kv
        
    sessions = SessionManager(on_session_end=on_end)
    
    # 1. Start Session
    sid = sessions.start_session("user_42")
    
    # 2. Add tokens
    sessions.update_session(sid, 64)
    
    # 3. End session (triggers save via callback)
    sessions.end_session(sid)
    
    # 4. Wait/Idle (simulate time passing)
    manager.demote(sid) # Push to warm tier
    
    # 5. Resume session
    new_sid = sessions.start_session("user_42", existing_session_id=sid)
    assert new_sid == sid
    
    loaded_kv = manager.load(sid)
    assert loaded_kv is not None
    assert kv_caches_equal(on_end.saved_kv, loaded_kv)
    
    # Entry should be promoted back to hot
    assert manager.index.get(sid).tier == "hot"
