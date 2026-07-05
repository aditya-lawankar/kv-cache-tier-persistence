"""Tests for the Clock abstraction and its integration with the manager/policies."""

import pytest

from kv_cache_tier.utils.clock import SimulatedClock, SystemClock
from kv_cache_tier.eviction.features import FeatureExtractor
from kv_cache_tier.core.cache_metadata import CacheEntry


def _entry(session_id="s1", created_at=0.0, last_accessed=0.0, token_count=512):
    return CacheEntry(
        session_id=session_id,
        user_id="u1",
        model_config_hash="x",
        created_at=created_at,
        last_accessed=last_accessed,
        token_count=token_count,
        num_blocks=32,
        tier="hot",
        size_bytes=token_count * 256,
        access_count=1,
    )


class TestSimulatedClock:
    def test_starts_at_zero_and_advances(self):
        clock = SimulatedClock()
        assert clock.now() == 0.0
        clock.advance(90)
        assert clock.now() == 90.0
        clock.set(3600)
        assert clock.now() == 3600.0

    def test_cannot_go_backwards(self):
        clock = SimulatedClock(start=100.0)
        with pytest.raises(ValueError):
            clock.set(50.0)

    def test_time_of_day_convention_matches_training(self):
        """t=0 is Monday 00:00 — same convention as train_predictors.py labels."""
        clock = SimulatedClock()
        assert clock.hour_of_day() == 0
        assert clock.day_of_week() == 0  # Monday
        clock.set(10 * 3600)  # Monday 10:00
        assert clock.hour_of_day() == 10
        assert clock.day_of_week() == 0
        clock.set(3 * 86400 + 23 * 3600)  # Thursday 23:00
        assert clock.hour_of_day() == 23
        assert clock.day_of_week() == 3

    def test_system_clock_sane(self):
        clock = SystemClock()
        assert clock.now() > 0
        assert 0 <= clock.hour_of_day() <= 23
        assert 0 <= clock.day_of_week() <= 6


class TestFeatureExtractorUsesClock:
    def test_temporal_features_follow_simulated_time(self):
        """The core train/serve-skew regression test: session age and idle time
        must reflect SIMULATED time, not the wall-clock speed of a replay."""
        clock = SimulatedClock()
        extractor = FeatureExtractor(clock)

        entry = _entry(created_at=0.0, last_accessed=0.0)
        clock.set(2 * 3600)  # two simulated hours pass (instantly in wall time)

        feats = extractor.build(entry)
        assert feats.session_age_minutes == pytest.approx(120.0)
        assert feats.time_since_last_access_minutes == pytest.approx(120.0)
        assert feats.hour_of_day == 2
        assert feats.day_of_week == 0

    def test_user_return_rate_from_history(self):
        clock = SimulatedClock()
        extractor = FeatureExtractor(clock)
        # one session accessed twice -> 1 unique session, 1 resume
        extractor.record_access("u1", "s1", 512)
        clock.advance(60)
        extractor.record_access("u1", "s1", 600)

        feats = extractor.build(_entry())
        assert feats.user_historical_return_rate == pytest.approx(1.0)


class TestManagerOnSimulatedTime:
    def test_entry_timestamps_use_simulated_clock(self, tmp_path):
        from kv_cache_tier.config import (
            SystemConfig, ModelConfig, TierConfig, EvictionConfig, SerializationConfig,
        )
        from kv_cache_tier.core.tiered_manager import TieredCacheManager
        import numpy as np

        clock = SimulatedClock(start=1000.0)
        config = SystemConfig(
            model=ModelConfig(num_layers=2, num_heads=2, head_dim=8, block_size=16, dtype="float16"),
            tiers=TierConfig(
                hot_capacity_mb=10, warm_capacity_mb=10, cold_capacity_mb=10,
                warm_storage_path=str(tmp_path / "warm"),
                cold_storage_path=str(tmp_path / "cold"),
                cold_backend="local",
            ),
            eviction=EvictionConfig(policy="lru"),
            serialization=SerializationConfig(format="raw_binary", compression="none"),
        )
        manager = TieredCacheManager(config, clock=clock)

        kv = {i: (np.zeros((2, 16, 8), dtype=np.float16),
                  np.zeros((2, 16, 8), dtype=np.float16)) for i in range(2)}
        manager.save("s1", "u1", kv)
        entry = manager.index.get("s1")
        assert entry.created_at == 1000.0

        clock.set(5000.0)
        assert manager.load("s1") is not None
        assert manager.index.get("s1").last_accessed == 5000.0
