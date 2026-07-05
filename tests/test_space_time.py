"""Tests for the V3 Space-Time Density eviction policy."""

import pytest

from kv_cache_tier.eviction.space_time import SpaceTimeDensityPolicy
from kv_cache_tier.utils.clock import SimulatedClock
from kv_cache_tier.core.cache_metadata import CacheEntry


def _entry(session_id, token_count, size_bytes, created_at=0.0, last_accessed=0.0):
    return CacheEntry(
        session_id=session_id,
        user_id=f"user_{session_id}",
        model_config_hash="x",
        created_at=created_at,
        last_accessed=last_accessed,
        token_count=token_count,
        num_blocks=1,
        tier="hot",
        size_bytes=size_bytes,
        access_count=1,
    )


class TestSpaceTimeDensity:
    def test_penalizes_long_expected_idle(self):
        """Two identical entries; the one whose observed rhythm is slower
        (larger inter-access gaps) must score lower and be evicted first."""
        clock = SimulatedClock()
        policy = SpaceTimeDensityPolicy(clock=clock)

        fast = _entry("fast", token_count=1000, size_bytes=1_000_000)
        slow = _entry("slow", token_count=1000, size_bytes=1_000_000)
        policy.on_insert("fast", fast)
        policy.on_insert("slow", slow)

        # fast session returns every ~2 min, slow every ~3 hours
        clock.set(120);        policy.on_access("fast", fast); fast.last_accessed = 120
        clock.set(3 * 3600);   policy.on_access("slow", slow); slow.last_accessed = 3 * 3600
        clock.set(3 * 3600 + 60)

        victim = policy.select_victim({"fast": fast, "slow": slow})
        assert victim == "slow"

    def test_quadratic_value_beats_size_at_equal_rhythm(self):
        """With equal access rhythm, the entry with quadratically larger
        recompute value per byte must be retained."""
        clock = SimulatedClock()
        policy = SpaceTimeDensityPolicy(clock=clock)

        # size grows linearly with tokens, recompute quadratically:
        # big has 16x the tokens -> 16x the size but ~256x the recompute
        small = _entry("small", token_count=512, size_bytes=1_000_000)
        big = _entry("big", token_count=8192, size_bytes=16_000_000)
        policy.on_insert("small", small)
        policy.on_insert("big", big)
        clock.set(600)

        victim = policy.select_victim({"small": small, "big": big})
        assert victim == "small"

    def test_idle_time_overrides_stale_ema(self):
        """An entry idle far beyond its typical gap must see its expected
        wait grow (inspection paradox), not stay anchored to the EMA."""
        clock = SimulatedClock()
        policy = SpaceTimeDensityPolicy(clock=clock)
        e = _entry("s", token_count=1000, size_bytes=1_000_000)
        policy.on_insert("s", e)

        clock.set(60); policy.on_access("s", e); e.last_accessed = 60  # 1-min rhythm
        assert policy.expected_gap_seconds("s", e) == pytest.approx(60, rel=0.1)

        clock.set(60 + 8 * 3600)  # idle 8 hours despite 1-min rhythm
        assert policy.expected_gap_seconds("s", e) == pytest.approx(8 * 3600, rel=0.01)

    def test_batch_and_single_scores_agree_without_predictor(self):
        clock = SimulatedClock()
        policy = SpaceTimeDensityPolicy(clock=clock)
        entries = {
            f"s{i}": _entry(f"s{i}", token_count=500 * (i + 1), size_bytes=10**6 * (i + 1))
            for i in range(4)
        }
        for sid, e in entries.items():
            policy.on_insert(sid, e)
        clock.set(1800)

        batch = policy.get_scores(entries)
        singles = {sid: policy.space_time_density(sid, e) for sid, e in entries.items()}
        for sid in entries:
            assert batch[sid] == pytest.approx(singles[sid])

    def test_factory_registration(self):
        from kv_cache_tier.eviction import create_eviction_policy
        p = create_eviction_policy("space_time", prior_gap_seconds=600.0)
        assert isinstance(p, SpaceTimeDensityPolicy)
        assert p.prior_gap_seconds == 600.0
