"""Tests for the architecture-parameterized cost model.

These exist because the previous cost model had NO test coverage and shipped
two hardcoded constants whose linear/quadratic balance did not correspond to
any real transformer. The invariants below would have caught that.
"""

import pytest

from src.kv_cache_tier.utils.cost_model import CostModel
from src.kv_cache_tier.utils.hardware import (
    TINYLLAMA_1_1B, LLAMA_2_7B, LLAMA_2_70B, A100, T4,
)


def test_kv_bytes_per_token_matches_published_geometry():
    # 2 (K,V) * layers * kv_heads * head_dim * 2 bytes (FP16)
    assert TINYLLAMA_1_1B.kv_bytes_per_token() == 2 * 22 * 4 * 64 * 2      # 22 KB
    assert LLAMA_2_7B.kv_bytes_per_token() == 2 * 32 * 32 * 128 * 2        # 512 KB
    assert LLAMA_2_70B.kv_bytes_per_token() == 2 * 80 * 8 * 128 * 2        # 320 KB


def test_prefill_has_both_linear_and_quadratic_terms():
    """The bug being guarded: a model with only one term, or the wrong balance.

    Doubling N must more than double prefill time (quadratic term present) but
    less than quadruple it at short contexts (linear term still material).
    """
    cm = CostModel(model_spec=TINYLLAMA_1_1B, gpu=A100)
    t1, t2 = cm.compute_recompute_time(1000), cm.compute_recompute_time(2000)
    assert t2 > 2 * t1
    assert t2 < 4 * t1


def test_linear_quadratic_crossover_is_architecture_correct():
    """Quadratic overtakes linear at N = 2P / (4*L*H_q*d), not at a constant."""
    for spec, expected in [(TINYLLAMA_1_1B, 12207), (LLAMA_2_7B, 26703),
                           (LLAMA_2_70B, 53406)]:
        n = 2 * spec.params_b * 1e9 / (4 * spec.layers * spec.q_heads * spec.head_dim)
        assert n == pytest.approx(expected, rel=0.01)


def test_breakeven_reproduces_published_table():
    """N* here must equal benchmarks/breakeven_analysis.py -- same physics."""
    for spec, n_warm, n_cold in [
        (TINYLLAMA_1_1B, 1561, 25901),
        (LLAMA_2_7B, 43559, 254203),
        (LLAMA_2_70B, 12, 289),
    ]:
        cm = CostModel(model_spec=spec, gpu=A100)
        assert cm.breakeven_tokens("warm") == n_warm
        assert cm.breakeven_tokens("cold") == n_cold


def test_hit_below_breakeven_is_worthless_not_negative():
    cm = CostModel(model_spec=TINYLLAMA_1_1B, gpu=A100)
    assert cm.savings_per_hit_seconds(500, "warm") == 0.0      # below N*=1561
    assert cm.savings_per_hit_seconds(4000, "warm") > 0.0      # above it


def test_tier_discounting_orders_hot_warm_cold():
    cm = CostModel(model_spec=LLAMA_2_70B, gpu=A100)  # profitable at every tier
    hot = cm.savings_per_hit_seconds(2048, "hot")
    warm = cm.savings_per_hit_seconds(2048, "warm")
    cold = cm.savings_per_hit_seconds(2048, "cold")
    assert hot > warm > cold > 0


def test_mha_7b_persistence_is_value_negative_in_workload_range():
    """Consistency with the break-even model: 7B MHA never clears N* below 8k."""
    cm = CostModel(model_spec=LLAMA_2_7B, gpu=A100)
    for n in (598, 2048, 8192):
        assert cm.savings_per_hit_seconds(n, "warm") == 0.0
        assert cm.savings_per_hit_seconds(n, "cold") == 0.0


def test_slower_gpu_raises_prefill_cost():
    n = 2048
    fast = CostModel(model_spec=TINYLLAMA_1_1B, gpu=A100).compute_recompute_time(n)
    slow = CostModel(model_spec=TINYLLAMA_1_1B, gpu=T4).compute_recompute_time(n)
    assert slow > fast


def test_predicts_measured_t4_prefill_within_2x():
    """Validation against §10 hardware: 598 tokens measured ~37ms on a T4.

    The superseded constant-based model predicted 57,747ms for this -- off by
    ~1,500x. Anything within 2x is a genuine calibration.
    """
    cm = CostModel(model_spec=TINYLLAMA_1_1B, gpu=T4)
    predicted_ms = cm.compute_recompute_time(598) * 1000
    assert 18.5 <= predicted_ms <= 74.0


def test_zero_and_negative_tokens_are_free():
    cm = CostModel(model_spec=TINYLLAMA_1_1B)
    assert cm.compute_recompute_time(0) == 0.0
    assert cm.compute_recompute_time(-5) == 0.0
