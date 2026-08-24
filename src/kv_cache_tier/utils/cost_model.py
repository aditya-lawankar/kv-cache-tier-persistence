"""
Economic model for KV cache tier persistence.

Translates cache hits into GPU-seconds saved and dollars, priced at a real
transformer architecture and a real accelerator (see utils/hardware.py).

WHY THIS IS PARAMETERIZED BY ARCHITECTURE
-----------------------------------------
A hit is worth the prefill it avoids MINUS the restore it costs. Both sides
depend on the model's geometry, and they depend on it differently:

    prefill  ~  2*P*N + 4*L*H_q*d*N^2     (FLOPs; superlinear in N)
    restore  ~  latency + 2*L*H_kv*d*b*N / bandwidth   (bytes; linear in N)

so whether a hit is worth anything at all is architecture-dependent, and
crosses over at the break-even length N* derived in the paper. Pricing both
sides from ONE ModelSpec is what keeps the value accounting consistent with
that break-even analysis: a tier whose restore exceeds the prefill it saves
now scores zero here, instead of being credited as a win.

An earlier version of this class used two hardcoded constants
(N/15 seconds plus 5e-5 * N^2) with no architecture in them. Those were not
merely mis-scaled: they placed the linear/quadratic crossover at N~1,333
tokens, where real architectures cross between 12k and 53k, making an
8k-token session look 82.6x more valuable than a 512-token one when the
true ratio is 18-26x. That differential -- not a constant factor -- is why
the correction changes results rather than just units.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .hardware import (
    ModelSpec, GPUSpec, StorageSpec,
    TINYLLAMA_1_1B, A100, NVME, OBJECT_STORE,
)


@dataclass
class CostModel:
    """Prices cache hits in GPU-seconds and dollars for a given architecture."""

    model_spec: ModelSpec = TINYLLAMA_1_1B
    gpu: GPUSpec = A100
    gpu_cost_per_hour: float = 2.50         # A100 80GB on-demand, rough $/hr

    # Tier restore characteristics. The hot tier is VRAM-resident, so a hit
    # there costs a pointer swap rather than a transfer.
    hot_restore_latency_s: float = 0.0001
    warm: StorageSpec = NVME
    cold: StorageSpec = OBJECT_STORE

    # ── Prefill (the cost a hit avoids) ────────────────────────────────

    def compute_recompute_time(self, token_count: int) -> float:
        """Seconds of GPU time to prefill `token_count` tokens from scratch."""
        if token_count <= 0:
            return 0.0
        return self.model_spec.prefill_flops(token_count) / self.gpu.effective_flops

    # ── Restore (the cost a hit incurs) ────────────────────────────────

    def kv_bytes(self, token_count: int) -> float:
        """KV cache bytes for `token_count` tokens at the priced architecture.

        Note this is NOT the simulator's serialized size: the simulation runs
        a downscaled geometry as a scale model. Value and restore must both
        be priced at the target architecture or the accounting is incoherent.
        """
        return self.model_spec.kv_bytes_per_token() * max(token_count, 0)

    def restore_time_seconds(self, tier: str, size_bytes: Optional[float] = None,
                             token_count: Optional[int] = None) -> float:
        """Modeled time to restore an entry from `tier`.

        Pass `token_count` (preferred) to price bytes at the target
        architecture; `size_bytes` is accepted for callers that already hold
        an architecture-consistent size.
        """
        if size_bytes is None:
            if token_count is None:
                raise ValueError("restore_time_seconds needs token_count or size_bytes")
            size_bytes = self.kv_bytes(token_count)

        if tier == "hot":
            return self.hot_restore_latency_s
        if tier == "warm":
            return self.warm.transfer_time_s(size_bytes)
        if tier == "cold":
            return self.cold.transfer_time_s(size_bytes)
        raise ValueError(f"Unknown tier: {tier}")

    def savings_per_hit_seconds(self, cached_token_count: int, tier: str,
                                size_bytes: Optional[float] = None) -> float:
        """
        GPU seconds saved by one cache hit: the avoided prefill of the CACHED
        context, minus the modeled restore from the tier where it was found.

        Floored at zero: a hit whose restore costs more than the prefill it
        avoids is worthless, not a liability -- the serving layer would simply
        recompute. This floor is load-bearing for architectures whose N*
        exceeds the workload's context lengths (e.g. MHA models from object
        storage), where persistence is value-negative by construction.
        """
        saved = self.compute_recompute_time(cached_token_count)
        saved -= self.restore_time_seconds(tier, size_bytes=size_bytes,
                                           token_count=cached_token_count)
        return max(saved, 0.0)

    def breakeven_tokens(self, tier: str, max_tokens: int = 300_000) -> Optional[int]:
        """Smallest N whose prefill exceeds its restore from `tier` (else None)."""
        lo, hi = 1, max_tokens
        if self.savings_per_hit_seconds(hi, tier) <= 0.0:
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            if self.savings_per_hit_seconds(mid, tier) > 0.0:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ── Aggregate reporting ────────────────────────────────────────────

    def gpu_seconds_to_usd_per_day(self, gpu_seconds: float, duration_days: float) -> Dict[str, float]:
        """Scale observed GPU-seconds saved over a window to per-day figures."""
        gpu_hours_per_day = (gpu_seconds / max(duration_days, 1e-9)) / 3600.0
        return {
            "gpu_hours_saved_per_day": round(gpu_hours_per_day, 4),
            "cost_saved_per_day_usd": round(gpu_hours_per_day * self.gpu_cost_per_hour, 2),
        }


if __name__ == "__main__":
    from .hardware import MODELS
    print(f"Prefill / restore economics ({A100.name}, MFU {A100.mfu})\n")
    for spec in MODELS:
        cm = CostModel(model_spec=spec)
        n_warm = cm.breakeven_tokens("warm")
        n_cold = cm.breakeven_tokens("cold")
        print(f"  {spec.name:<16} {spec.kv_bytes_per_token()/1024:>6.0f} KB/token  "
              f"N*(NVMe)={n_warm if n_warm else 'never':>7}  "
              f"N*(S3)={n_cold if n_cold else 'never':>7}")
        for n in (598, 2048, 8192):
            print(f"      {n:>5} tok: prefill={cm.compute_recompute_time(n)*1000:>8.1f}ms  "
                  f"warm hit=+{cm.savings_per_hit_seconds(n,'warm')*1000:>8.1f}ms  "
                  f"cold hit=+{cm.savings_per_hit_seconds(n,'cold')*1000:>8.1f}ms")
