"""
Model, GPU, and storage specifications shared by the cost model and the
break-even analysis.

These live in one place deliberately. An earlier version of this project
carried two independent notions of "what does a prefill cost": the
break-even analysis (benchmarks/breakeven_analysis.py) computed it from
architecture and hardware, while the simulator's CostModel used two
hardcoded constants (N/15 seconds plus 5e-5*N^2). The constants were not a
rescaling of the physics -- they placed the linear/quadratic crossover at
N~1,333 tokens where real architectures cross between 12k and 53k, which
over-rewarded large sessions by 3-4.5x relative to any real model and
inflated exactly the effect the evaluation was measuring. Both consumers
now import from here so the two can no longer drift.

Prefill FLOPs decompose into two terms:

    weights:   2 * P * N            (every parameter touched once per token)
    attention: 4 * L * H_q * d * N^2   (QK^T scores plus the AV product)

Attention FLOPs scale with QUERY heads even under grouped-query attention:
GQA shrinks the KV cache, not the score computation. The KV cache itself
scales with KV heads:

    bytes:     2 * L * H_kv * d * b * N
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Transformer geometry sufficient to price prefill and KV bytes."""
    name: str
    params_b: float          # billions of parameters
    layers: int
    q_heads: int             # attention (query) heads -- drive attention FLOPs
    kv_heads: int            # KV heads (GQA) -- drive cache size
    head_dim: int

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * bytes_per_elem

    def prefill_flops(self, n_tokens: float) -> float:
        n = float(n_tokens)
        weight_flops = 2.0 * self.params_b * 1e9 * n
        attn_flops = 4.0 * self.layers * self.q_heads * self.head_dim * n * n
        return weight_flops + attn_flops


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_flops: float        # dense FP16/BF16 FLOP/s
    mfu: float               # achievable model FLOPs utilization during prefill

    @property
    def effective_flops(self) -> float:
        return self.peak_flops * self.mfu


@dataclass(frozen=True)
class StorageSpec:
    name: str
    latency_s: float
    bandwidth_bytes_s: float

    def transfer_time_s(self, size_bytes: float) -> float:
        return self.latency_s + size_bytes / self.bandwidth_bytes_s


# ── Registries ────────────────────────────────────────────────────────

TINYLLAMA_1_1B = ModelSpec("TinyLlama-1.1B", 1.1, layers=22, q_heads=32, kv_heads=4,  head_dim=64)
LLAMA_2_7B     = ModelSpec("Llama-2-7B",     7.0, layers=32, q_heads=32, kv_heads=32, head_dim=128)
LLAMA_2_70B    = ModelSpec("Llama-2-70B",   70.0, layers=80, q_heads=64, kv_heads=8,  head_dim=128)

MODELS = [TINYLLAMA_1_1B, LLAMA_2_7B, LLAMA_2_70B]
MODELS_BY_KEY = {
    "tinyllama": TINYLLAMA_1_1B,
    "llama7b": LLAMA_2_7B,
    "llama70b": LLAMA_2_70B,
}

A100 = GPUSpec("A100 80GB", 312e12, mfu=0.45)
T4 = GPUSpec("Tesla T4", 65e12, mfu=0.30)

NVME = StorageSpec("NVMe (warm)", 0.010, 2e9)
OBJECT_STORE = StorageSpec("S3/MinIO (cold)", 0.100, 500e6)

# The simulator serializes a downscaled KV geometry (2 layers, 2 heads,
# head_dim 32, FP16) so that traces fit in memory. It is a SCALE MODEL:
# capacity dynamics are driven by these bytes, while value and restore are
# priced at a real architecture's economics via ModelSpec. The ratio below
# is what makes the two views equivalent -- a 500 MB cache at the simulated
# geometry holds the same NUMBER of sessions as a
# (500 MB * scale_factor) cache at the target architecture.
SIM_GEOMETRY_BYTES_PER_TOKEN = 2 * 2 * 2 * 32 * 2  # = 512 bytes/token


def sim_capacity_equivalent_bytes(spec: ModelSpec, sim_capacity_bytes: float) -> float:
    """Capacity at `spec` geometry that holds as many sessions as the sim's."""
    return sim_capacity_bytes * (
        spec.kv_bytes_per_token() / SIM_GEOMETRY_BYTES_PER_TOKEN
    )
