"""
Break-even analysis: when does KV cache restoration beat prefill recompute?

Restoring a persisted KV cache costs O(N) in bytes moved:
    t_restore(N) = tier_latency + size(N) / bandwidth
    size(N)      = 2 (K,V) x layers x kv_heads x head_dim x bytes_per_elem x N

Recomputing the prefill costs O(N) in weight FLOPs plus O(N^2) attention:
    t_prefill(N) = 2 x P x N / (peak_flops x efficiency)  +  attention(N^2)

Because restore is linear and prefill is superlinear, there is a crossover
context length N* beyond which persistence ALWAYS wins, for every
(model, hardware, storage tier) triple. Below N*, persistence loses on pure
latency and only pays off through GPU-occupancy savings.

This script derives N* analytically across model sizes and storage tiers,
validates the CPU TinyLlama measurement against the model, and produces
figure4_breakeven.png for the paper.

Usage:
    python benchmarks/breakeven_analysis.py
"""

import os
import json
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ──────────────────────────────────────────────────────────────────
# Hardware / model / storage parameterization
# ──────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    params_b: float          # billions of parameters
    layers: int
    q_heads: int             # attention (query) heads — drive attention FLOPs
    kv_heads: int            # KV heads (GQA) — drive cache size
    head_dim: int

    def kv_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * bytes_per_elem


MODELS = [
    ModelSpec("TinyLlama-1.1B", 1.1, layers=22, q_heads=32, kv_heads=4,  head_dim=64),
    ModelSpec("Llama-2-7B",     7.0, layers=32, q_heads=32, kv_heads=32, head_dim=128),
    ModelSpec("Llama-2-70B",   70.0, layers=80, q_heads=64, kv_heads=8,  head_dim=128),
]

@dataclass
class GPUSpec:
    name: str
    peak_flops: float        # dense FP16/BF16 FLOP/s
    mfu: float               # achievable model FLOPs utilization during prefill


A100 = GPUSpec("A100 80GB", 312e12, mfu=0.45)

@dataclass
class StorageSpec:
    name: str
    latency_s: float
    bandwidth_bytes_s: float


STORAGE_TIERS = [
    StorageSpec("NVMe (warm)",     0.010, 2e9),
    StorageSpec("S3/MinIO (cold)", 0.100, 500e6),
]


# ──────────────────────────────────────────────────────────────────
# Cost functions
# ──────────────────────────────────────────────────────────────────

def prefill_time_s(model: ModelSpec, n_tokens: np.ndarray, gpu: GPUSpec) -> np.ndarray:
    """Prefill wall time: 2*P FLOPs per token for weights, plus the quadratic
    attention term 4 * layers * q_heads * head_dim * N^2 (QK^T scores + AV).
    Attention FLOPs scale with QUERY heads even under GQA — grouping shrinks
    the cache, not the score computation."""
    weight_flops = 2.0 * model.params_b * 1e9 * n_tokens
    attn_flops = 4.0 * model.layers * (model.q_heads * model.head_dim) * n_tokens.astype(np.float64) ** 2
    return (weight_flops + attn_flops) / (gpu.peak_flops * gpu.mfu)


def restore_time_s(model: ModelSpec, n_tokens: np.ndarray, storage: StorageSpec) -> np.ndarray:
    size = model.kv_bytes_per_token() * n_tokens.astype(np.float64)
    return storage.latency_s + size / storage.bandwidth_bytes_s


def crossover_tokens(model: ModelSpec, storage: StorageSpec, gpu: GPUSpec) -> float:
    """Smallest N where prefill_time > restore_time (persistence wins)."""
    n = np.arange(1, 300_000, dtype=np.float64)
    diff = prefill_time_s(model, n, gpu) - restore_time_s(model, n, storage)
    idx = np.argmax(diff > 0)
    return float(n[idx]) if diff[idx] > 0 else float("inf")


# ──────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────

def make_figure(gpu: GPUSpec = A100):
    n = np.logspace(1.5, 5.2, 400)  # ~30 to ~160k tokens
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.2), sharey=True)

    results = {}
    for ax, model in zip(axes, MODELS):
        prefill = prefill_time_s(model, n, gpu) * 1000
        ax.plot(n, prefill, label="Prefill recompute", color="#c0392b", lw=2)

        for storage, ls in zip(STORAGE_TIERS, ["-", "--"]):
            restore = restore_time_s(model, n, storage) * 1000
            ax.plot(n, restore, label=f"Restore: {storage.name}", color="#2c3e50",
                    lw=1.6, linestyle=ls)
            n_star = crossover_tokens(model, storage, gpu)
            results[(model.name, storage.name)] = n_star
            if np.isfinite(n_star):
                ax.axvline(n_star, color="#7f8c8d", lw=0.8, linestyle=":")
                ax.annotate(f"N*={n_star:,.0f}",
                            xy=(n_star, ax.get_ylim()[0]),
                            xytext=(n_star * 1.15, 0.35),
                            fontsize=8, color="#34495e", rotation=90)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Context length N (tokens)")
        ax.set_title(f"{model.name}\n({model.kv_bytes_per_token()/1024:.0f} KB KV/token)")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].set_ylabel(f"Latency (ms) on {gpu.name}, MFU={gpu.mfu:.0%}")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Break-even context length N*: KV restore (O(N)) vs prefill recompute (O(N$^2$))",
                 y=1.02, fontsize=13)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure4_breakeven.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")
    return results


def validate_against_tinyllama(results_path="benchmarks/results/tinyllama_integration_results.json"):
    """Sanity-check the restore-side model against the measured CPU integration:
    measured warm restore (load + deserialize) should be same order as modeled NVMe."""
    if not os.path.exists(results_path):
        print("  [--] No TinyLlama integration results found; skipping validation")
        return
    with open(results_path) as f:
        rows = json.load(f)
    tiny = MODELS[0]
    nvme = STORAGE_TIERS[0]
    print("\n  Validation vs measured TinyLlama integration (CPU):")
    for r in rows:
        n_tok = r["prompt_tokens"]
        modeled_restore_ms = restore_time_s(tiny, np.array([n_tok]), nvme)[0] * 1000
        measured_ms = r["load_latency_ms"]
        print(f"    {n_tok:>5} tokens: modeled NVMe restore={modeled_restore_ms:6.1f}ms | "
              f"measured local load={measured_ms:6.1f}ms")


if __name__ == "__main__":
    print("Break-even analysis (prefill O(N^2) vs restore O(N))\n")
    results = make_figure()
    print("\n  Crossover context lengths N* (persistence wins beyond N*):")
    for (model, storage), n_star in results.items():
        label = f"{n_star:,.0f} tokens" if np.isfinite(n_star) else "never (within 160k)"
        print(f"    {model:<16} from {storage:<16} -> N* = {label}")
    validate_against_tinyllama()
