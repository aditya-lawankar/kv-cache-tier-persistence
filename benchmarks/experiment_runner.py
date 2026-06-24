"""
Experiment Runner V2: 5+ Policies x 3 Workload Profiles

Simulates the full KV cache lifecycle using WorkloadSimulator traces,
measuring cache hit rates, save/load latencies, and cost savings for
each (policy, workload) combination.

Policies:
  - LRU (baseline)
  - Heuristic (V1 weighted formula)
  - Logistic V1 (binary P(resume) — expected to fail under capacity constraints)
  - Value Density V2 (size-aware: P(resume) × recompute_cost / size_bytes)
  - Value Density V2+AC (V2 with admission control threshold)

The V1 → failure → V2 arc is the research narrative.

Produces a structured results table with 95% confidence intervals
via bootstrap resampling.

Usage:
    python benchmarks/experiment_runner.py
    python benchmarks/experiment_runner.py --duration 0.25 --seed 42
"""

import json
import os
import sys
import time
import tempfile
import shutil
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.config import SystemConfig, ModelConfig, TierConfig, EvictionConfig, SerializationConfig
from src.kv_cache_tier.core.tiered_manager import TieredCacheManager
from src.kv_cache_tier.eviction.predictive import PredictiveEvictionPolicy
from src.kv_cache_tier.eviction.value_density import ValueDensityPolicy
from src.kv_cache_tier.eviction.predictors import ResumePredictor
from src.kv_cache_tier.utils.cost_model import CostModel
from benchmarks.workload_simulator import WorkloadSimulator, PROFILES

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    policy: str
    workload: str
    hit_rate: float
    miss_rate: float
    total_loads: int
    total_hits: int
    total_misses: int
    save_latency_p50_ms: float
    save_latency_p95_ms: float
    load_latency_p50_ms: float
    load_latency_p95_ms: float
    cost_saved_per_day_usd: float
    gpu_hours_saved_per_day: float
    hit_rate_ci_lower: float = 0.0
    hit_rate_ci_upper: float = 0.0


# ──────────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────────

def _make_config(policy_name: str, tmp_dir: str) -> SystemConfig:
    """Create a SystemConfig tuned for fast simulation.

    Total capacity: 30 MB (2 hot + 8 warm + 20 cold).
    This is deliberately small to force heavy evictions under
    enterprise workloads, testing the true victim-selection logic.
    """
    # Map experiment policy names to config eviction policy names
    if policy_name in ("lru", "ttl"):
        eviction_policy = policy_name
    elif policy_name in ("value_density", "value_density_ac"):
        eviction_policy = "value_density"
    else:
        eviction_policy = "predictive"

    cfg = SystemConfig(
        model=ModelConfig(num_layers=2, num_heads=2, head_dim=32, block_size=16, dtype="float16"),
        tiers=TierConfig(
            hot_capacity_mb=50,      # 50 MB hot tier
            warm_capacity_mb=150,    # 150 MB warm
            cold_capacity_mb=300,    # 300 MB cold (Total 500 MB capacity)
            warm_storage_path=os.path.join(tmp_dir, "warm"),
            cold_storage_path=os.path.join(tmp_dir, "cold"),
            cold_backend="local",
        ),
        eviction=EvictionConfig(policy=eviction_policy),
        serialization=SerializationConfig(format="raw_binary", compression="none", compression_level=1),
    )
    return cfg


def _generate_kv_data(model: ModelConfig, token_count: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Generate dummy KV tensors. Token count is clamped to 4096 to keep
    simulation feasible while preserving the log-normal size variance
    that is critical for value-density differentiation."""
    kv_data = {}
    tc = min(token_count, 8192)
    for layer_idx in range(model.num_layers):
        k = np.zeros((model.num_heads, tc, model.head_dim), dtype=np.float16)
        v = np.zeros((model.num_heads, tc, model.head_dim), dtype=np.float16)
        kv_data[layer_idx] = (k, v)
    return kv_data


# ──────────────────────────────────────────────────────────────────
# Single experiment
# ──────────────────────────────────────────────────────────────────

def run_single_experiment(
    policy_name: str,
    profile_name: str,
    duration_days: float = 0.25,
    seed: int = 42,
    predictor: Optional[ResumePredictor] = None,
) -> ExperimentResult:
    """
    Simulate the cache lifecycle for one (policy, workload) pair.

    For each event in the trace:
      - 'start': manager.save() — record save latency
      - 'resume': manager.load() then manager.save() — record hit/miss + load latency
      - 'end': no-op (the session just becomes idle)
    """
    tmp_dir = tempfile.mkdtemp(prefix="kvcache_exp_")

    try:
        config = _make_config(policy_name, tmp_dir)
        manager = TieredCacheManager(config)

        # Inject ML predictor into the appropriate policy
        if predictor:
            if isinstance(manager.eviction_policy, PredictiveEvictionPolicy):
                manager.eviction_policy.set_predictor(predictor)
            elif isinstance(manager.eviction_policy, ValueDensityPolicy):
                manager.eviction_policy.set_predictor(predictor)
                # Set admission threshold for the AC variant
                if policy_name == "value_density_ac":
                    manager.eviction_policy.admission_threshold = 0.00015

        # Generate trace
        sim = WorkloadSimulator(profile_name, duration_days=duration_days, seed=seed)
        events = sim.generate()

        save_latencies: List[float] = []
        load_latencies: List[float] = []
        hit_miss_log: List[int] = []  # 1 = hit, 0 = miss
        actual_tokens_saved: List[int] = []  # Token counts of actual hits

        saved_sessions = set()

        for evt in events:
            if evt.action == "start":
                kv_data = _generate_kv_data(config.model, evt.token_count)
                t0 = time.perf_counter()
                manager.save(evt.session_id, evt.user_id, kv_data)
                t1 = time.perf_counter()
                save_latencies.append((t1 - t0) * 1000)
                saved_sessions.add(evt.session_id)

            elif evt.action == "resume":
                t0 = time.perf_counter()
                hit = manager.load(evt.session_id)
                t_load = (time.perf_counter() - t0) * 1000
                load_latencies.append(t_load)
                
                if hit:
                    hit_miss_log.append(1)
                    actual_tokens_saved.append(evt.token_count)
                else:
                    hit_miss_log.append(0)
                    kv_data = _generate_kv_data(config.model, evt.token_count)
                    manager.save(evt.session_id, evt.user_id, kv_data)
                    saved_sessions.add(evt.session_id)

        # ── Compute metrics ──
        total_loads = len(hit_miss_log)
        total_hits = sum(hit_miss_log)
        total_misses = total_loads - total_hits
        hit_rate = total_hits / max(total_loads, 1)

        save_arr = np.array(save_latencies) if save_latencies else np.array([0.0])
        load_arr = np.array(load_latencies) if load_latencies else np.array([0.0])

        ci_lower, ci_upper = _bootstrap_ci(hit_miss_log, n_bootstrap=1000, seed=seed)

        cost_model = CostModel()
        profile = PROFILES[profile_name]
        
        # Calculate ACTUAL GPU seconds saved based on the exact token counts of the hits
        actual_gpu_seconds_saved = sum(
            cost_model.compute_recompute_time(tc) - (tc / cost_model.tokens_per_second_with_cache)
            for tc in actual_tokens_saved
        )
        
        # Scale to per-day metrics based on simulation duration
        scaling_factor = 1.0 / duration_days
        gpu_seconds_saved_per_day = actual_gpu_seconds_saved * scaling_factor
        gpu_hours_saved_per_day = gpu_seconds_saved_per_day / 3600.0
        cost_saved_per_day_usd = gpu_hours_saved_per_day * cost_model.gpu_cost_per_hour

        return ExperimentResult(
            policy=policy_name,
            workload=profile_name,
            hit_rate=round(hit_rate, 4),
            miss_rate=round(1 - hit_rate, 4),
            total_loads=total_loads,
            total_hits=total_hits,
            total_misses=total_misses,
            save_latency_p50_ms=round(float(np.percentile(save_arr, 50)), 3),
            save_latency_p95_ms=round(float(np.percentile(save_arr, 95)), 3),
            load_latency_p50_ms=round(float(np.percentile(load_arr, 50)), 3),
            load_latency_p95_ms=round(float(np.percentile(load_arr, 95)), 3),
            cost_saved_per_day_usd=round(cost_saved_per_day_usd, 2),
            gpu_hours_saved_per_day=round(gpu_hours_saved_per_day, 2),
            hit_rate_ci_lower=round(ci_lower, 4),
            hit_rate_ci_upper=round(ci_upper, 4),
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    data: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of binary data."""
    if not data:
        return (0.0, 0.0)

    rng = np.random.RandomState(seed)
    arr = np.array(data)
    n = len(arr)
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())

    means = np.array(means)
    alpha = (1 - confidence) / 2
    return (float(np.percentile(means, alpha * 100)),
            float(np.percentile(means, (1 - alpha) * 100)))


# ──────────────────────────────────────────────────────────────────
# Full experiment matrix
# ──────────────────────────────────────────────────────────────────

def run_full_experiment(
    duration_days: float = 0.25,
    seed: int = 42,
    output_dir: str = "benchmarks/results",
) -> List[ExperimentResult]:
    """
    Run the 5 x 3 experiment matrix:
      Policies:  LRU, Heuristic, Logistic V1, Value Density V2, Value Density V2+AC
      workloads = ["power_user"]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Policy definitions: (display_name, config_policy, needs_predictor)
    policies = [
        ("lru",              False),
        ("heuristic",        False),
        ("logistic_v1",      True),    # V1: binary P(resume) — the negative result
        ("value_density",    True),    # V2: P(resume) × cost / size — the fix
        ("value_density_ac", True),    # V2+AC: with admission control
    ]
    workloads = ["casual", "enterprise", "power_user"]

    # Load trained predictor (use logistic for both V1 and V2 for fair comparison)
    logistic_predictor = None
    if os.path.exists("models/logistic_predictor.pkl"):
        logistic_predictor = ResumePredictor.load("models/logistic_predictor.pkl")
        print("  [OK] Loaded logistic predictor from models/logistic_predictor.pkl")
    else:
        print("  [--] No trained logistic predictor found -- ML policies will be skipped")

    results: List[ExperimentResult] = []

    print("\n" + "=" * 100)
    print("  EXPERIMENT V2: Predictive KV Cache Eviction -- V1 vs V2 Value Density")
    print("=" * 100)
    print(f"  Duration: {duration_days * 24:.1f} hours simulated | Seed: {seed}")
    print(f"  Capacity: 500 MB total (50 hot + 150 warm + 300 cold)")
    print(f"  Policies: {', '.join(p[0] for p in policies)}")
    print(f"  Workloads: {', '.join(workloads)}")
    print("=" * 100 + "\n")

    for profile_name in workloads:
        for policy_name, needs_predictor in policies:
            predictor = logistic_predictor if needs_predictor else None

            if needs_predictor and predictor is None:
                print(f"  [SKIP] {policy_name:>18} x {profile_name:<12} -- no trained model")
                continue

            print(f"  [RUN]  {policy_name:>18} x {profile_name:<12} ...", end="", flush=True)
            t0 = time.perf_counter()
            result = run_single_experiment(
                policy_name=policy_name,
                profile_name=profile_name,
                duration_days=duration_days,
                seed=seed,
                predictor=predictor,
            )
            elapsed = time.perf_counter() - t0
            print(f" done ({elapsed:.1f}s) | hit_rate={result.hit_rate:.2%} "
                  f"[{result.hit_rate_ci_lower:.2%}, {result.hit_rate_ci_upper:.2%}]")
            results.append(result)

    # ── Print results table ──
    _print_results_table(results)

    # ── Save results ──
    results_path = os.path.join(output_dir, "experiment_results_v2.json")
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def _print_results_table(results: List[ExperimentResult]):
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 130)
    print("  RESULTS TABLE: V1 -> Failure -> V2 Recovery")
    print("=" * 130)
    header = (f"  {'Policy':<18} {'Workload':<12} {'Hit Rate':>10} {'95% CI':>22} "
              f"{'P50 Load':>10} {'P95 Load':>10} {'$/Day Saved':>12} {'GPU-hr/Day':>12}")
    print(header)
    print("  " + "-" * 126)

    for r in results:
        ci_str = f"[{r.hit_rate_ci_lower:.2%}, {r.hit_rate_ci_upper:.2%}]"
        print(f"  {r.policy:<18} {r.workload:<12} {r.hit_rate:>10.2%} {ci_str:>22} "
              f"{r.load_latency_p50_ms:>9.2f}ms {r.load_latency_p95_ms:>9.2f}ms "
              f"${r.cost_saved_per_day_usd:>10.2f} {r.gpu_hours_saved_per_day:>11.2f}")

    print("=" * 130)


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run KV Cache eviction policy experiments (V2)")
    parser.add_argument("--duration", type=float, default=0.25,
                        help="Simulation duration in days (default: 0.25 = 6 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="benchmarks/results",
                        help="Output directory for results")
    args = parser.parse_args()

    run_full_experiment(
        duration_days=args.duration,
        seed=args.seed,
        output_dir=args.output,
    )
