"""
Experiment Runner V3: 5 Policies x 3 Workload Profiles x N Seeds

Simulates the full KV cache lifecycle using WorkloadSimulator traces,
measuring cache hit rates, hit-tier distribution, and modeled GPU cost
savings for each (policy, workload) combination.

Methodological guarantees (see PROJECT_REVIEW.md §1):
  * Virtual clock: the manager and all eviction policies run on a
    SimulatedClock advanced to each trace event's timestamp, so temporal
    features are on the same scale the ML predictors were trained on.
  * Cached-token accounting: a hit is credited with the recompute cost of
    the tokens that were actually cached, not the resume event's (grown)
    token count.
  * Tier-aware value: savings per hit are discounted by the modeled cost
    of restoring from the tier where the entry was found.
  * Multi-seed statistics: each (policy, workload) cell is run across N
    seeds; policies share traces per seed, so policy deltas are PAIRED.
    Aggregates report mean and 95% t-based CIs across seeds.

Policies:
  - LRU (baseline)
  - Heuristic (V1 weighted formula)
  - Logistic V1 (binary P(resume))
  - Value Density V2 (size-aware: P(resume) x recompute_cost / size_bytes)
  - Value Density V2+AC (V2 with admission control threshold)

Usage:
    python benchmarks/experiment_runner.py
    python benchmarks/experiment_runner.py --duration 0.25 --seeds 10
"""

import json
import os
import sys
import time
import tempfile
import shutil
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.config import SystemConfig, ModelConfig, TierConfig, EvictionConfig, SerializationConfig
from src.kv_cache_tier.core.tiered_manager import TieredCacheManager
from src.kv_cache_tier.eviction.predictive import PredictiveEvictionPolicy
from src.kv_cache_tier.eviction.value_density import ValueDensityPolicy
from src.kv_cache_tier.eviction.space_time import SpaceTimeDensityPolicy
from src.kv_cache_tier.eviction.predictors import ResumePredictor
from src.kv_cache_tier.utils.cost_model import CostModel
from src.kv_cache_tier.utils.clock import SimulatedClock
from benchmarks.workload_simulator import WorkloadSimulator, PROFILES

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """One (policy, workload, seed) run."""
    policy: str
    workload: str
    seed: int
    hit_rate: float
    miss_rate: float
    total_loads: int
    total_hits: int
    total_misses: int
    hits_by_tier: Dict[str, int]
    save_latency_p50_ms: float
    save_latency_p95_ms: float
    load_latency_p50_ms: float
    load_latency_p95_ms: float
    cost_saved_per_day_usd: float
    gpu_hours_saved_per_day: float


@dataclass
class AggregateResult:
    """A (policy, workload) cell aggregated across seeds."""
    policy: str
    workload: str
    n_seeds: int
    hit_rate_mean: float
    hit_rate_ci95: Tuple[float, float]
    cost_saved_per_day_mean: float
    cost_saved_per_day_ci95: Tuple[float, float]
    hits_by_tier_mean: Dict[str, float]
    # Paired deltas vs LRU on the SAME traces (None for the LRU row itself)
    delta_cost_vs_lru_mean: Optional[float] = None
    delta_cost_vs_lru_ci95: Optional[Tuple[float, float]] = None
    delta_hit_rate_vs_lru_mean: Optional[float] = None
    delta_hit_rate_vs_lru_ci95: Optional[Tuple[float, float]] = None


# ──────────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────────

def _make_config(policy_name: str, tmp_dir: str) -> SystemConfig:
    """Create a SystemConfig tuned for fast simulation.

    Total capacity: 500 MB (50 hot + 150 warm + 300 cold).
    Small enough to force sustained eviction pressure under enterprise
    and power-user workloads, exercising the victim-selection logic.
    """
    # Map experiment policy names to config eviction policy names
    if policy_name in ("lru", "ttl"):
        eviction_policy = policy_name
    elif policy_name in ("value_density", "value_density_ac"):
        eviction_policy = "value_density"
    elif policy_name == "space_time":
        eviction_policy = "space_time"
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


# Token counts are clamped to keep simulated tensor sizes feasible while
# preserving the log-normal variance critical for value differentiation.
# EVERYTHING must see the clamped value — the tensors we store, the entry
# metadata policies score on, AND the savings we credit on a hit — or the
# evaluation credits a different quantity than the policies optimize.
TOKEN_CLAMP = 8192


def _generate_kv_data(model: ModelConfig, token_count: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Generate dummy KV tensors (token count clamped to TOKEN_CLAMP)."""
    kv_data = {}
    tc = min(token_count, TOKEN_CLAMP)
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
    events: Optional[list] = None,
) -> ExperimentResult:
    """
    Simulate the cache lifecycle for one (policy, workload, seed) triple.

    The trace is replayed on a SimulatedClock advanced to each event's
    timestamp, so time-dependent policies observe realistic session ages,
    idle gaps, and time-of-day — not the wall-clock time of the replay.

    If `events` is given, that preloaded trace (any WorkloadEvent list,
    e.g. a real-trace window from benchmarks/azure_trace_loader.py) is
    replayed instead of generating a synthetic one; `profile_name` then
    only labels the workload column and `duration_days` must equal the
    trace's span for the per-day scaling to be correct.

    For each event in the trace:
      - 'start':  manager.save() — record save latency
      - 'resume': manager.load(); on hit, credit tier-discounted savings for
                  the CACHED token count, then save the grown context; on
                  miss, save the recomputed context.
      - 'end':    no-op (the session just becomes idle)
    """
    tmp_dir = tempfile.mkdtemp(prefix="kvcache_exp_")

    try:
        clock = SimulatedClock()
        config = _make_config(policy_name, tmp_dir)
        manager = TieredCacheManager(config, clock=clock)

        # Inject ML predictor into the appropriate policy
        if predictor:
            if isinstance(manager.eviction_policy, (PredictiveEvictionPolicy,
                                                    SpaceTimeDensityPolicy)):
                manager.eviction_policy.set_predictor(predictor)
            elif isinstance(manager.eviction_policy, ValueDensityPolicy):
                manager.eviction_policy.set_predictor(predictor)
                # Set admission threshold for the AC variant
                if policy_name == "value_density_ac":
                    manager.eviction_policy.admission_threshold = 0.00015

        # Generate trace (unless a preloaded one was injected)
        if events is None:
            sim = WorkloadSimulator(profile_name, duration_days=duration_days, seed=seed)
            events = sim.generate()

        cost_model = CostModel()

        save_latencies: List[float] = []
        load_latencies: List[float] = []
        hit_miss_log: List[int] = []              # 1 = hit, 0 = miss
        hits_by_tier: Dict[str, int] = {"hot": 0, "warm": 0, "cold": 0}
        gpu_seconds_saved: float = 0.0

        # session_id -> token count of the state actually in the cache.
        # A hit saves recomputing the CACHED context; the resume event's
        # token count includes new turns that were never cached.
        cached_tokens: Dict[str, int] = {}

        # Periodic maintenance activates proactive policies (TTL expiry,
        # V2 admission-control demotion). Without this, should_evict()-based
        # policies are silently inert and AC == plain V2.
        MAINTENANCE_INTERVAL_S = 300.0
        last_maintenance = 0.0

        for evt in events:
            clock.set(evt.timestamp)

            if evt.timestamp - last_maintenance >= MAINTENANCE_INTERVAL_S:
                manager.run_maintenance()
                last_maintenance = evt.timestamp

            if evt.action == "start":
                kv_data = _generate_kv_data(config.model, evt.token_count)
                t0 = time.perf_counter()
                manager.save(evt.session_id, evt.user_id, kv_data)
                t1 = time.perf_counter()
                save_latencies.append((t1 - t0) * 1000)
                cached_tokens[evt.session_id] = min(evt.token_count, TOKEN_CLAMP)

            elif evt.action == "resume":
                # Snapshot tier/size BEFORE load (load promotes to hot inline)
                entry = manager.index.get(evt.session_id)
                hit_tier = entry.tier if entry else None
                hit_size = entry.size_bytes if entry else 0

                t0 = time.perf_counter()
                hit = manager.load(evt.session_id)
                t_load = (time.perf_counter() - t0) * 1000
                load_latencies.append(t_load)

                if hit:
                    hit_miss_log.append(1)
                    hits_by_tier[hit_tier] += 1
                    gpu_seconds_saved += cost_model.savings_per_hit_seconds(
                        cached_token_count=cached_tokens.get(evt.session_id, 0),
                        tier=hit_tier,
                        size_bytes=hit_size,
                    )
                else:
                    hit_miss_log.append(0)

                # Either way the conversation continues and its grown context
                # is (re)cached — on a hit only the new turns were computed,
                # on a miss the whole prefix was recomputed.
                kv_data = _generate_kv_data(config.model, evt.token_count)
                manager.save(evt.session_id, evt.user_id, kv_data)
                cached_tokens[evt.session_id] = min(evt.token_count, TOKEN_CLAMP)

        # ── Compute metrics ──
        total_loads = len(hit_miss_log)
        total_hits = sum(hit_miss_log)
        total_misses = total_loads - total_hits
        hit_rate = total_hits / max(total_loads, 1)

        save_arr = np.array(save_latencies) if save_latencies else np.array([0.0])
        load_arr = np.array(load_latencies) if load_latencies else np.array([0.0])

        # Scale to per-day metrics based on simulation duration
        scaling_factor = 1.0 / duration_days
        gpu_hours_saved_per_day = (gpu_seconds_saved * scaling_factor) / 3600.0
        cost_saved_per_day_usd = gpu_hours_saved_per_day * cost_model.gpu_cost_per_hour

        return ExperimentResult(
            policy=policy_name,
            workload=profile_name,
            seed=seed,
            hit_rate=round(hit_rate, 4),
            miss_rate=round(1 - hit_rate, 4),
            total_loads=total_loads,
            total_hits=total_hits,
            total_misses=total_misses,
            hits_by_tier=hits_by_tier,
            save_latency_p50_ms=round(float(np.percentile(save_arr, 50)), 3),
            save_latency_p95_ms=round(float(np.percentile(save_arr, 95)), 3),
            load_latency_p50_ms=round(float(np.percentile(load_arr, 50)), 3),
            load_latency_p95_ms=round(float(np.percentile(load_arr, 95)), 3),
            cost_saved_per_day_usd=round(cost_saved_per_day_usd, 2),
            gpu_hours_saved_per_day=round(gpu_hours_saved_per_day, 2),
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────
# Cross-seed statistics
# ──────────────────────────────────────────────────────────────────

def _mean_ci95(values: List[float]) -> Tuple[float, Tuple[float, float]]:
    """Mean and 95% t-based CI across seeds."""
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    n = len(arr)
    if n < 2:
        return mean, (mean, mean)
    from scipy import stats
    sem = arr.std(ddof=1) / np.sqrt(n)
    half = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return mean, (round(mean - half, 4), round(mean + half, 4))


def aggregate_results(results: List[ExperimentResult]) -> List[AggregateResult]:
    """Aggregate per-seed runs into per-cell means with 95% CIs, plus
    PAIRED deltas vs LRU (policies share the same trace per seed)."""
    by_cell: Dict[Tuple[str, str], List[ExperimentResult]] = {}
    for r in results:
        by_cell.setdefault((r.policy, r.workload), []).append(r)

    aggregates: List[AggregateResult] = []
    for (policy, workload), runs in by_cell.items():
        runs = sorted(runs, key=lambda r: r.seed)
        hit_mean, hit_ci = _mean_ci95([r.hit_rate for r in runs])
        cost_mean, cost_ci = _mean_ci95([r.cost_saved_per_day_usd for r in runs])
        tier_means = {
            t: float(np.mean([r.hits_by_tier.get(t, 0) for r in runs]))
            for t in ("hot", "warm", "cold")
        }

        agg = AggregateResult(
            policy=policy,
            workload=workload,
            n_seeds=len(runs),
            hit_rate_mean=round(hit_mean, 4),
            hit_rate_ci95=hit_ci,
            cost_saved_per_day_mean=round(cost_mean, 2),
            cost_saved_per_day_ci95=cost_ci,
            hits_by_tier_mean=tier_means,
        )

        # Paired comparison against LRU on identical traces
        lru_runs = {r.seed: r for r in by_cell.get(("lru", workload), [])}
        if policy != "lru" and lru_runs:
            paired = [(r, lru_runs[r.seed]) for r in runs if r.seed in lru_runs]
            if len(paired) >= 2:
                d_cost = [r.cost_saved_per_day_usd - l.cost_saved_per_day_usd for r, l in paired]
                d_hit = [r.hit_rate - l.hit_rate for r, l in paired]
                dc_mean, dc_ci = _mean_ci95(d_cost)
                dh_mean, dh_ci = _mean_ci95(d_hit)
                agg.delta_cost_vs_lru_mean = round(dc_mean, 2)
                agg.delta_cost_vs_lru_ci95 = dc_ci
                agg.delta_hit_rate_vs_lru_mean = round(dh_mean, 4)
                agg.delta_hit_rate_vs_lru_ci95 = dh_ci

        aggregates.append(agg)

    return aggregates


# ──────────────────────────────────────────────────────────────────
# Full experiment matrix
# ──────────────────────────────────────────────────────────────────

def run_full_experiment(
    duration_days: float = 0.25,
    seeds: Optional[List[int]] = None,
    output_dir: str = "benchmarks/results",
    only_policies: Optional[List[str]] = None,
) -> Tuple[List[ExperimentResult], List[AggregateResult]]:
    """
    Run the 5 x 3 x N experiment matrix:
      Policies:  LRU, Heuristic, Logistic V1, Value Density V2, Value Density V2+AC
      Workloads: casual, enterprise, power_user
      Seeds:     N independent traces; all policies replay the same trace
                 per seed, enabling paired policy comparisons.
    """
    if seeds is None:
        seeds = list(range(42, 52))  # 10 seeds by default

    os.makedirs(output_dir, exist_ok=True)

    policies = [
        ("lru",              False),
        ("heuristic",        False),
        ("logistic_v1",      True),    # V1: binary P(resume)
        ("value_density",    True),    # V2: P(resume) x cost / size
        ("value_density_ac", True),    # V2+AC: with admission control
        ("space_time",       True),    # V3: P(resume) x cost / (size x E[dt])
    ]
    if only_policies:
        policies = [p for p in policies if p[0] in only_policies]
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
    print("  EXPERIMENT V3: Multi-seed, virtual-clock, tier-aware evaluation")
    print("=" * 100)
    print(f"  Duration: {duration_days * 24:.1f} hours simulated | Seeds: {seeds}")
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

            t0 = time.perf_counter()
            cell_results = []
            for seed in seeds:
                cell_results.append(run_single_experiment(
                    policy_name=policy_name,
                    profile_name=profile_name,
                    duration_days=duration_days,
                    seed=seed,
                    predictor=predictor,
                ))
            elapsed = time.perf_counter() - t0
            results.extend(cell_results)

            mean_hit = float(np.mean([r.hit_rate for r in cell_results]))
            mean_cost = float(np.mean([r.cost_saved_per_day_usd for r in cell_results]))
            print(f"  [RUN]  {policy_name:>18} x {profile_name:<12} "
                  f"({len(seeds)} seeds, {elapsed:.1f}s) | "
                  f"hit_rate={mean_hit:.2%} | ${mean_cost:,.0f}/day")

    aggregates = aggregate_results(results)
    _print_results_table(aggregates)

    # ── Save results ──
    raw_path = os.path.join(output_dir, "experiment_results_v3_raw.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    agg_path = os.path.join(output_dir, "experiment_results_v3_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump([asdict(a) for a in aggregates], f, indent=2)

    print(f"\nRaw results saved to {raw_path}")
    print(f"Aggregates saved to {agg_path}")

    return results, aggregates


def _print_results_table(aggregates: List[AggregateResult]):
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 132)
    print("  RESULTS (mean over seeds, 95% t-CI; deltas are PAIRED vs LRU on identical traces)")
    print("=" * 132)
    header = (f"  {'Policy':<18} {'Workload':<12} {'Hit Rate':>9} {'Hit CI':>18} "
              f"{'$/Day':>10} {'$/Day CI':>22} {'d$ vs LRU':>10} {'d$ CI':>22}")
    print(header)
    print("  " + "-" * 128)

    order = {"lru": 0, "heuristic": 1, "logistic_v1": 2, "value_density": 3,
             "value_density_ac": 4, "space_time": 5}
    for a in sorted(aggregates, key=lambda x: (x.workload, order.get(x.policy, 9))):
        hit_ci = f"[{a.hit_rate_ci95[0]:.2%}, {a.hit_rate_ci95[1]:.2%}]"
        cost_ci = f"[{a.cost_saved_per_day_ci95[0]:,.0f}, {a.cost_saved_per_day_ci95[1]:,.0f}]"
        if a.delta_cost_vs_lru_mean is not None:
            d_cost = f"{a.delta_cost_vs_lru_mean:+,.0f}"
            d_ci = f"[{a.delta_cost_vs_lru_ci95[0]:+,.0f}, {a.delta_cost_vs_lru_ci95[1]:+,.0f}]"
        else:
            d_cost, d_ci = "--", "--"
        print(f"  {a.policy:<18} {a.workload:<12} {a.hit_rate_mean:>9.2%} {hit_ci:>18} "
              f"{a.cost_saved_per_day_mean:>10,.0f} {cost_ci:>22} {d_cost:>10} {d_ci:>22}")

    print("=" * 132)


# ──────────────────────────────────────────────────────────────────
# Real-trace experiment (Azure LLM inference trace)
# ──────────────────────────────────────────────────────────────────

def run_azure_experiment(
    output_dir: str = "benchmarks/results",
    only_policies: Optional[List[str]] = None,
) -> Tuple[List[ExperimentResult], List[AggregateResult]]:
    """
    Replay ten 6-hour windows of the Azure LLM inference trace
    (see benchmarks/azure_trace_loader.py for the conversion semantics)
    through all six policies. Each window is one replicate: all policies
    replay the identical event list per window, so deltas vs LRU are
    PAIRED exactly as in the synthetic matrix, with windows playing the
    role of seeds. Results are written to separate azure_* files so the
    synthetic canonical results are never clobbered.
    """
    from benchmarks.azure_trace_loader import AzureTraceWorkload, WINDOWS, ensure_trace_npz

    os.makedirs(output_dir, exist_ok=True)
    ensure_trace_npz()

    policies = [
        ("lru",              False),
        ("heuristic",        False),
        ("logistic_v1",      True),
        ("value_density",    True),
        ("value_density_ac", True),
        ("space_time",       True),
    ]
    if only_policies:
        policies = [p for p in policies if p[0] in only_policies]

    logistic_predictor = None
    if os.path.exists("models/logistic_predictor.pkl"):
        logistic_predictor = ResumePredictor.load("models/logistic_predictor.pkl")
        print("  [OK] Loaded logistic predictor from models/logistic_predictor.pkl")
    else:
        print("  [--] No trained logistic predictor found -- ML policies will be skipped")

    # Pre-generate every window's trace once; policies share them.
    window_events = []
    for w in range(len(WINDOWS)):
        wl = AzureTraceWorkload(w, seed=1000 + w)
        window_events.append(wl.generate())

    duration_days = 6.0 / 24.0  # 6-hour windows, same horizon as synthetic

    print("\n" + "=" * 100)
    print("  REAL-TRACE EXPERIMENT: Azure LLM inference trace (conv, 1 week, 2024)")
    print("=" * 100)
    print(f"  Windows: {len(window_events)} x 6h ({', '.join(l for l, _ in WINDOWS)})")
    print(f"  Capacity: 500 MB total (50 hot + 150 warm + 300 cold)")
    print(f"  Policies: {', '.join(p[0] for p in policies)}")
    print("=" * 100 + "\n")

    results: List[ExperimentResult] = []
    for policy_name, needs_predictor in policies:
        predictor = logistic_predictor if needs_predictor else None
        if needs_predictor and predictor is None:
            print(f"  [SKIP] {policy_name:>18} x azure_conv -- no trained model")
            continue

        t0 = time.perf_counter()
        cell_results = []
        for w, events in enumerate(window_events):
            cell_results.append(run_single_experiment(
                policy_name=policy_name,
                profile_name="azure_conv",
                duration_days=duration_days,
                seed=w,
                predictor=predictor,
                events=events,
            ))
        elapsed = time.perf_counter() - t0
        results.extend(cell_results)

        mean_hit = float(np.mean([r.hit_rate for r in cell_results]))
        mean_cost = float(np.mean([r.cost_saved_per_day_usd for r in cell_results]))
        print(f"  [RUN]  {policy_name:>18} x azure_conv   "
              f"({len(window_events)} windows, {elapsed:.1f}s) | "
              f"hit_rate={mean_hit:.2%} | ${mean_cost:,.0f}/day")

    aggregates = aggregate_results(results)
    _print_results_table(aggregates)

    raw_path = os.path.join(output_dir, "experiment_results_azure_raw.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    agg_path = os.path.join(output_dir, "experiment_results_azure_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump([asdict(a) for a in aggregates], f, indent=2)

    print(f"\nRaw results saved to {raw_path}")
    print(f"Aggregates saved to {agg_path}")
    return results, aggregates


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run KV Cache eviction policy experiments (V3)")
    parser.add_argument("--duration", type=float, default=0.25,
                        help="Simulation duration in days (default: 0.25 = 6 hours)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (traces) per cell (default: 10)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First seed value (default: 42)")
    parser.add_argument("--output", type=str, default="benchmarks/results",
                        help="Output directory for results")
    parser.add_argument("--policies", type=str, default=None,
                        help="Comma-separated policy filter (e.g. 'space_time')")
    parser.add_argument("--azure", action="store_true",
                        help="Replay the Azure LLM inference trace instead of "
                             "synthetic workloads (downloads ~1.1 GB on first use)")
    args = parser.parse_args()

    if args.azure:
        run_azure_experiment(
            output_dir=args.output,
            only_policies=args.policies.split(",") if args.policies else None,
        )
    else:
        run_full_experiment(
            duration_days=args.duration,
            seeds=list(range(args.seed_base, args.seed_base + args.seeds)),
            output_dir=args.output,
            only_policies=args.policies.split(",") if args.policies else None,
        )
