"""
Cache hit rate benchmarks.
"""

import os
import logging
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kv_cache_tier.config import SystemConfig, ModelConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache

from .workload_generator import WorkloadGenerator, WorkloadConfig

logger = logging.getLogger(__name__)


def _small_model_config() -> ModelConfig:
    """A tiny model config for hit-rate testing."""
    return ModelConfig(num_layers=1, num_heads=1, head_dim=16, block_size=16, dtype="float16")


class HitRateBenchmark:
    """Measures cache hit rates under different eviction policies."""

    def __init__(self, output_dir: str = "benchmarks/results",
                 num_users: int = 50, duration_seconds: int = 1800):
        self.output_dir = output_dir
        self.num_users = num_users
        self.duration_seconds = duration_seconds
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        gen = WorkloadGenerator(WorkloadConfig(
            num_users=self.num_users,
            duration_seconds=self.duration_seconds
        ))
        events = gen.generate()
        logger.info(f"Generated trace with {len(events)} events")

        policies = ["lru", "ttl", "predictive"]
        results = {}

        for policy in policies:
            logger.info(f"Running hit rate bench for policy: {policy}")
            hit_rate = self._run_trace(events, policy)
            results[policy] = hit_rate

        self.plot_results(results, self.output_dir)
        return results

    def _run_trace(self, events: List, policy: str) -> float:
        config = SystemConfig.default()
        config.eviction.policy = policy
        # Use a miniature model — we only care about hit/miss logic, not tensor size.
        config.model = _small_model_config()
        config.tiers.hot_capacity_mb = 1
        config.tiers.warm_capacity_mb = 2

        manager = TieredCacheManager(config)

        # Pre-generate tiny dummy tensors
        dummy_caches = {
            32: generate_random_kv_cache(config.model, 32),
            64: generate_random_kv_cache(config.model, 64),
            128: generate_random_kv_cache(config.model, 128),
        }

        hits = 0
        resumes = 0

        for event in events:
            if event.action == 'resume':
                resumes += 1
                data = manager.load(event.session_id)
                if data is not None:
                    hits += 1
            elif event.action == 'end':
                size = min([32, 64, 128], key=lambda x: abs(x - event.token_count))
                manager.save(event.session_id, event.user_id, dummy_caches[size])

        return hits / resumes if resumes > 0 else 0.0

    def plot_results(self, results: Dict[str, float], output_dir: str):
        plt.figure(figsize=(8, 6))

        policies = list(results.keys())
        rates = [results[p] * 100 for p in policies]

        plt.bar(policies, rates, color=['#4C72B0', '#55A868', '#C44E52'])

        plt.xlabel("Eviction Policy")
        plt.ylabel("Cache Hit Rate (%)")
        plt.title("Hit Rate by Eviction Policy")
        plt.ylim(0, 100)

        for i, rate in enumerate(rates):
            plt.text(i, rate + 1, f"{rate:.1f}%", ha='center')

        plt.savefig(os.path.join(output_dir, "hit_rate_bench.png"))
        plt.close()
