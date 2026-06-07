"""
Latency benchmarks for tier transitions.
"""

import os
import time
import uuid
import logging
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kv_cache_tier.config import SystemConfig, ModelConfig, TierConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache

logger = logging.getLogger(__name__)


def _small_model_config() -> ModelConfig:
    """A tiny model config for quick benchmarks (~2KB per 64-token KV cache)."""
    return ModelConfig(num_layers=2, num_heads=2, head_dim=32, block_size=16, dtype="float16")


class LatencyBenchmark:
    """Measures tier transition latencies."""

    def __init__(self, iterations: int = 50, output_dir: str = "benchmarks/results",
                 use_small_model: bool = False):
        self.iterations = iterations
        self.output_dir = output_dir
        self.use_small_model = use_small_model
        os.makedirs(output_dir, exist_ok=True)

        self.config = SystemConfig.default()
        if use_small_model:
            self.config.model = _small_model_config()
        self.config.tiers.hot_capacity_mb = 16384

    def run(self) -> Dict[str, Any]:
        results = {}
        token_sizes = [64, 256, 512] if self.use_small_model else [128, 512, 1024]

        for tokens in token_sizes:
            logger.info(f"Running latency bench for {tokens} tokens")
            size_results = self._run_for_size(tokens)
            results[f"{tokens}_tokens"] = size_results

        self.plot_results(results, self.output_dir)
        return results

    def _run_for_size(self, token_count: int) -> Dict[str, float]:
        manager = TieredCacheManager(self.config)

        times = {
            "save_hot": [],
            "load_hot": [],
            "promote_warm_hot": [],
            "demote_hot_warm": []
        }

        kv_data = generate_random_kv_cache(self.config.model, token_count)

        for i in range(self.iterations):
            session_id = f"bench_{uuid.uuid4()}"

            start = time.perf_counter()
            manager.save(session_id, "user_1", kv_data)
            times["save_hot"].append(time.perf_counter() - start)

            start = time.perf_counter()
            manager.load(session_id)
            times["load_hot"].append(time.perf_counter() - start)

            start = time.perf_counter()
            manager.demote(session_id)
            times["demote_hot_warm"].append(time.perf_counter() - start)

            start = time.perf_counter()
            manager.load(session_id)
            times["promote_warm_hot"].append(time.perf_counter() - start)

            manager.delete(session_id)

        avgs = {k: sum(v) / len(v) for k, v in times.items()}
        return avgs

    def plot_results(self, results: Dict[str, Any], output_dir: str):
        plt.figure(figsize=(10, 6))

        token_sizes = list(results.keys())
        operations = list(results[token_sizes[0]].keys())

        x = range(len(token_sizes))
        width = 0.2

        for i, op in enumerate(operations):
            y = [results[ts][op] * 1000 for ts in token_sizes]  # ms
            plt.bar([pos + i * width for pos in x], y, width, label=op)

        plt.xlabel("Token Count")
        plt.ylabel("Latency (ms)")
        plt.title("Tier Transition Latencies")
        plt.xticks([pos + width * 1.5 for pos in x], token_sizes)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.savefig(os.path.join(output_dir, "latency_bench.png"))
        plt.close()
