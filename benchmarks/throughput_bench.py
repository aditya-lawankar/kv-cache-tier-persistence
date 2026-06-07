"""
Throughput benchmarks.
"""

import os
import time
import uuid
import logging
import threading
from typing import Dict, Any, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kv_cache_tier.config import SystemConfig, ModelConfig
from kv_cache_tier.core.tiered_manager import TieredCacheManager
from kv_cache_tier.utils.tensor_utils import generate_random_kv_cache

logger = logging.getLogger(__name__)


def _small_model_config() -> ModelConfig:
    """A tiny model config for quick benchmarks."""
    return ModelConfig(num_layers=2, num_heads=2, head_dim=32, block_size=16, dtype="float16")


class ThroughputBenchmark:
    """Measures operations per second."""

    def __init__(self, iterations: int = 100, output_dir: str = "benchmarks/results",
                 use_small_model: bool = False):
        self.iterations = iterations
        self.output_dir = output_dir
        self.use_small_model = use_small_model
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        results = {}
        formats = ["safetensors", "raw_binary"]
        workers_list = [1, 2, 4]

        for fmt in formats:
            logger.info(f"Running throughput bench for {fmt}")
            fmt_results = {}
            for w in workers_list:
                ops_sec = self._run_workers(fmt, w)
                fmt_results[f"{w}_workers"] = ops_sec
            results[fmt] = fmt_results

        self.plot_results(results, self.output_dir)
        return results

    def _run_workers(self, format_name: str, num_workers: int) -> float:
        config = SystemConfig.default()
        if self.use_small_model:
            config.model = _small_model_config()
        config.serialization.format = format_name
        config.serialization.compression = "none"
        config.tiers.hot_capacity_mb = 16384

        manager = TieredCacheManager(config)
        token_count = 64 if self.use_small_model else 256
        kv_data = generate_random_kv_cache(config.model, token_count)

        ops_per_worker = self.iterations // num_workers
        total_ops = ops_per_worker * num_workers

        def worker_task():
            for _ in range(ops_per_worker):
                session_id = f"bench_{uuid.uuid4()}"
                manager.save(session_id, "user_1", kv_data)
                manager.load(session_id)
                manager.delete(session_id)

        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker_task)
            threads.append(t)

        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        return total_ops / elapsed

    def plot_results(self, results: Dict[str, Any], output_dir: str):
        plt.figure(figsize=(8, 6))

        formats = list(results.keys())
        workers = list(results[formats[0]].keys())

        x = range(len(workers))
        width = 0.35

        for i, fmt in enumerate(formats):
            y = [results[fmt][w] for w in workers]
            plt.bar([pos + i * width for pos in x], y, width, label=fmt)

        plt.xlabel("Number of Workers")
        plt.ylabel("Operations per Second (Save+Load)")
        plt.title("Throughput by Serialization Format")
        plt.xticks([pos + width / 2 for pos in x], [w.replace("_workers", "") for w in workers])
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plt.savefig(os.path.join(output_dir, "throughput_bench.png"))
        plt.close()
