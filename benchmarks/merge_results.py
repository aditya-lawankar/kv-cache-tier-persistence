"""
Merge sharded experiment results (from parallel seed runs) into the
canonical raw + aggregate result files, and print the final table.

Usage:
    python benchmarks/merge_results.py benchmarks/results/shard_*/experiment_results_v3_raw.json
"""

import json
import os
import sys
from dataclasses import asdict

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from benchmarks.experiment_runner import (
    ExperimentResult, aggregate_results, _print_results_table,
)


def main(paths, output_dir="benchmarks/results"):
    results = []
    for path in paths:
        with open(path) as f:
            for row in json.load(f):
                results.append(ExperimentResult(**row))

    # Sanity: no duplicate (policy, workload, seed)
    keys = [(r.policy, r.workload, r.seed) for r in results]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate (policy, workload, seed) rows across shards — check --seed-base ranges")

    seeds = sorted(set(r.seed for r in results))
    print(f"Merged {len(results)} runs across {len(seeds)} seeds: {seeds}")

    aggregates = aggregate_results(results)
    _print_results_table(aggregates)

    raw_path = os.path.join(output_dir, "experiment_results_v3_raw.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    agg_path = os.path.join(output_dir, "experiment_results_v3_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump([asdict(a) for a in aggregates], f, indent=2)
    print(f"\nRaw results saved to {raw_path}")
    print(f"Aggregates saved to {agg_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
