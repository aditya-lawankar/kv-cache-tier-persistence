"""
Merge sharded experiment results (from parallel runs) into the
canonical raw + aggregate result files, and print the final table.

Usage:
    # Synthetic matrix, sharded by seed range:
    python benchmarks/merge_results.py benchmarks/results/shard_*/experiment_results_v3_raw.json

    # Azure real-trace runs, sharded by policy:
    python benchmarks/merge_results.py --prefix azure benchmarks/results/azure_shards/*/experiment_results_azure_raw.json

The --prefix selects the canonical output pair
(experiment_results_<prefix>_raw.json / _aggregate.json); default: v3.
"""

import argparse
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


def main(paths, output_dir="benchmarks/results", prefix="v3"):
    results = []
    for path in paths:
        with open(path) as f:
            for row in json.load(f):
                results.append(ExperimentResult(**row))

    # Sanity: no duplicate (policy, workload, seed)
    keys = [(r.policy, r.workload, r.seed) for r in results]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate (policy, workload, seed) rows across shards — check shard ranges")

    seeds = sorted(set(r.seed for r in results))
    print(f"Merged {len(results)} runs across {len(seeds)} seeds: {seeds}")

    # Canonical row order, independent of shard glob order: workload-major,
    # then policy in presentation order, then seed.
    policy_order = {"lru": 0, "heuristic": 1, "logistic_v1": 2, "value_density": 3,
                    "value_density_ac": 4, "space_time": 5}
    results.sort(key=lambda r: (r.workload, policy_order.get(r.policy, 9), r.seed))

    aggregates = aggregate_results(results)
    _print_results_table(aggregates)

    raw_path = os.path.join(output_dir, f"experiment_results_{prefix}_raw.json")
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    agg_path = os.path.join(output_dir, f"experiment_results_{prefix}_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump([asdict(a) for a in aggregates], f, indent=2)
    print(f"\nRaw results saved to {raw_path}")
    print(f"Aggregates saved to {agg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Sharded *_raw.json files to merge")
    parser.add_argument("--prefix", default="v3",
                        help="Canonical output prefix: v3 (synthetic) or azure (default: v3)")
    parser.add_argument("--output", default="benchmarks/results",
                        help="Output directory (default: benchmarks/results)")
    args = parser.parse_args()
    main(args.paths, output_dir=args.output, prefix=args.prefix)
