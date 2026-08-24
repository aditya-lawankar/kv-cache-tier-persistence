"""
Re-price completed runs under a different architecture, without re-simulating.

Each run logs a histogram over (cached_token_count, hit_tier), which is a
sufficient statistic for the value accounting: savings depend only on those
two quantities plus the cost model. So the dollar figures for a finished run
can be recomputed for any ModelSpec in milliseconds.

    python benchmarks/rescore_results.py --arch llama70b \\
        benchmarks/results/arch_tinyllama/shard_*/experiment_results_v3_raw.json

CORRECTNESS BOUNDARY -- read before using the output:

  EXACT for policies whose eviction DECISIONS ignore the cost model, because
  the sequence of hits is then identical regardless of how hits are priced:
      lru, heuristic, logistic_v1, belady, oracle_v1

  INVALID for policies that score victims by recompute cost -- changing the
  cost model changes which entries they keep, so the hit sequence itself
  differs and the run must be repeated:
      value_density, value_density_ac, space_time, oracle_v3

The script refuses to rescore the second group unless --force is passed,
which exists only for sensitivity checks that explicitly hold behavior fixed.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.kv_cache_tier.utils.cost_model import CostModel
from src.kv_cache_tier.utils.hardware import MODELS_BY_KEY
from benchmarks.experiment_runner import (
    ExperimentResult, aggregate_results, _print_results_table,
)

COST_MODEL_FREE = {"lru", "heuristic", "logistic_v1", "belady", "oracle_v1"}
COST_MODEL_BOUND = {"value_density", "value_density_ac", "space_time", "oracle_v3"}


def rescore(row: dict, cost_model: CostModel, duration_days: float) -> dict:
    gpu_seconds = 0.0
    for key, count in row.get("hit_histogram", {}).items():
        tokens_s, tier = key.split("|")
        gpu_seconds += count * cost_model.savings_per_hit_seconds(int(tokens_s), tier)
    per_day = cost_model.gpu_seconds_to_usd_per_day(gpu_seconds, duration_days)
    out = dict(row)
    out.update(per_day)
    out["arch"] = cost_model.model_spec.name
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="raw result JSON files")
    ap.add_argument("--arch", required=True, choices=sorted(MODELS_BY_KEY.keys()))
    ap.add_argument("--duration-days", type=float, default=0.25)
    ap.add_argument("--output", type=str, default=None,
                    help="write aggregates JSON here")
    ap.add_argument("--output-raw", type=str, default=None,
                    help="write re-priced RAW rows here, so downstream tools "
                         "(e.g. make_paper_tables.py) treat them like a normal run")
    ap.add_argument("--force", action="store_true",
                    help="also rescore cost-model-bound policies (see docstring)")
    args = ap.parse_args()

    cm = CostModel(model_spec=MODELS_BY_KEY[args.arch])

    rows, skipped = [], set()
    for path in args.paths:
        for row in json.load(open(path)):
            if not row.get("hit_histogram"):
                raise SystemExit(
                    f"{path}: run for policy '{row['policy']}' has no hit_histogram "
                    "(produced before per-hit logging existed) -- re-run it instead."
                )
            if row["policy"] in COST_MODEL_BOUND and not args.force:
                skipped.add(row["policy"])
                continue
            rows.append(ExperimentResult(**rescore(row, cm, args.duration_days)))

    if skipped:
        print(f"  [SKIP] cost-model-bound policies (must be re-run under "
              f"--arch {args.arch}): {', '.join(sorted(skipped))}\n")
    if not rows:
        raise SystemExit("Nothing to rescore.")

    print(f"  Re-priced {len(rows)} runs at {cm.model_spec.name} "
          f"({cm.model_spec.kv_bytes_per_token()/1024:.0f} KB/token, "
          f"N*(NVMe)={cm.breakeven_tokens('warm')}, N*(S3)={cm.breakeven_tokens('cold')})")

    aggregates = aggregate_results(rows)
    _print_results_table(aggregates)

    if args.output_raw:
        os.makedirs(os.path.dirname(args.output_raw) or ".", exist_ok=True)
        with open(args.output_raw, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\nRe-priced raw rows saved to {args.output_raw}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([asdict(a) for a in aggregates], f, indent=2)
        print(f"\nAggregates saved to {args.output}")


if __name__ == "__main__":
    main()
