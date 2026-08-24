"""
Produce the paper's results tables from the corrected multi-architecture runs.

Reads every shard directory, merges by (arch, workload, policy) across seeds,
and emits both a console table and LaTeX bodies, with PAIRED per-seed deltas
against LRU.

Value is reported primarily in GPU-seconds saved per day rather than dollars.
At the workload's scale (200-ish sessions/hour) and small-model economics the
dollar figures round to cents, which says more about the deployment size than
about the policies; GPU-seconds is scale-free and converts to dollars by
multiplying by the reader's own $/GPU-hour.

    python benchmarks/make_paper_tables.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

RESULTS = os.path.join(_project_root, "benchmarks", "results")

# Directory -> architecture label. Later entries override earlier ones for the
# same (arch, workload, policy, seed), so corrected re-runs win.
SOURCES = [
    ("arch_tinyllama",    "TinyLlama-1.1B"),
    ("arch_tinyllama_ac", "TinyLlama-1.1B"),
    ("arch_llama70b",     "Llama-2-70B"),
    ("arch_llama70b_ac",  "Llama-2-70B"),
]

POLICY_ORDER = ["lru", "heuristic", "logistic_v1", "value_density",
                "value_density_ac", "space_time",
                "oracle_v1", "oracle_v3", "belady"]
PRETTY = {
    "lru": "LRU", "heuristic": "Heuristic", "logistic_v1": "Logistic V1",
    "value_density": "ValDensity V2", "value_density_ac": "ValDens+AC",
    "space_time": "SpaceTime V3", "oracle_v1": "Oracle V1",
    "oracle_v3": "Oracle V3", "belady": "B\\'{e}l\\'{a}dy",
}
WORKLOAD_ORDER = ["casual", "enterprise", "power_user"]
WL_PRETTY = {"casual": "Casual", "enterprise": "Enterprise", "power_user": "Power User"}


def load():
    """(arch, workload, policy, seed) -> run dict."""
    runs = {}
    for subdir, arch in SOURCES:
        pattern = os.path.join(RESULTS, subdir, "shard_*", "experiment_results_v3_raw.json")
        for path in glob.glob(pattern):
            for r in json.load(open(path)):
                runs[(arch, r["workload"], r["policy"], r["seed"])] = r
    return runs


def gpu_s_per_day(run) -> float:
    return run["gpu_hours_saved_per_day"] * 3600.0


def ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()), (float(a.mean()), float(a.mean()))
    half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), (float(a.mean() - half), float(a.mean() + half))


def summarize(runs):
    cells = defaultdict(dict)          # (arch, wl, policy) -> seed -> run
    for (arch, wl, pol, seed), r in runs.items():
        cells[(arch, wl, pol)][seed] = r

    out = []
    for (arch, wl, pol), by_seed in cells.items():
        lru = cells.get((arch, wl, "lru"), {})
        seeds = sorted(set(by_seed) & set(lru)) if pol != "lru" else sorted(by_seed)
        if not seeds:
            continue
        hit_mean, hit_ci = ci95([by_seed[s]["hit_rate"] for s in sorted(by_seed)])
        val_mean, val_ci = ci95([gpu_s_per_day(by_seed[s]) for s in sorted(by_seed)])

        row = {
            "arch": arch, "workload": wl, "policy": pol, "n_seeds": len(by_seed),
            "hit_mean": hit_mean, "hit_ci": hit_ci,
            "val_mean": val_mean, "val_ci": val_ci,
            "delta": None, "delta_ci": None, "ratio": None, "sig": False,
        }
        if pol != "lru" and lru:
            d = [gpu_s_per_day(by_seed[s]) - gpu_s_per_day(lru[s]) for s in seeds]
            dm, dci = ci95(d)
            lru_mean = np.mean([gpu_s_per_day(lru[s]) for s in seeds])
            row.update(delta=dm, delta_ci=dci,
                       ratio=(val_mean / lru_mean) if lru_mean else None,
                       sig=(dci[0] * dci[1] > 0))
        out.append(row)
    return out


def console(rows):
    for arch in sorted({r["arch"] for r in rows}):
        print(f"\n{'='*104}\n  {arch}\n{'='*104}")
        print(f"  {'Workload':<12}{'Policy':<16}{'Hit%':>7}{'GPU-s/day':>11}{'x LRU':>7}"
              f"{'  paired delta [95% CI]':>30}")
        print("  " + "-" * 100)
        for wl in WORKLOAD_ORDER:
            for pol in POLICY_ORDER:
                r = next((x for x in rows if x["arch"] == arch and x["workload"] == wl
                          and x["policy"] == pol), None)
                if not r:
                    continue
                if r["delta"] is None:
                    tail = "   ---"
                else:
                    tail = (f"   {r['delta']:+7.1f} [{r['delta_ci'][0]:+7.1f},"
                            f"{r['delta_ci'][1]:+7.1f}] {'SIG' if r['sig'] else 'ns '}")
                ratio = f"{r['ratio']:.2f}x" if r["ratio"] else "1.00x"
                print(f"  {WL_PRETTY[wl]:<12}{PRETTY[pol].replace(chr(92),''):<16}"
                      f"{r['hit_mean']:>6.1%}{r['val_mean']:>11.1f}{ratio:>7}{tail}")


def latex(rows, arch, workloads):
    lines = []
    for wl in workloads:
        for pol in POLICY_ORDER:
            r = next((x for x in rows if x["arch"] == arch and x["workload"] == wl
                      and x["policy"] == pol), None)
            if not r:
                continue
            if r["delta"] is None:
                delta = "---"
            else:
                delta = (f"${r['delta']:+.1f}$ $[{r['delta_ci'][0]:+.1f}, "
                         f"{r['delta_ci'][1]:+.1f}]$")
            lines.append(f"{PRETTY[r['policy']]} & {WL_PRETTY[wl]} & "
                         f"{r['hit_mean']*100:.1f} & {r['val_mean']:.1f} & {delta} \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


if __name__ == "__main__":
    runs = load()
    if not runs:
        raise SystemExit("No results found. Run the matrices first.")
    rows = summarize(runs)
    console(rows)
    print("\n\n" + "=" * 104)
    print("  LaTeX table bodies")
    print("=" * 104)
    for arch in sorted({r["arch"] for r in rows}):
        print(f"\n% ---- {arch} ----")
        print(latex(rows, arch, ["enterprise", "power_user"]))
    out = os.path.join(RESULTS, "paper_tables_corrected.json")
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print(f"\nSaved {out}")
