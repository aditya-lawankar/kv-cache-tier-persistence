"""
Generate the capacity regime figure (Figure 5) from the capacity sweep.

Reads benchmarks/results/capacity_sweep/cap_<MB>/experiment_results_v3_aggregate.json
for each capacity point and renders a two-panel figure: hit rate vs capacity
and value vs capacity, for the three deployable policies (LRU, Logistic V1,
Space-Time V3) and the three oracles (Belady, Oracle-V1, Oracle-V3).

Output: benchmarks/results/figures/figure5_capacity_sweep.pdf (vector)
        and a copy in paper/latex/figures/.
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CAPACITIES = [125, 250, 500, 1000, 2000]
SWEEP_DIR = "capacity_v2"        # corrected, architecture-priced runs
MAIN_DIRS = ["arch_tinyllama", "arch_tinyllama_ac"]   # supplies the 500 MB point
POLICIES = [
    ("lru",           "LRU",             "#1f77b4", "o", "-"),
    ("logistic_v1",   "Logistic V1",     "#ff7f0e", "s", "-"),
    ("value_density", "Value Density V2", "#d62728", "P", "-"),
    ("space_time",    "Space-Time V3",   "#2ca02c", "D", "-"),
    ("belady",        "Bélády (oracle)", "#7f7f7f", "^", "--"),
]
# oracle_v3 tracks belady closely; omitted so the oracle band stays readable.


def load():
    """capacity -> policy -> {hit_rate_mean, hit_ci, value_mean, value_ci}.

    Value is GPU-seconds saved per day. The 500 MB point comes from the main
    matrix (same seeds, same economics) rather than being re-simulated.
    """
    import numpy as np
    from scipy import stats
    GPU_S = 3600.0
    SEEDS = list(range(42, 47))

    def ci(vals):
        a = np.asarray(vals, dtype=float)
        if len(a) < 2:
            return float(a.mean()), (float(a.mean()), float(a.mean()))
        h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        return float(a.mean()), (float(a.mean() - h), float(a.mean() + h))

    data = {}
    for cap in CAPACITIES:
        if cap == 500:
            pattern = [os.path.join(_project_root, "benchmarks", "results", d,
                                    "shard_*", "experiment_results_v3_raw.json")
                       for d in MAIN_DIRS]
        else:
            pattern = [os.path.join(_project_root, "benchmarks", "results", SWEEP_DIR,
                                    f"cap_{cap}", "experiment_results_v3_raw.json")]
        runs = {}
        for pat in pattern:
            for path in glob.glob(pat):
                for r in json.load(open(path)):
                    if r["workload"] == "enterprise" and r["seed"] in SEEDS:
                        runs[(r["policy"], r["seed"])] = r
        cell = {}
        for key, _, _, _, _ in POLICIES:
            hits = [runs[(key, s)]["hit_rate"] * 100 for s in SEEDS if (key, s) in runs]
            vals = [runs[(key, s)]["gpu_hours_saved_per_day"] * GPU_S
                    for s in SEEDS if (key, s) in runs]
            if not hits:
                continue
            hm, hc = ci(hits)
            vm, vc = ci(vals)
            cell[key] = {"hit_rate_mean": hm, "hit_ci95": hc,
                         "value_mean": vm, "value_ci95": vc}
        data[cap] = cell
    return data


def main():
    data = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    for key, label, color, marker, ls in POLICIES:
        caps = [c for c in CAPACITIES if key in data[c]]
        if not caps:
            continue
        hit = [data[c][key]["hit_rate_mean"] for c in caps]
        hit_ci = np.array([data[c][key]["hit_ci95"] for c in caps])
        val = [data[c][key]["value_mean"] for c in caps]
        val_ci = np.array([data[c][key]["value_ci95"] for c in caps])

        ax1.plot(caps, hit, marker=marker, ls=ls, color=color, label=label, ms=4.5, lw=1.5)
        ax1.fill_between(caps, hit_ci[:, 0], hit_ci[:, 1], color=color, alpha=0.12, lw=0)
        ax2.plot(caps, val, marker=marker, ls=ls, color=color, label=label, ms=4.5, lw=1.5)
        ax2.fill_between(caps, val_ci[:, 0], val_ci[:, 1], color=color, alpha=0.12, lw=0)

    for ax, ylab in ((ax1, "Hit rate (%)"), (ax2, "GPU-seconds saved per day")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(CAPACITIES)
        ax.set_xticklabels([str(c) for c in CAPACITIES])
        ax.set_xlabel("Total cache capacity (MB)")
        ax.set_ylabel(ylab)
        ax.axvline(500, color="k", lw=0.8, alpha=0.35, ls=":")
        ax.grid(alpha=0.25, lw=0.5)
        ax.minorticks_off()

    ax1.annotate("canonical", xy=(500, ax1.get_ylim()[0]), xytext=(3, 4),
                 textcoords="offset points", fontsize=7, alpha=0.6, rotation=90)
    ax2.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()

    out1 = os.path.join(_project_root, "benchmarks", "results", "figures",
                        "figure5_capacity_sweep.pdf")
    out2 = os.path.join(_project_root, "paper", "latex", "figures",
                        "figure5_capacity_sweep.pdf")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig.savefig(out1)
    fig.savefig(out2)
    print(f"  [OK] {out1}")
    print(f"  [OK] {out2}")


if __name__ == "__main__":
    main()
