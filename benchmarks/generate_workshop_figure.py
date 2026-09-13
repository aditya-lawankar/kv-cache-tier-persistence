"""
Compact two-panel figure for the 4-page workshop paper.

Panel A is cache hit rate, panel B is GPU-seconds saved per day, both on the
enterprise-pressure workloads: the synthetic enterprise profile and the Azure
LLM-inference trace replay. Plotting them together is the point of the figure:
the bar ordering reverses between the panels on both workloads, so the
reversal is visible without cross-referencing two tables.

Values come from the same raw run records the full paper's tables are built
from (per-run gpu_hours_saved_per_day, not the cent-rounded aggregate), so
this figure cannot drift from the numbers in the text.

    python benchmarks/generate_workshop_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
OUT = os.path.join(ROOT, "paper", "workshop", "figures")
os.makedirs(OUT, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

C_SYNTH = "#4878CF"
C_AZURE = "#E8743B"

ORDER = ["lru", "heuristic", "logistic_v1", "value_density", "space_time"]
LABELS = ["LRU", "Heuristic", "V1", "V2", "V3"]


def ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(half)


def series(raw_path, workload):
    """(hit%, hit CI, GPU-s/day, GPU-s CI) per policy, in ORDER."""
    rows = json.load(open(raw_path))
    by = {}
    for r in rows:
        if r["workload"] == workload:
            by.setdefault(r["policy"], []).append(r)
    hit, hit_e, val, val_e = [], [], [], []
    for p in ORDER:
        runs = by[p]
        m, h = ci95([r["hit_rate"] * 100 for r in runs])
        hit.append(m)
        hit_e.append(h)
        m, h = ci95([r["gpu_hours_saved_per_day"] * 3600.0 for r in runs])
        val.append(m)
        val_e.append(h)
    return np.array(hit), np.array(hit_e), np.array(val), np.array(val_e)


def main():
    s_hit, s_hit_e, s_val, s_val_e = series(
        os.path.join(RESULTS, "experiment_results_v3_raw.json"), "enterprise")
    a_hit, a_hit_e, a_val, a_val_e = series(
        os.path.join(RESULTS, "experiment_results_azure_raw.json"), "azure_conv")

    x = np.arange(len(ORDER))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.75))

    panels = [
        (axes[0], s_hit, s_hit_e, a_hit, a_hit_e, "Cache hit rate (%)",
         "(a) Hit rate: LRU wins both", "{:.0f}", (0, 108)),
        (axes[1], s_val, s_val_e, a_val, a_val_e, "GPU-seconds saved / day",
         "(b) Delivered value: LRU loses both", "{:.0f}", (0, 92)),
    ]

    for ax, sm, se, am, ae, ylabel, title, fmt, ylim in panels:
        b1 = ax.bar(x - w / 2, sm, w, yerr=se, capsize=2.5, color=C_SYNTH,
                    edgecolor="white", linewidth=0.5, label="Synthetic enterprise",
                    error_kw=dict(lw=0.8, capthick=0.8))
        b2 = ax.bar(x + w / 2, am, w, yerr=ae, capsize=2.5, color=C_AZURE,
                    edgecolor="white", linewidth=0.5, label="Azure trace replay",
                    error_kw=dict(lw=0.8, capthick=0.8))
        for group in (b1, b2):
            for bar in group:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ylim[1] * 0.035,
                        fmt.format(bar.get_height()), ha="center", va="bottom",
                        fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
                   fontsize=7, loc="lower left")

    fig.tight_layout(pad=0.4)
    path = os.path.join(OUT, "figure1_reversal.pdf")
    fig.savefig(path)
    print("wrote", path)

    # Echo the plotted values so they can be diffed against the paper's tables.
    for name, hit, val in (("synthetic-enterprise", s_hit, s_val),
                           ("azure", a_hit, a_val)):
        print(name, [(p, round(h, 1), round(v, 1))
                     for p, h, v in zip(ORDER, hit, val)])


if __name__ == "__main__":
    main()
