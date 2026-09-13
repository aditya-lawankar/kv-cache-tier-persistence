"""
Generate publication-quality figures for the KV Cache Eviction Policy paper.

All data figures are generated FROM the experiment artifacts
(benchmarks/results/experiment_results_v3_aggregate.json) — no hardcoded
numbers — so every figure in the paper is reproducible by rerunning:

    python benchmarks/experiment_runner.py --seeds 10
    python benchmarks/generate_figures.py

Produces:
  figure1_hit_rate.png      – Grouped bar chart of cache hit rates (95% t-CI over seeds)
  figure2_cost_savings.png  – Grouped bar chart of daily GPU cost savings (95% t-CI)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ───────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
AGG_PATH = os.path.join(RESULTS_DIR, "experiment_results_v3_aggregate.json")
OUT_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
})

# Professional color palette
C_ENTERPRISE = "#4878CF"   # steel blue
C_POWERUSER  = "#E8743B"   # warm orange

POLICY_ORDER = ["lru", "heuristic", "logistic_v1", "value_density", "value_density_ac", "space_time"]
POLICY_LABELS = ["LRU", "Heuristic", "Logistic V1", "Value\nDensity", "Value\nDensity AC", "Space-Time\nDensity V3"]


def _available_policies(agg):
    present = {p for (p, _) in agg}
    order = [p for p in POLICY_ORDER if p in present]
    labels = [POLICY_LABELS[POLICY_ORDER.index(p)] for p in order]
    return order, labels


def load_aggregates():
    """Load per-cell aggregates keyed by (policy, workload)."""
    with open(AGG_PATH) as f:
        rows = json.load(f)
    return {(r["policy"], r["workload"]): r for r in rows}


RAW_PATH = os.path.join(RESULTS_DIR, "experiment_results_v3_raw.json")
GPU_S_PER_USD = 3600.0 / 2.50


def load_raw():
    """Per-run records, keyed by (policy, workload) -> list of runs.

    Value must come from here rather than from the aggregate. The aggregate
    stores dollars rounded to cents, and at small-model economics a whole
    workload is worth a few cents per day, so converting cents back to
    GPU-seconds quantizes every bar to the nearest 1440/100 = 14.4 GPU-s and
    collapses distinct workloads onto identical bars. The raw records carry
    gpu_hours_saved_per_day at four decimals, which is 0.36 GPU-s.
    """
    with open(RAW_PATH) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        out.setdefault((r["policy"], r["workload"]), []).append(r)
    return out


def _ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    from scipy import stats
    half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(half)


def _series(agg, workload, order, raw=None):
    """Means and symmetric CI half-widths for one workload, policy-ordered.

    Hit rates come from the aggregate; value is re-derived from the raw runs
    in GPU-seconds per day (see load_raw)."""
    if raw is None:
        raw = load_raw()
    hit_mean, hit_err, cost_mean, cost_err, delta = [], [], [], [], []
    lru_runs = {r["seed"]: r for r in raw.get(("lru", workload), [])}
    for p in order:
        r = agg[(p, workload)]
        hit_mean.append(r["hit_rate_mean"] * 100)
        lo, hi = r["hit_rate_ci95"]
        hit_err.append((hi - lo) * 100 / 2)

        runs = raw[(p, workload)]
        vals = [x["gpu_hours_saved_per_day"] * 3600.0 for x in runs]
        m, h = _ci95(vals)
        cost_mean.append(m)
        cost_err.append(h)

        if p == "lru" or not lru_runs:
            delta.append(None)
        else:
            paired = [x["gpu_hours_saved_per_day"] * 3600.0
                      - lru_runs[x["seed"]]["gpu_hours_saved_per_day"] * 3600.0
                      for x in runs if x["seed"] in lru_runs]
            delta.append(_ci95(paired)[0] if paired else None)
    return (np.array(hit_mean), np.array(hit_err),
            np.array(cost_mean), np.array(cost_err), delta)


# ════════════════════════════════════════════════════════════════════════
# Figure 1 – Hit Rate Comparison
# ════════════════════════════════════════════════════════════════════════
def make_figure1(agg):
    n_seeds = next(iter(agg.values()))["n_seeds"]
    order, labels = _available_policies(agg)
    hit_ent, err_ent, *_ = _series(agg, "enterprise", order)
    hit_pow, err_pow, *_ = _series(agg, "power_user", order)

    x = np.arange(len(order))
    bar_w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars1 = ax.bar(
        x - bar_w / 2, hit_ent, bar_w,
        yerr=err_ent, capsize=4,
        color=C_ENTERPRISE, edgecolor="white", linewidth=0.6,
        label="Enterprise", error_kw=dict(lw=1.0, capthick=1.0),
    )
    bars2 = ax.bar(
        x + bar_w / 2, hit_pow, bar_w,
        yerr=err_pow, capsize=4,
        color=C_POWERUSER, edgecolor="white", linewidth=0.6,
        label="Power User", error_kw=dict(lw=1.0, capthick=1.0),
    )

    # Value labels on top of bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 2.2,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="medium",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Cache Hit Rate by Policy and Workload "
                 f"(500 MB, 6 hr, {n_seeds} seeds, 95% CI)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure1_hit_rate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# Figure 2 – $/Day Saved Comparison
# ════════════════════════════════════════════════════════════════════════
def make_figure2(agg):
    n_seeds = next(iter(agg.values()))["n_seeds"]
    order, labels = _available_policies(agg)
    _, _, cost_ent, err_ent, _ = _series(agg, "enterprise", order)
    _, _, cost_pow, err_pow, delta_pow = _series(agg, "power_user", order)

    x = np.arange(len(order))
    bar_w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars1 = ax.bar(
        x - bar_w / 2, cost_ent, bar_w,
        yerr=err_ent, capsize=4,
        color=C_ENTERPRISE, edgecolor="white", linewidth=0.6,
        label="Enterprise", error_kw=dict(lw=1.0, capthick=1.0),
    )
    bars2 = ax.bar(
        x + bar_w / 2, cost_pow, bar_w,
        yerr=err_pow, capsize=4,
        color=C_POWERUSER, edgecolor="white", linewidth=0.6,
        label="Power User", error_kw=dict(lw=1.0, capthick=1.0),
    )

    # Value labels. Offset is a fraction of the data range: a fixed offset was
    # calibrated for dollar-scale magnitudes and floats labels off the axes now
    # that the unit is GPU-seconds.
    ymax = max(cost_pow.max(), cost_ent.max())
    for bar_group, errs in ((bars1, err_ent), (bars2, err_pow)):
        for bar, e in zip(bar_group, errs):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + e + ymax * 0.025,
                f"{h:,.0f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="medium",
            )

    # Key-finding annotation: paired delta vs LRU for Logistic V1 (power user),
    # only if the effect is present in the data
    logistic_idx = order.index("logistic_v1")
    d = delta_pow[logistic_idx]
    if d is not None and d > 0:
        r = agg[("logistic_v1", "power_user")]
        lo, hi = [v * GPU_S_PER_USD for v in r["delta_cost_vs_lru_ci95"]]
        target_x = x[logistic_idx] + bar_w / 2
        target_y = cost_pow[logistic_idx]
        ax.annotate(
            f"{d:+,.0f} GPU-s/day vs LRU (paired)\n95% CI [{lo:+,.0f}, {hi:+,.0f}]",
            xy=(target_x, target_y),
            xytext=(target_x + 1.05, target_y + max(cost_pow) * 0.08),
            fontsize=9, fontstyle="italic", color="#c0392b",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#c0392b",
                lw=1.3,
                connectionstyle="arc3,rad=-0.2",
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#e6b0aa", lw=0.8),
            ha="center", va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GPU-seconds saved per day")
    ax.set_ylim(0, ymax * 1.28)
    ax.set_title(f"GPU Time Saved by Policy and Workload ({n_seeds} seeds, 95% CI)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure2_cost_savings.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ════════════════════════════════════════════════════════════════════════
# Figure 3 – Failure-Mode Conceptual 2×2 Matrix
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures from", AGG_PATH)
    aggregates = load_aggregates()
    make_figure1(aggregates)
    make_figure2(aggregates)
    print("Done — all figures saved to", OUT_DIR)
