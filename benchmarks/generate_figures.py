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
  figure3_failure_modes.png – 2x2 conceptual decision matrix
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


def _series(agg, workload, order):
    """Extract means and symmetric CI half-widths for one workload, policy-ordered."""
    hit_mean, hit_err, cost_mean, cost_err, delta = [], [], [], [], []
    for p in order:
        r = agg[(p, workload)]
        hit_mean.append(r["hit_rate_mean"] * 100)
        lo, hi = r["hit_rate_ci95"]
        hit_err.append((hi - lo) * 100 / 2)
        cost_mean.append(r["cost_saved_per_day_mean"])
        lo, hi = r["cost_saved_per_day_ci95"]
        cost_err.append((hi - lo) / 2)
        delta.append(r.get("delta_cost_vs_lru_mean"))
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

    # Value labels
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 50,
                f"${h:,.0f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="medium",
            )

    # Key-finding annotation: paired delta vs LRU for Logistic V1 (power user),
    # only if the effect is present in the data
    logistic_idx = order.index("logistic_v1")
    d = delta_pow[logistic_idx]
    if d is not None and d > 0:
        r = agg[("logistic_v1", "power_user")]
        lo, hi = r["delta_cost_vs_lru_ci95"]
        target_x = x[logistic_idx] + bar_w / 2
        target_y = cost_pow[logistic_idx]
        ax.annotate(
            f"+${d:,.0f}/day vs LRU (paired)\n95% CI [{lo:+,.0f}, {hi:+,.0f}]",
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
    ax.set_ylabel("$/Day Saved (USD)")
    ax.set_ylim(0, max(cost_pow.max(), cost_ent.max()) * 1.3)
    ax.set_title(f"Daily GPU Cost Savings by Policy and Workload ({n_seeds} seeds, 95% CI)")
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
def make_figure3(agg):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Quadrant colours (muted pastels)
    colors = {
        "TL": "#d6eaf8",  # light blue
        "TR": "#fdebd0",  # light peach
        "BL": "#d5f5e3",  # light green
        "BR": "#f5b7b1",  # light coral
    }

    quadrants = [
        (0.0, 0.5, 0.5, 0.5, "TL"),
        (0.5, 0.5, 0.5, 0.5, "TR"),
        (0.0, 0.0, 0.5, 0.5, "BL"),
        (0.5, 0.0, 0.5, 0.5, "BR"),
    ]
    for x0, y0, w, h, key in quadrants:
        rect = mpatches.FancyBboxPatch(
            (x0 + 0.01, y0 + 0.01), w - 0.02, h - 0.02,
            boxstyle="round,pad=0.02",
            facecolor=colors[key], edgecolor="#888888", linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

    # Pull headline numbers from the data
    lru_ent = agg[("lru", "enterprise")]["hit_rate_mean"] * 100
    lru_pow = agg[("lru", "power_user")]
    heur_pow = agg[("heuristic", "power_user")]
    vd_pow_hit = agg[("value_density", "power_user")]["hit_rate_mean"] * 100
    casual_hit = agg[("lru", "casual")]["hit_rate_mean"] * 100

    # Value/hit-rate decoupling: heuristic's value retention at lower hit rate
    hit_gap = (lru_pow["hit_rate_mean"] - heur_pow["hit_rate_mean"]) * 100
    value_retained = heur_pow["cost_saved_per_day_mean"] / lru_pow["cost_saved_per_day_mean"] * 100
    tr_line = (f"Power User\nHits $\\neq$ Value:\n$-${hit_gap:.0f}pt hits $\\Rightarrow$ "
               f"{value_retained:.0f}% of value")

    st_delta = None
    if ("space_time", "power_user") in agg and ("value_density", "power_user") in agg:
        st_delta = (agg[("space_time", "power_user")]["cost_saved_per_day_mean"]
                    - agg[("value_density", "power_user")]["cost_saved_per_day_mean"])
    br_line = f"V2 Cardinality\nCollapse Zone\n({vd_pow_hit:.0f}% hit rate)"
    if st_delta is not None:
        br_line += f"\nV3 recovers +\\${st_delta:,.0f}/day"

    texts = [
        (0.25, 0.75, f"Enterprise\nLRU Dominates\n({lru_ent:.0f}% hit rate)",   "#1a5276"),
        (0.75, 0.75, tr_line,                                                    "#784212"),
        (0.25, 0.25, f"Casual\nAll Tied\n({casual_hit:.0f}% — unconstrained)", "#196f3d"),
        (0.75, 0.25, br_line,                                                    "#922b21"),
    ]
    for tx, ty, label, color in texts:
        ax.text(
            tx, ty, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="semibold",
            color=color, linespacing=1.45,
        )

    ax.set_xlabel("Token-Count Variance (σ)", fontsize=12, labelpad=12)
    ax.set_ylabel("Prediction Confidence Variance", fontsize=12, labelpad=12)

    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["Low", "High"], fontsize=10)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Low", "High"], fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.axhline(0.5, color="#555555", linewidth=1.2, linestyle="-")
    ax.axvline(0.5, color="#555555", linewidth=1.2, linestyle="-")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#555555")

    ax.set_title("Policy Selection Decision Matrix", fontsize=13, pad=14)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figure3_failure_modes.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures from", AGG_PATH)
    aggregates = load_aggregates()
    make_figure1(aggregates)
    make_figure2(aggregates)
    make_figure3(aggregates)
    print("Done — all figures saved to", OUT_DIR)
