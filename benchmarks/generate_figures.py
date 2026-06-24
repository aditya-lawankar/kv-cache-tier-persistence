"""
Generate publication-quality figures for KV Cache Eviction Policy paper.

Produces:
  figure1_hit_rate.png      – Grouped bar chart of cache hit rates
  figure2_cost_savings.png  – Grouped bar chart of daily GPU cost savings
  figure3_failure_modes.png – 2×2 conceptual decision matrix
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Output directory ────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "figures")
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

# ── Data ────────────────────────────────────────────────────────────────
policies = ["LRU", "Heuristic", "Logistic V1", "Value\nDensity", "Value\nDensity AC"]
policies_flat = ["LRU", "Heuristic", "Logistic V1", "Value Density", "Value Density AC"]

hit_enterprise = np.array([97.09, 61.81, 79.55, 70.14, 70.14])
hit_power      = np.array([92.64, 69.26, 87.45, 69.26, 69.26])

cost_enterprise = np.array([1037.71, 660.69, 850.29, 749.70, 749.70])
cost_power      = np.array([3446.89, 2831.56, 3515.24, 3302.79, 3302.79])

# 95 % CI  →  symmetric error = (upper − lower) / 2
ci_ent = np.array([
    [96.21, 97.90],
    [59.24, 64.25],
    [77.45, 81.52],
    [67.77, 72.45],
    [67.77, 72.45],
])
ci_pow = np.array([
    [89.18, 96.10],
    [63.20, 74.89],
    [82.68, 91.34],
    [63.64, 75.76],
    [63.64, 75.76],
])

err_ent = (ci_ent[:, 1] - ci_ent[:, 0]) / 2
err_pow = (ci_pow[:, 1] - ci_pow[:, 0]) / 2

x = np.arange(len(policies))
bar_w = 0.34


# ════════════════════════════════════════════════════════════════════════
# Figure 1 – Hit Rate Comparison
# ════════════════════════════════════════════════════════════════════════
def make_figure1():
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars1 = ax.bar(
        x - bar_w / 2, hit_enterprise, bar_w,
        yerr=err_ent, capsize=4,
        color=C_ENTERPRISE, edgecolor="white", linewidth=0.6,
        label="Enterprise", error_kw=dict(lw=1.0, capthick=1.0),
    )
    bars2 = ax.bar(
        x + bar_w / 2, hit_power, bar_w,
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
    ax.set_xticklabels(policies)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Cache Hit Rate by Policy and Workload (500 MB, 6 hr)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")

    # Clean up: only horizontal grid, no x-axis grid
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
def make_figure2():
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars1 = ax.bar(
        x - bar_w / 2, cost_enterprise, bar_w,
        color=C_ENTERPRISE, edgecolor="white", linewidth=0.6,
        label="Enterprise",
    )
    bars2 = ax.bar(
        x + bar_w / 2, cost_power, bar_w,
        color=C_POWERUSER, edgecolor="white", linewidth=0.6,
        label="Power User",
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

    # Key-finding annotation: Logistic V1 power-user bar
    logistic_idx = 2
    target_x = x[logistic_idx] + bar_w / 2
    target_y = cost_power[logistic_idx]
    ax.annotate(
        "Higher $/Day\ndespite lower hit rate",
        xy=(target_x, target_y),
        xytext=(target_x + 1.05, target_y + 250),
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
    ax.set_xticklabels(policies)
    ax.set_ylabel("$/Day Saved (USD)")
    ax.set_ylim(0, 4400)
    ax.set_title("Daily GPU Cost Savings by Policy and Workload")
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
def make_figure3():
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Quadrant colours (muted pastels)
    colors = {
        "TL": "#d6eaf8",  # light blue
        "TR": "#fdebd0",  # light peach
        "BL": "#d5f5e3",  # light green
        "BR": "#f5b7b1",  # light coral
    }

    # Draw quadrants as filled rectangles
    quadrants = [
        # (x0, y0, width, height, key)
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

    # Quadrant text
    texts = [
        (0.25, 0.75, "Enterprise\nLRU Dominates\n(97% hit rate)",             "#1a5276"),
        (0.75, 0.75, "Power User\nV1 wins on \$/Day\n(\$3515 vs \$3447)",       "#784212"),
        (0.25, 0.25, "Casual\nAll Tied\n(100% \u2014 unconstrained)",              "#196f3d"),
        (0.75, 0.25, "V2 Cardinality\nCollapse Zone\n(69% hit rate)",         "#922b21"),
    ]
    for tx, ty, label, color in texts:
        ax.text(
            tx, ty, label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="semibold",
            color=color, linespacing=1.45,
        )

    # Axis labels (placed outside the unit square)
    ax.set_xlabel("Token-Count Variance (σ)", fontsize=12, labelpad=12)
    ax.set_ylabel("Prediction Confidence Variance", fontsize=12, labelpad=12)

    # Custom ticks to label Low / High
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["Low", "High"], fontsize=10)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Low", "High"], fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Divider lines
    ax.axhline(0.5, color="#555555", linewidth=1.2, linestyle="-")
    ax.axvline(0.5, color="#555555", linewidth=1.2, linestyle="-")

    # Remove grid (conceptual diagram, not a data chart)
    ax.grid(False)

    # Border
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
    print("Generating figures...")
    make_figure1()
    make_figure2()
    make_figure3()
    print("Done — all figures saved to", OUT_DIR)
