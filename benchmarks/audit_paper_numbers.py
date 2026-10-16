"""
Cross-check every number that appears in the paper's prose against the committed
result files, and check the paper's tables against those same files.

This exists because the paper's remaining error class is arithmetic hygiene: a
figure quoted in prose that no longer matches the table it came from. Those
errors survive proofreading because each sentence is locally plausible. They do
not survive a diff against the source data.

    python benchmarks/audit_paper_numbers.py

Exits non-zero if any checked claim disagrees with the data.
"""

import sys

import numpy as np

from paper_audit import Audit, load, load_script, read_tex, summarize

PAPER = ('paper', 'latex', 'paper.tex')


def build_claims():
    """Every numeric claim the full paper makes, paired with the data behind it."""
    # ---- sources -----------------------------------------------------------
    runs10 = load('benchmarks/results/arch_tinyllama*/shard_*/'
                  'experiment_results_v3_raw.json', seeds=set(range(42, 52)))
    main10 = summarize(runs10)
    lru10 = [r for (p, _), r in runs10.items() if p == 'lru']
    two = summarize(load('benchmarks/results/twotier/shard_*/experiment_results_v3_raw.json',
                         seeds=set(range(42, 52))))
    azure = summarize(load('benchmarks/results/azure_v2/*/experiment_results_azure_raw.json',
                           workload='azure_conv'))
    oracle = summarize(load('benchmarks/results/oracle_v2/shard_*/'
                            'experiment_results_v3_raw.json', seeds=set(range(42, 52))))

    # ---- claims made in prose, each tied to its source ---------------------
    # (label, claimed value as written, actual value, tolerance)
    claims = [
        ("main: LRU hit%", 79.1, main10['lru'][0], 0.05),
        ("main: LRU GPU-s/day", 36.8, main10['lru'][1], 0.05),
        ("main: V1 hit%", 67.9, main10['logistic_v1'][0], 0.05),
        ("main: V2 hit%", 58.2, main10['value_density'][0], 0.05),
        ("main: V2 GPU-s/day", 73.2, main10['value_density'][1], 0.05),
        ("main: V2 ratio (abstract/9.3 '1.99x')", 1.99, main10['value_density'][2], 0.005),
        ("main: V3 ratio ('1.97x')", 1.97, main10['space_time'][2], 0.005),
        ("main: V1 ratio ('1.73x')", 1.73, main10['logistic_v1'][2], 0.005),
        ("main: heuristic ratio ('1.65x')", 1.65, main10['heuristic'][2], 0.005),
        # Gaps quoted in prose are differences of the ROUNDED figures the table
        # prints, which is what a reader can verify; checking them against
        # unrendered data would flag correct text.
        ("9.3: LRU-V1 hit gap ('11.2 points')", 11.2,
         round(main10['lru'][0], 1) - round(main10['logistic_v1'][0], 1), 0.05),
        ("9.3: LRU-V2 hit gap (was 21.3)", 20.9,
         round(main10['lru'][0], 1) - round(main10['value_density'][0], 1), 0.05),
        # The paper prints these mean hit counts as integers, so half a hit is rounding.
        ("reversal: LRU cold-tier hits ('812')", 812,
         float(np.mean([r['hits_by_tier']['cold'] for r in lru10])), 0.5),
        ("reversal: LRU total hits ('1,237')", 1237,
         float(np.mean([r['total_hits'] for r in lru10])), 0.5),
        ("two-tier: LRU GPU-s/day ('131.0')", 131.0, two['lru'][1], 0.1),
        ("two-tier: LRU hit% ('79.4')", 79.4, two['lru'][0], 0.05),
        ("two-tier: V3 ratio ('1.15x')", 1.15, two['space_time'][2], 0.005),
        ("two-tier: provisioning gain ('3.6x')", 3.6, two['lru'][1] / main10['lru'][1], 0.05),
        ("azure: LRU hit% ('83.9')", 83.9, azure['lru'][0], 0.05),
        ("azure: LRU GPU-s/day ('32.1')", 32.1, azure['lru'][1], 0.05),
        ("azure: V2 ratio ('2.18x')", 2.18, azure['value_density'][2], 0.005),
        ("oracle: Belady GPU-s/day ('209.8')", 209.8, oracle['belady'][1], 0.1),
        ("oracle: headroom ('5.70x')", 5.70, oracle['belady'][2], 0.01),
        ("oracle: LRU-oracle hit gap ('20.9 points')", 20.9,
         100.0 - round(oracle['lru'][0], 1), 0.05),
    ]

    return claims


def main():
    text = read_tex(*PAPER)
    audit = Audit()

    audit.section("Prose claims vs committed result data")
    audit.check_all(build_claims())

    # ---- claims that must also literally appear in the text ---------------
    audit.section("Text assertions")
    must_appear = ["20.9 over Value~Density~V2", "1.99$\\times$", "131.0", "3.6$",
                   "812 of its 1,237"]
    must_not_appear = ["21.3 over Value", "include two oracles", "1{,}800 GPU-hours",
                       "figure3_failure_modes", "static value-density collapses in practice",
                       # Seed 42's count, not the ten-seed mean every other figure uses.
                       "778 of its 1,206",
                       # The two residual estimates (1.07-1.08x, 1.13-1.15x) differ by 5-8 points.
                       "within two points",
                       # The cost-model defect was not caught by a reader.
                       "a reader asked",
                       # Value is modeled, not observed.
                       "actually saved"]
    audit.require_text(text, must_appear, must_not_appear)

    # ---- figures must plot the same values the tables print ---------------
    # The prose audit above cannot see figures. Figure 2 once plotted values
    # derived from cent-rounded dollars, which quantized every bar to 14.4
    # GPU-s and collapsed two workloads onto identical bars.
    audit.section("Figure data vs table data")
    gf = load_script('generate_figures.py')
    fagg = gf.load_aggregates()
    forder, _ = gf._available_policies(fagg)
    fraw = gf.load_raw()
    for wl in ("enterprise", "power_user"):
        _, _, plotted, _, _ = gf._series(fagg, wl, forder)
        for i, pol in enumerate(forder):
            truth = float(np.mean([x["gpu_hours_saved_per_day"] * 3600
                                   for x in fraw[(pol, wl)]]))
            ok = abs(plotted[i] - truth) <= 0.1
            audit.flag(f"figure2 {pol}/{wl}", ok,
                   f"  plot={plotted[i]:<8.1f} data={truth:<8.1f}")

    # ---- mechanical damage that has broken silently before ----------------
    audit.section("Mechanical checks")
    mech = [("no tab characters", chr(9) not in text),
            ("no stray table commas", "& ,  " not in text),
            ("no broken times macros",
             not [i for i in range(len(text) - 5)
                  if text[i:i+5] == 'imes$' and text[i-1] != 't']),
            ("no currency in GPU-second figure",
             'f"$' not in read_tex('benchmarks', 'generate_figures.py'))]
    for label, ok in mech:
        audit.flag(label, ok)

    return audit.report("text and mechanical checks clean")


if __name__ == '__main__':
    sys.exit(main())
