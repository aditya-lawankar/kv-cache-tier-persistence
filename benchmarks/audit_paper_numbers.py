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

import glob
import io
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'paper', 'latex', 'paper.tex')


def gpu_s(run):
    return run['gpu_hours_saved_per_day'] * 3600


def load(pattern, workload='enterprise', seeds=None):
    out = {}
    for path in glob.glob(os.path.join(ROOT, pattern)):
        for r in json.load(open(path)):
            if r['workload'] != workload:
                continue
            if seeds is not None and r['seed'] not in seeds:
                continue
            out[(r['policy'], r['seed'])] = r
    return out


def summarize(runs):
    """policy -> (mean hit rate %, mean GPU-s/day, ratio vs LRU)."""
    pols = sorted({p for p, _ in runs})
    seeds = sorted({s for _, s in runs})
    stats = {}
    for p in pols:
        rs = [runs[(p, s)] for s in seeds if (p, s) in runs]
        if not rs:
            continue
        stats[p] = [float(np.mean([r['hit_rate'] for r in rs]) * 100),
                    float(np.mean([gpu_s(r) for r in rs]))]
    lru = stats.get('lru', [None, None])[1]
    for p in stats:
        stats[p].append(stats[p][1] / lru if lru else float('nan'))
    return stats


def check(label, claimed, actual, tol):
    ok = abs(claimed - actual) <= tol
    print(f"  [{'OK ' if ok else 'BAD'}] {label:<52} paper={claimed:<9.4g} data={actual:<9.4g}")
    return ok


def main():
    text = io.open(PAPER, encoding='utf-8').read()
    failures = []

    # ---- sources -----------------------------------------------------------
    main10 = summarize(load('benchmarks/results/arch_tinyllama*/shard_*/'
                            'experiment_results_v3_raw.json', seeds=set(range(42, 52))))
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

    print("Prose claims vs committed result data")
    print("=" * 78)
    for label, claimed, actual, tol in claims:
        if not check(label, claimed, actual, tol):
            failures.append(label)

    # ---- claims that must also literally appear in the text ---------------
    print("\nText assertions")
    print("=" * 78)
    must_appear = ["20.9 over Value~Density~V2", "1.99$\\times$", "131.0", "3.6$"]
    must_not_appear = ["21.3 over Value", "include two oracles", "1{,}800 GPU-hours",
                       "figure3_failure_modes", "static value-density collapses in practice"]
    for frag in must_appear:
        ok = frag in text
        print(f"  [{'OK ' if ok else 'BAD'}] present: {frag}")
        if not ok:
            failures.append('missing: ' + frag)
    for frag in must_not_appear:
        ok = frag not in text
        print(f"  [{'OK ' if ok else 'BAD'}] absent:  {frag}")
        if not ok:
            failures.append('present: ' + frag)

    # ---- mechanical damage that has broken silently before ----------------
    print("\nMechanical checks")
    print("=" * 78)
    mech = [("no tab characters", chr(9) not in text),
            ("no stray table commas", "& ,  " not in text),
            ("no broken times macros",
             not [i for i in range(len(text) - 5)
                  if text[i:i+5] == 'imes$' and text[i-1] != 't']),
            ("no currency in GPU-second figure",
             'f"$' not in io.open(os.path.join(ROOT, 'benchmarks', 'generate_figures.py'),
                                  encoding='utf-8').read())]
    for label, ok in mech:
        print(f"  [{'OK ' if ok else 'BAD'}] {label}")
        if not ok:
            failures.append(label)

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL: {len(failures)} discrepancies")
        for f in failures:
            print("   -", f)
        return 1
    print(f"PASS: {len(claims)} numeric claims agree with the data; "
          "text and mechanical checks clean")
    return 0


if __name__ == '__main__':
    sys.exit(main())
