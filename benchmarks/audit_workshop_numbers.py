"""
Cross-check every number in the 4-page workshop paper against the committed
result files.

The workshop paper (paper/workshop/paper.tex) is a condensed selection of the
full paper's results, so its remaining error class is transcription: a figure
copied across during compression that no longer matches the run it came from.
Those errors survive proofreading because each sentence is locally plausible.
They do not survive a diff against the source data.

It also enforces the 4-page body limit, which is the other thing that breaks
silently: an edit that pushes the conclusion onto page 5 looks fine in the
source and is a desk reject at the venue.

    python benchmarks/audit_workshop_numbers.py

Exits non-zero if any checked claim disagrees with the data. Like the companion
script benchmarks/audit_paper_numbers.py, which does the same job for the full
paper, this reads per-shard result directories that .gitignore excludes, so it
needs a working copy where the experiments have actually been run.
"""

import json
import os
import sys

import numpy as np

from paper_audit import (ROOT, Audit, gpu_s, load, load_script, paired_delta,
                         read_tex, summarize)

PAPER = ('paper', 'workshop', 'paper.tex')


def main():
    text = read_tex(*PAPER)
    audit = Audit()

    ten = set(range(42, 52))
    tiny = load('benchmarks/results/arch_tinyllama*/shard_*/'
                'experiment_results_v3_raw.json', seeds=ten)
    tiny_pu = load('benchmarks/results/arch_tinyllama*/shard_*/'
                   'experiment_results_v3_raw.json', workload='power_user', seeds=ten)
    l70 = load('benchmarks/results/arch_llama70b/shard_*/'
               'experiment_results_v3_raw.json', seeds=ten)
    two = load('benchmarks/results/twotier/shard_*/'
               'experiment_results_v3_raw.json', seeds=ten)
    azure = load('benchmarks/results/azure_v2/*/'
                 'experiment_results_azure_raw.json', workload='azure_conv')
    oracle = load('benchmarks/results/oracle_v2/shard_*/'
                  'experiment_results_v3_raw.json', seeds=ten)
    oracle_pu = load('benchmarks/results/oracle_v2/shard_*/'
                     'experiment_results_v3_raw.json', workload='power_user', seeds=ten)

    m, m_pu = summarize(tiny), summarize(tiny_pu)
    a, t, o, o_pu = summarize(azure), summarize(two), summarize(oracle), summarize(oracle_pu)
    s70 = summarize(l70)

    # Several of these live in per-shard directories that .gitignore excludes, so
    # a fresh clone has the paper but not the runs behind it. Say so plainly
    # rather than dying on a KeyError halfway through the claim list.
    missing = [name for name, stats in [('tinyllama', m), ('tinyllama power-user', m_pu),
                                        ('llama-2-70b', s70), ('two-tier', t),
                                        ('azure', a), ('oracle', o)] if not stats]
    if missing:
        print("Missing result data for: %s" % ', '.join(missing))
        print("These runs live in per-shard directories that are not committed; "
              "regenerate them with the matching `make reproduce-*` target.")
        return 2

    audit.section("Prose and table claims vs committed result data")

    # -- Section 3: the reversal, synthetic and real ------------------------
    claims = [
        ("S3: LRU hit% enterprise ('79.1')", 79.1, m['lru'][0], 0.05),
        ("S3: V1 hit% enterprise ('67.9')", 67.9, m['logistic_v1'][0], 0.05),
        ("S3: LRU hit% power-user ('96.0')", 96.0, m_pu['lru'][0], 0.05),
        ("S3: V1 hit% power-user ('83.9')", 83.9, m_pu['logistic_v1'][0], 0.05),
        ("S3: V2 ratio ('1.99x')", 1.99, m['value_density'][2], 0.005),
        ("S3: V3 ratio ('1.97x')", 1.97, m['space_time'][2], 0.005),
        ("S3: V1 ratio ('1.73x')", 1.73, m['logistic_v1'][2], 0.005),
        ("S3: heuristic ratio ('1.65x')", 1.65, m['heuristic'][2], 0.005),
        ("S3: V2 paired delta ('+36.4')", 36.4, paired_delta(tiny, 'value_density'), 0.05),
        ("S3: azure LRU hit% ('83.9')", 83.9, a['lru'][0], 0.05),
        ("S3: azure V2 ratio ('2.18x')", 2.18, a['value_density'][2], 0.005),
        ("S3: azure V2 paired delta ('+37.9')", 37.9,
         paired_delta(azure, 'value_density'), 0.05),
        # Gaps quoted in prose are differences of the ROUNDED figures the paper
        # prints, which is what a reader can verify.
        ("S3: azure LRU-V2 hit gap ('16.9 points')", 16.9,
         round(a['lru'][0], 1) - round(a['value_density'][0], 1), 0.05),
    ]

    # -- Section 4: table cells, both controls ------------------------------
    # The heuristic row was dropped from Table 1 for space; its margin is
    # quoted in S3 instead and checked there.
    for pol, hit, val in [('lru', 79.1, 36.8),
                          ('logistic_v1', 67.9, 63.5), ('value_density', 58.2, 73.2),
                          ('space_time', 57.8, 72.4)]:
        claims.append(("T1 tiny-3tier %s hit%%" % pol, hit, m[pol][0], 0.05))
        claims.append(("T1 tiny-3tier %s GPU-s/d" % pol, val, m[pol][1], 0.05))
    for pol, hit, val in [('lru', 79.1, 8173),
                          ('logistic_v1', 67.9, 8806), ('value_density', 67.1, 8722),
                          ('space_time', 63.2, 8783)]:
        claims.append(("T1 70b-3tier %s hit%%" % pol, hit, s70[pol][0], 0.05))
        claims.append(("T1 70b-3tier %s GPU-s/d" % pol, val, s70[pol][1], 0.5))
    for pol, hit, val in [('lru', 79.4, 131.0),
                          ('logistic_v1', 68.0, 150.7), ('value_density', 58.5, 147.4),
                          ('space_time', 58.1, 150.0)]:
        claims.append(("T1 tiny-2tier %s hit%%" % pol, hit, t[pol][0], 0.05))
        claims.append(("T1 tiny-2tier %s GPU-s/d" % pol, val, t[pol][1], 0.05))

    best70 = max(s70[p][2] for p in s70 if p != 'lru')
    claims += [
        ("S4: 70B best ratio ('1.08x')", 1.08, best70, 0.005),
        ("S4: 2-tier V1 ratio ('1.15x')", 1.15, t['logistic_v1'][2], 0.005),
        ("S4: 2-tier V3 ratio ('1.15x')", 1.15, t['space_time'][2], 0.005),
        ("S4: 2-tier V1 paired delta ('+19.7')", 19.7,
         paired_delta(two, 'logistic_v1'), 0.05),
        ("S4: 2-tier V3 paired delta ('+19.0')", 19.0,
         paired_delta(two, 'space_time'), 0.05),
        ("S4: provisioning gain ('3.6x')", 3.6, t['lru'][1] / m['lru'][1], 0.05),
        ("S4: 2-tier LRU vs best 3-tier ('1.8x')", 1.8,
         t['lru'][1] / max(m[p][1] for p in m), 0.05),
        ("S4: 2-tier LRU-V1 hit gap ('11 points')", 11,
         round(t['lru'][0], 1) - round(t['logistic_v1'][0], 1), 0.5),
    ]

    # -- Section 5: oracles -------------------------------------------------
    claims += [
        ("S5: Belady hit% ('100.0')", 100.0, o['belady'][0], 0.05),
        ("S5: OracleV3 hit% ('100.0')", 100.0, o['oracle_v3'][0], 0.05),
        ("S5: LRU-oracle hit gap ('20.9 points')", 20.9, 100.0 - round(o['lru'][0], 1), 0.05),
        ("S5: oracle headroom ('5.70x')", 5.70, o['belady'][2], 0.01),
        ("S5: OracleV3-Belady enterprise ('+2.9')", 2.9,
         o['oracle_v3'][1] - o['belady'][1], 0.05),
        ("S5: OracleV3-Belady power-user ('-0.5')", -0.5,
         o_pu['oracle_v3'][1] - o_pu['belady'][1], 0.05),
        ("S5: tier-placement effect ('1.4%')", 1.4,
         100 * (o['oracle_v3'][1] / o['belady'][1] - 1), 0.05),
    ]

    # -- Admission control (App. B) and the cost-model validation (App. A) ---
    ac = load('benchmarks/results/arch_tinyllama_ac/shard_*/'
              'experiment_results_v3_raw.json', seeds=ten)
    ac_az = {k: v for k, v in azure.items() if k[0] == 'value_density_ac'}
    ac_stats, ac_az_stats = summarize(ac), summarize(azure)

    def paired_between(a, b, pa, pb):
        """Mean paired delta of pb (in b) minus pa (in a) over shared seeds."""
        seeds_ = sorted({s for (q, s) in a if q == pa} & {s for (q, s) in b if q == pb})
        return float(np.mean([gpu_s(b[(pb, s)]) - gpu_s(a[(pa, s)]) for s in seeds_]))

    claims += [
        ("AppB: V2+AC hit% enterprise", 58.2, ac_stats['value_density_ac'][0], 0.05),
        ("AppB: V2+AC GPU-s/d enterprise", 68.4, ac_stats['value_density_ac'][1], 0.05),
        ("AppB: V2+AC minus V2, enterprise ('-4.8')", -4.8,
         paired_between(tiny, ac, 'value_density', 'value_density_ac'), 0.05),
        ("AppB: V2+AC hit% azure", 66.8, ac_az_stats['value_density_ac'][0], 0.05),
        ("AppB: V2+AC GPU-s/d azure", 63.9, ac_az_stats['value_density_ac'][1], 0.05),
        ("AppB: V2+AC minus V2, azure ('-6.1')", -6.1,
         paired_between(azure, azure, 'value_density', 'value_density_ac'), 0.05),
    ]
    assert ac_az  # azure V2+AC runs must be present for the two rows above

    sweep_path = os.path.join(ROOT, 'benchmarks', 'results', 'gpu_sweep',
                              'context_sweep_llama32_t4.json')
    if os.path.exists(sweep_path):
        sweep = json.load(open(sweep_path))
        ratios = [r['cold_ttft_ms'] / r['predicted_prefill_ms'] for r in sweep['rows']]
        claims += [
            ("AppA: min predicted/measured ratio ('1.15x')", 1.15, min(ratios), 0.005),
            ("AppA: max predicted/measured ratio ('1.66x')", 1.66, max(ratios), 0.005),
            ("AppA: cold prefill growth ('14.8x')", 14.8,
             sweep['rows'][-1]['cold_ttft_ms'] / sweep['rows'][0]['cold_ttft_ms'], 0.05),
            ("AppA: warm path growth ('1.8x')", 1.8,
             sweep['rows'][-1]['warm_path_ms'] / sweep['rows'][0]['warm_path_ms'], 0.05),
        ]
        for row, meas, pred in [(0, 111, 67), (1, 157, 137), (2, 340, 288),
                                (3, 878, 632), (4, 1645, 1032)]:
            r = sweep['rows'][row]
            claims.append(("AppA: %d-token measured" % r['tokens'], meas,
                           r['cold_ttft_ms'], 0.5))
            claims.append(("AppA: %d-token predicted" % r['tokens'], pred,
                           r['predicted_prefill_ms'], 0.5))

    # -- Section 6: persona-strength sweep quoted in Limitations ------------
    persona = {sig: summarize(load('benchmarks/results/persona_sweep/sigma_%s/'
                                   'experiment_results_v3_raw.json' % sig))
               for sig in ('0.0', '1.0')}
    claims += [
        ("S6: V3 ratio at sigma=0 ('1.34x')", 1.34, persona['0.0']['space_time'][2], 0.005),
        ("S6: V3 ratio at sigma=1 ('2.25x')", 2.25, persona['1.0']['space_time'][2], 0.005),
        ("S6: V1 ratio at sigma=0 ('0.90x')", 0.90, persona['0.0']['logistic_v1'][2], 0.005),
    ]

    # -- Break-even lengths quoted in Section 4 -----------------------------
    sys.path.insert(0, os.path.join(ROOT, 'benchmarks'))
    import breakeven_analysis as be
    models = {m_.name: m_ for m_ in be.MODELS}
    tiers = {t_.name: t_ for t_ in be.STORAGE_TIERS}
    for model_key, tier_key, expected in [
            ('TinyLlama-1.1B', 'NVMe', 1561), ('TinyLlama-1.1B', 'S3', 25901),
            ('Llama-2-7B', 'NVMe', 43559), ('Llama-2-7B', 'S3', 254203),
            ('Llama-2-70B', 'NVMe', 12), ('Llama-2-70B', 'S3', 289)]:
        model = next(v for k, v in models.items() if k.startswith(model_key))
        tier = next(v for k, v in tiers.items() if k.startswith(tier_key))
        claims.append(("S4: N* %s/%s" % (model_key, tier_key), expected,
                       be.crossover_tokens(model, tier, be.A100), 1.0))

    audit.check_all(claims)

    # -- Claims that must literally appear, and retracted ones that must not -
    audit.section("Text assertions")
    must_appear = ["$1.99\\times$", "131.0", "factor of $3.6$",
                   "100.0\\% against 100.0\\%", "778 of its 1{,}206", "$5.70\\times$",
                   # The ML for Systems CFP says review is non-blind, so the paper
                   # must render its real author block. dblblindworkshop would
                   # silently replace it with "Anonymous Author(s)".
                   "[sglblindworkshop]{neurips_2026}",
                   "Aditya Lawankar",
                   # NOT `final`: the official template reserves that for camera-
                   # ready, so a paper under review stays in submission mode.
                   # Submission mode loads lineno, which is why hyperref needs
                   # [draft]. Numeric natbib keeps \cite from reading as a
                   # sentence fragment.
                   "\\usepackage[draft]{hyperref}",
                   "\\PassOptionsToPackage{numbers,compress}{natbib}",
                   # Reviewer-facing content that must not silently vanish.
                   "github.com/aditya-lawankar/kv-cache-tier-persistence",
                   "cao1997greedydualsize",
                   "\\section{Cost-model validation}",
                   "\\section{Admission control}"]
    must_not_appear = ["static value density collapses in practice",
                       "48$\\times$", "19.1$\\times$",
                       "\\section{Conclusion}",  # folded into S6 to hold 4 pages
                       "dblblindworkshop",
                       # `final` is camera-ready only per the official template;
                       # a paper under review must stay in submission mode.
                       "sglblindworkshop, final",
                       # The abstract must not claim the decision rule beats
                       # every policy; only the provisioning step is evidenced.
                       "outperforms every policy we built"]
    audit.require_text(text, must_appear, must_not_appear)

    # -- Figure 1 must plot the values the prose quotes ---------------------
    audit.section("Figure data vs prose")
    gwf = load_script('generate_workshop_figure.py')
    for name, path, workload, truth in [
            ('synthetic', 'experiment_results_v3_raw.json', 'enterprise', m),
            ('azure', 'experiment_results_azure_raw.json', 'azure_conv', a)]:
        hit, _, val, _ = gwf.series(
            os.path.join(ROOT, 'benchmarks', 'results', path), workload)
        for i, pol in enumerate(gwf.ORDER):
            ok = (abs(hit[i] - truth[pol][0]) <= 0.05
                  and abs(val[i] - truth[pol][1]) <= 0.05)
            audit.flag('fig1 %-9s %-14s' % (name, pol), ok,
                   ' plot=%.1f/%.1f  data=%.1f/%.1f'
                   % (hit[i], val[i], truth[pol][0], truth[pol][1]))

    # -- The page limit is the other thing that silently breaks -------------
    audit.section("Page limit")
    audit.page_limit(os.path.join(ROOT, 'paper', 'workshop', 'paper.pdf'), 4)

    return audit.report("text, figure, and page-limit checks clean")


if __name__ == '__main__':
    sys.exit(main())
