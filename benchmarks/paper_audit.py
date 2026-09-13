"""
Shared machinery for the per-venue paper audits.

Each venue keeps its own audit script, because each venue makes different
claims and enforces different formatting rules: audit_paper_numbers.py covers
the full paper in paper/latex/, audit_workshop_numbers.py the 4-page ML for
Systems version in paper/workshop/. What does *not* differ between them is the
machinery -- loading committed run records, summarizing them per policy,
comparing a claimed figure against the data, asserting that required text is
present, and checking the body respects a page limit. Before this module those
five things existed twice, verbatim, and a third venue would have made three.

The formatting assertions are the reason this is worth factoring rather than
copying. The workshop paper is reviewed NON-blind, so its audit asserts the
real author block is present. Every archival venue on the shortlist (FAST,
ASPLOS, EuroSys, MLSys) is double-blind, where that same assertion inverts: the
name must be ABSENT. Copying the workshop script to seed a venue variant would
carry the wrong polarity across silently, and anonymization would fail review.
Against this module that is one argument.

Adding a venue: build the claims list against `load`/`summarize`, then

    a = Audit()
    a.section("Prose claims vs committed result data")
    a.check_all(claims)
    a.require_text(read_tex('paper', 'fast27', 'paper.tex'),
                   must_appear=[...], must_not_appear=["Aditya Lawankar"])
    a.section("Page limit")
    a.page_limit(os.path.join(ROOT, 'paper', 'fast27', 'paper.pdf'), 12)
    return a.report("text and page-limit checks clean")
"""

import glob
import importlib.util
import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RULE = "=" * 78


# --- loading committed run records ------------------------------------------

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
    """policy -> [mean hit rate %, mean GPU-s/day, ratio vs LRU]."""
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


def paired_delta(runs, policy):
    """Mean paired GPU-s/day delta of `policy` against LRU, over shared seeds."""
    lru = {s: r for (p, s), r in runs.items() if p == 'lru'}
    deltas = [gpu_s(r) - gpu_s(lru[s])
              for (p, s), r in runs.items() if p == policy and s in lru]
    return float(np.mean(deltas))


def read_tex(*parts):
    """Read a source file under the repo root, e.g. read_tex('paper','latex','paper.tex')."""
    return io.open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def load_script(name):
    """Import a benchmarks/ script by filename so an audit can reuse its plotting
    functions and compare what a figure PLOTS against what the data says."""
    path = os.path.join(ROOT, 'benchmarks', name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(name)[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the audit itself --------------------------------------------------------

class Audit:
    """Accumulates pass/fail results and renders the report both scripts print."""

    def __init__(self):
        self.failures = []
        self.n_claims = 0
        self._opened = False

    def section(self, title):
        print(("\n" if self._opened else "") + title)
        print(_RULE)
        self._opened = True

    def flag(self, label, ok, detail=''):
        """Record a boolean check that has no claimed-vs-actual pair."""
        print("  [%s] %s%s" % ('OK ' if ok else 'BAD', label, detail))
        if not ok:
            self.failures.append(label)
        return ok

    def check(self, label, claimed, actual, tol):
        """Compare a figure as written in the prose against the data behind it."""
        self.n_claims += 1
        ok = abs(claimed - actual) <= tol
        print("  [%s] %-52s paper=%-9.4g data=%-9.4g"
              % ('OK ' if ok else 'BAD', label, claimed, actual))
        if not ok:
            self.failures.append(label)
        return ok

    def check_all(self, claims):
        """claims: iterable of (label, claimed, actual, tolerance)."""
        for label, claimed, actual, tol in claims:
            self.check(label, claimed, actual, tol)

    def require_text(self, text, must_appear=(), must_not_appear=()):
        """Fragments that must survive an edit, and retracted ones that must not
        come back. Both directions matter: must_not_appear is what keeps a
        withdrawn claim, or a de-anonymizing author name, from reappearing."""
        for frag in must_appear:
            ok = frag in text
            print("  [%s] present: %s" % ('OK ' if ok else 'BAD', frag))
            if not ok:
                self.failures.append('missing: ' + frag)
        for frag in must_not_appear:
            ok = frag not in text
            print("  [%s] absent:  %s" % ('OK ' if ok else 'BAD', frag))
            if not ok:
                self.failures.append('present: ' + frag)

    def page_limit(self, pdf, limit, heading='References'):
        """Enforce a venue's body page limit against the built PDF.

        "`heading` appears on page limit+1" is not sufficient on its own: body
        text can spill onto that page *above* the heading and still satisfy it.
        The limit is met only when the heading starts its page with nothing
        before it, so the spill is measured explicitly.
        """
        if not os.path.exists(pdf):
            print("  [--] %s not built; page count not checked" % os.path.basename(pdf))
            return None
        try:
            import fitz
        except ImportError:
            print("  [--] PyMuPDF not installed; page count not checked")
            return None
        doc = fitz.open(pdf)
        ref_page, spill = None, ''
        for i, pg in enumerate(doc):
            text = pg.get_text()
            at = text.find(heading)
            if at >= 0:
                ref_page, spill = i + 1, text[:at].strip()
                break
        ok = ref_page is not None and ref_page >= limit + 1 and not spill
        print("  [%s] body ends by page %d (%s start on page %s%s)"
              % ('OK ' if ok else 'BAD', limit, heading, ref_page,
                 ', but %d chars of body precede them' % len(spill) if spill else ''))
        if not ok:
            self.failures.append('body exceeds %d pages' % limit)
            if spill:
                # The spill is body prose and may carry maths glyphs the console
                # encoding cannot render; show it lossily.
                preview = spill[:120].encode('ascii', 'replace').decode('ascii')
                print("      spilled body text: %s..." % preview)
        return ok

    def report(self, clean_summary):
        """Print the verdict and return the process exit code."""
        print("\n" + _RULE)
        if self.failures:
            print("FAIL: %d discrepancies" % len(self.failures))
            for f in self.failures:
                print("   -", f)
            return 1
        print("PASS: %d numeric claims agree with the data; %s"
              % (self.n_claims, clean_summary))
        return 0
