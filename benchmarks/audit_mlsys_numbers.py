"""
Cross-check the MLSys 2027 submission, and guard its anonymity.

Two jobs. The first is the same one the other audits do: every number in the
prose is checked against the committed run records. While paper/mlsys27/ is
still a copy of the full paper it shares that paper's claim list rather than
duplicating it (see audit_paper_numbers.build_claims); once the workshop
reviews land on Sept 29, 2026 and the content diverges, fork the list here.

The second job has no counterpart in the other audits and is the reason this
file exists early. MLSys 2027 reviews DOUBLE-BLIND and states that submissions
which do not follow the anonymization guidelines "will be rejected without
review" -- no rebuttal, no second look. The failure mode is mechanical: this
paper began as a copy of paper/latex/paper.tex, which carries a real author
block, a real email, and an artifact URL containing the author's name. Any
paste-back from the full paper during the October edits reintroduces one of
them, and the source still compiles and still reads correctly. So the identity
strings are asserted ABSENT here, which is exactly the inversion of
audit_workshop_numbers.py, where the ML for Systems CFP's non-blind review
means the same strings must be PRESENT.

    python benchmarks/audit_mlsys_numbers.py

Exits non-zero if any claim disagrees with the data, if the paper is
de-anonymized, or if the body exceeds the 10-page limit.
"""

import os
import sys

from audit_paper_numbers import build_claims
from paper_audit import ROOT, Audit, read_tex

PAPER = ('paper', 'mlsys27', 'paper.tex')

# MLSys 2027: 10 pages excluding references, unlimited appendix.
PAGE_LIMIT = 10

# Every string that would identify the author. Sourced from the full paper's
# title block and artifact statement -- the three places a copy reintroduces.
IDENTITY = ["Aditya Lawankar",
            "lawankaraditya",
            "github.com/aditya-lawankar",
            "aditya-lawankar"]


def main():
    text = read_tex(*PAPER)
    audit = Audit()

    audit.section("Prose claims vs committed result data")
    claims = build_claims()
    if not claims:
        print("  [--] no result data found; run the matching `make reproduce-*` targets")
        return 2
    audit.check_all(claims)

    # -- Anonymity. A failure here is a desk reject, not a review comment. ---
    audit.section("Double-blind anonymity (MLSys rejects non-anonymized WITHOUT review)")
    audit.require_text(text,
                       must_appear=["Anonymous Author(s)"],
                       must_not_appear=IDENTITY)

    audit.section("Page limit")
    audit.page_limit(os.path.join(ROOT, 'paper', 'mlsys27', 'paper.pdf'), PAGE_LIMIT)

    return audit.report("anonymity and page-limit checks clean")


if __name__ == '__main__':
    sys.exit(main())
