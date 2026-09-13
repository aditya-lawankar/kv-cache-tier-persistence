# MLSys 2027 submission

**Deadline: October 30, 2026, 20:00 UTC.** Notification February 28, 2027.
Conference in Bellevue, WA, May 2027.

This directory started as a copy of `paper/latex/paper.tex` (17pp) with
anonymization applied. It is a scaffold, not a finished submission.

## Why this venue

Of the four archival deadlines open after the NeurIPS ML for Systems submission
(ASPLOS Sept 9, FAST Sept 15, EuroSys Sept 24, MLSys Oct 30), the review windows
all overlap, so only one can be used per cycle. MLSys is the only one whose
deadline falls *after* the workshop reviews land on Sept 29, 2026, and its
format — 10 pages excluding references, **unlimited appendix** — suits a paper
whose supporting material (cost-model validation, admission control, capacity
sweep) is what reviewers ask about.

## Rules that bite

- **Double-blind, strictly.** Submissions that do not follow the anonymization
  guidelines are "rejected without review". `make audit` asserts the author
  name, email, and repo URL are absent. This is the exact inverse of the
  workshop audit, where the non-blind ML for Systems CFP requires them present —
  which is why a paste-back from `paper/latex/` is the failure mode to fear.
- **Workshop disclosure is mandatory.** MLSys permits expanded workshop papers
  "subject to approval by the program chairs": email them the ML for Systems
  version and explain the added novelty. Do this when submitting, not after.
- **arXiv is fine.** Posting a preprint does not violate the dual-submission
  policy.
- **One archival venue at a time.** Submitting here forecloses FAST, ASPLOS and
  EuroSys until this decision returns on Feb 28, 2027.

## Checklist

- [ ] **Sept 29** — workshop reviews arrive. Fold them in *before* cutting, so
      the trimming happens on the answered version of the paper.
- [ ] Swap in the official MLSys 2027 `.sty` once posted. The preamble is still
      the USENIX-ish one inherited from the full paper.
- [ ] Cut body to 10pp. This is a *move*, not a deletion — the appendix is
      unlimited, so supporting sections relocate rather than disappear.
- [ ] Replace the withheld artifact URL with an `anonymous.4open.science` mirror.
- [ ] Email the chairs the workshop paper + novelty statement.
- [ ] `make mlsys` — builds, audits numbers, checks anonymity and page limit.
- [ ] `make mlsys-bundle` — zips a flat, self-contained submission.

## Build

```bash
make mlsys
```

Numbers are checked against the same committed run records as the full paper;
while the prose is still shared, `benchmarks/audit_mlsys_numbers.py` reuses
`audit_paper_numbers.build_claims()` rather than copying the list. When the
content diverges after the reviews, fork that list — deliberately, in one place.
