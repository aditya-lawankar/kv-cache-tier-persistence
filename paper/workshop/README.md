# 4-page workshop paper

A condensed version of the full paper for the **NeurIPS 2026 Machine Learning for
Systems** workshop, held to the 4-page body limit (references excluded).

This is a **separate deliverable** from the full paper in `paper/latex/` and from the
arXiv bundle built by `make arxiv`; the two share result data and figures scripts but
nothing else. Editing one does not change the other.

| | Full paper | Workshop paper |
|---|---|---|
| Source | `paper/latex/paper.tex` | `paper/workshop/paper.tex` |
| Format | USENIX two-column | NeurIPS single-column (`neurips_2026.sty`) |
| Length | ~15 pages | 4 pages + references |
| Build | `make paper` / `make arxiv` | `make workshop` |
| Audit | `benchmarks/audit_paper_numbers.py` | `benchmarks/audit_workshop_numbers.py` |
| Bundle | `arxiv_bundle.zip` | `workshop_submission.zip` |

## Building

```bash
make workshop         # compile paper.pdf and verify it fits 4 pages
make workshop-bundle  # package sources into workshop_submission.zip
```

## Submission facts (from the CFP, checked 2026-08-26)

- **Deadline:** August 29, 2026, midnight AoE.
- **Notification:** September 29, 2026.
- **Length:** up to 4 pages, *not* counting references or appendices, and the CFP calls
  this "a strict limit" — hence the page check in the audit script.
- **Review:** non-blind. "Submissions do not have to be anonymized."
- **Archival:** no. No formal proceedings, so submitting here does not block arXiv or a
  later conference submission.
- **Submission site:** OpenReview, `NeurIPS.cc/2026/Workshop/MLForSys`.
- **Format:** the NeurIPS 2026 template, which is the `neurips_2026.sty` in this directory.

## Blind mode

Review is non-blind, so `paper.tex` loads
`\usepackage[sglblindworkshop]{neurips_2026}` — that option renders the real author block.

**Deliberately without `final`.** The official template reserves it for camera-ready:
"After being accepted, the authors should add 'final' behind the track to compile a
camera-ready version." Submission mode is correct for a paper under review, and it keeps
the two things a reviewer expects: line numbers, and the "Submitted to 40th Conference…
Do not distribute" banner. Add `, final` only after acceptance.

Do **not** switch to `dblblindworkshop`; that prints "Anonymous Author(s)". The audit
asserts the track option, the absence of `final`, and the author name.

`\workshoptitle{Machine Learning for Systems}` matches the workshop's official name in the
NeurIPS 2026 workshop list.

## Three layout settings the build depends on

- `\PassOptionsToPackage{numbers,compress}{natbib}` — natbib defaults to author-year here,
  which renders `\cite` inline as "evict idle caches Kwon et al. [2023]", a sentence
  fragment, and costs several lines: a four-work citation group prints four full author
  strings instead of `[10-13]`. The audit asserts this.
- `\usepackage[draft]{hyperref}` is **required** in submission mode. The style file loads
  `lineno` whenever `final` is absent, and a hyperlink straddling a page break under
  `lineno` aborts the build with `This can't happen (pdfvlistout)`. `[draft]` keeps every
  hyperref command working but emits no link objects; URLs still render as text. Drop
  `[draft]` at the same time you add `final`, when `lineno` goes away.
- Figure 1 is authored at exactly `\textwidth` (5.5\,in) and included at
  `width=\textwidth`, so the scale factor is 1.0 and matplotlib's nominal font sizes are
  what the reader gets. An earlier version authored it 6.6\,in wide and displayed it at
  `0.86\textwidth`, which scaled every font by 0.72 and put the smallest text at 4.6\,pt.
  If you resize the figure, keep source width and display width equal, and re-check the
  smallest rendered size. Axis ticks use the V1/V2/V3 shorthand the body already uses;
  spelling the policies out in full made the labels collide.
- Tables use `\small`, never smaller. The template's "do not change font sizes" is about
  the style file's text rectangle and body font, and `\small` is what it sanctions
  elsewhere; shrinking further to buy space would be gaming the limit.
- The style file sets `\flushbottom`, so any page that cannot be filled has its slack
  stretched into the most elastic glue on it — which is the space around a `\section`
  heading. A short page 2 once showed 73\,pt above "2 Setup" and 45\,pt below, against
  18\,pt for every other heading. The fix is to give that page real content, not to
  reach for `\raggedbottom` (which made it worse here) or to relocate Figure 1 to page 2
  (which fixed the spacing but put the results plot ahead of the section that defines the
  policies it names). If a heading ever looks isolated again, check whether its page is
  short before touching the heading itself.
- Table 1 is `[b]`, not `[t]`. At this length `[t]` places it at the top of page 4 and
  pushes the conclusion onto page 5, over the body limit. If the paper's length changes,
  re-check the placement rather than assuming `[b]` is still right — the audit will catch
  it either way.

## Appendices

The CFP excludes appendices from the 4-page limit ("Authors are welcome to put additional
material in the optional Appendix section, but reviewers are not required to read the
Appendix"), and the NeurIPS template adds the binding constraint: "The paper must be able
to stand alone without the appendix; adding critical experiments that support the main
claims to an appendix is inappropriate."

Every main claim therefore carries its numbers in the body — the reversal, the mechanism,
both decomposition controls, the provisioning result, the oracle ablation, the V2+AC
result, and the cost model's validation range are all stated in the four pages. The
appendices hold the supporting tables and narrative:

- **A. Cost-model validation** — predicted vs measured prefill across a 12x context sweep
  on a T4 (within 1.15-1.66x), plus the cross-check against an independently coded
  analytic model. This is the answer to "your simulator was once catastrophically wrong;
  why should I believe it now?"
- **B. Admission control** — the V2+AC numbers, which never help, and why that is a result
  about density-thresholded admission rather than about the N*-thresholded rule.
- **C. The two evaluation defects** — both bugs in full, with the asymmetry argument.
- **D. Persona strength** — the sigma sweep separating the metric finding from the learned
  policy's standing.

## Files

- `paper.tex` — the paper
- `references.bib` — copied from `paper/latex/`; keep the two in sync if citations change
- `neurips_2026.sty` — the official NeurIPS 2026 style file, unmodified
- `figures/figure1_reversal.pdf` — regenerate with
  `python benchmarks/generate_workshop_figure.py`
