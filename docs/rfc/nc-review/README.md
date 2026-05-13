# NC Reviewer Assessment — Network Specification DSL for brainpy.state

**Manuscript:** [`../network-spec-dsl.md`](../network-spec-dsl.md) (2,146 lines after revision 1)
**Reviewer role:** Senior editor, computational neuroscience / neuromorphic computing
**Original review:** 2026-05-13
**Revision-1 re-review:** 2026-05-13

## Verdict (post-revision-1)

Still **major revision** before *Nature Communications*-grade research-contribution status. The novelty repositioning in the new §1.1 (training-paradigm pluralism over a single IR) is a substantive improvement and is the *right* axis to claim; however, the load-bearing claim is now *asserted but not demonstrated*, and concerns C2–C11 from the original report are untouched.

- **NC tier:** not yet suitable. Needs the seven minimum revision items in [`01-revision-1-review.md`](./01-revision-1-review.md), plus a single worked comparative-study benchmark (event-prop vs BPTT on the same IR).
- **Neuroinformatics / Frontiers in Neuroinformatics / JOSS / SoftwareX tier:** likely sufficient with C2 / C3 / C4 / C7 fixes from [`04-concerns.md`](./04-concerns.md).
- **Internal `brainpy.state` engineering adoption:** approximately ready; address C2 / C3 / C4 before implementation.

## Reading order

1. [`01-revision-1-review.md`](./01-revision-1-review.md) — **Read first.** Re-review of the §1.1 / D28 rewrite. Contains the current verdict and the minimum revision path to NC.
2. [`02-executive-summary.md`](./02-executive-summary.md) — Original executive summary.
3. [`03-strengths.md`](./03-strengths.md) — Major strengths (8 items).
4. [`04-concerns.md`](./04-concerns.md) — Major concerns C1–C11. C1 is partially addressed by revision 1; the rest are unchanged.
5. [`05-novelty-assessment.md`](./05-novelty-assessment.md) — Per-claim novelty table and analysis.
6. [`06-significance.md`](./06-significance.md) — Significance for the broader audiences.
7. [`07-semantic-gaps.md`](./07-semantic-gaps.md) — Ten conceptual / semantic gaps.
8. [`08-missing-features.md`](./08-missing-features.md) — Underdeveloped features.
9. [`09-prior-art-comparison.md`](./09-prior-art-comparison.md) — Capability matrix and audience overlap with PyNN, NeuroML, SONATA, Nengo, snnTorch, etc.
10. [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) — Essential / high-priority / optional revisions.
11. [`11-final-recommendation.md`](./11-final-recommendation.md) — Final editorial recommendation (post-revision-1).

## Extending this assessment

This directory is designed to evolve as `network-spec-dsl.md` is revised across multiple cycles. See [`MAINTENANCE.md`](./MAINTENANCE.md) for the conventions:

- New revision reviews are appended as `01-revision-2-review.md`, `01-revision-3-review.md`, … (older revision reviews are retained as the historical record)
- Concerns get inline `> **Status after revision N: ...**` blocks rather than being deleted (see `04-concerns.md` C1 for the model)
- Concern codes (`C1`–`C11`, `N1`–`N6`, …) and improvement-suggestion numbers are stable identifiers — never reuse
- `README.md` and `11-final-recommendation.md` always reflect the *current* verdict; the reasoning lives in the most recent `01-revision-N-review.md`

## Status of original concerns after revision 1

| Concern | Topic | Status |
|---|---|---|
| C1 | Novelty asserted, not demonstrated | **Partially addressed** by §1.1 / D28 rewrite. See [`01-revision-1-review.md`](./01-revision-1-review.md). |
| C2 | Spec / runtime semantic boundary | Unchanged |
| C3 | Determinism contract has unstated dependencies | Unchanged |
| C4 | Training-to-inference round-trip undefined | Unchanged |
| C5 | Hardware-mapping story is thin | Unchanged (partially de-scoped by D28) |
| C6 | Spatial / biophysical scope is narrow | Unchanged |
| C7 | Experiment / protocol / dataset abstractions missing | **Now load-bearing** for the new §1.1 claim |
| C8 | Plasticity expressiveness shallow | **Now load-bearing** (e-prop needs cross-projection eligibility hooks) |
| C9 | Composability restricted to chains | Unchanged |
| C10 | `Trainable` constraint vocabulary too coarse | Unchanged |
| C11 | `Builder` / `NetSpec` coexistence | Unchanged |

New concerns introduced by the revision-1 rewrite (N1–N6) are documented in [`01-revision-1-review.md`](./01-revision-1-review.md).
