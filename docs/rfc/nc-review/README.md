# NC Reviewer Assessment — Network Specification DSL for brainpy.state

**Manuscript:** [`../network-spec-dsl/`](../network-spec-dsl/) — 11-chapter directory.
**Reviewer role:** Senior editor, computational neuroscience / neuromorphic computing.

## Verdict

**Major revision** required before *Nature Communications*-grade research-contribution status. The spec-design surface is mature; the gap is *evidence* for the §1.1 training-paradigm-pluralism claim, not *design*.

- **NC tier:** not yet suitable. The load-bearing missing pieces are listed in [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) under *Essential* — most consequential: a `G13` training-paradigm-pluralism goal traced through the IR, a per-shipped-training-backend capability matrix instantiated in §5, and one worked comparative-study benchmark reproducing a published result (e.g. event-prop vs BPTT on YinYang or N-MNIST) by switching the backend on a single IR.
- **Neuroinformatics / Frontiers in Neuroinformatics / JOSS / SoftwareX tier:** likely sufficient as it stands. The expressive surface plus reference design clears the bar at this tier.
- **Internal `brainpy.state` engineering adoption:** **ready.** Begin implementation. The IR is well-specified, the determinism contract is end-to-end (§9.1), the validation catalog is comprehensive (§9.2), and the `Builder` / `NetSpec` coexistence story is sharp (`Builder` is the substrate of `brainpy.state.clock`; `NetSpec` is the user-facing entry point).

## Reading order

1. [`02-executive-summary.md`](./02-executive-summary.md) — Executive summary.
2. [`03-strengths.md`](./03-strengths.md) — Major strengths.
3. [`04-concerns.md`](./04-concerns.md) — Open concerns.
4. [`05-novelty-assessment.md`](./05-novelty-assessment.md) — Per-claim novelty.
5. [`06-significance.md`](./06-significance.md) — Significance for the broader audiences.
6. [`07-semantic-gaps.md`](./07-semantic-gaps.md) — Open conceptual / semantic gaps.
7. [`08-missing-features.md`](./08-missing-features.md) — Open underdeveloped features.
8. [`09-prior-art-comparison.md`](./09-prior-art-comparison.md) — Capability matrix and audience overlap.
9. [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) — Essential / high-priority / optional revisions.
10. [`11-final-recommendation.md`](./11-final-recommendation.md) — Final editorial recommendation.

## Maintenance

Each assessment file reflects the **current** state of the spec — no revision history, no resolved items. When the spec changes, each file is updated in place: closed concerns are deleted; addressed missing features are deleted; novelty claims that no longer hold are deleted.

See [`MAINTENANCE.md`](./MAINTENANCE.md) for per-file editing conventions.
