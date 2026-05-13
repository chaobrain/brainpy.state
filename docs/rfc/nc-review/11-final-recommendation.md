# Final Editorial Recommendation (post-revision-1)

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation. The detailed reasoning behind this recommendation lives in [`01-revision-1-review.md`](./01-revision-1-review.md).

**Decision: Major revision required for *Nature Communications*. The revision-1 rewrite is a substantive improvement on novelty positioning but is not yet sufficient on novelty evidence.**

- **As an engineering specification for internal `brainpy.state` adoption:** the document is approximately ready. Address C2 (semantic surface), C3 (RNG contract), and C4 (training round-trip) before implementation; the rest can iterate with the codebase.
- **As a research contribution for *Neuroinformatics* / *Frontiers in Neuroinformatics* / *JOSS* / *SoftwareX*:** the §1.1 rewrite plus C2 / C3 / C4 / C7 fixes likely clears the bar. The training-paradigm-pluralism positioning is publishable at this tier on positioning + design alone.
- **As a research contribution for *Nature Communications*:** **not yet suitable.** The §1.1 rewrite correctly identifies the empty quadrant on the prior-art map (every existing SNN framework commits to one training paradigm) and pivots the claim to that wedge — this is the right move and is necessary. But the body of the spec must follow through: the IR (§5) must surface what each paradigm requires, the per-training-backend capability matrix (§8.2) must be written out, validation (§14) must catch IR × backend mismatches, testing (§16) must exercise the four-paradigm claim, and **at least one published comparative-study result must be reproduced by switching backend on a single IR** (e.g., event-prop vs BPTT on YinYang or N-MNIST, matching Wunderlich & Pehle 2021 within reported error bars).

## Minimum revision to reach NC standard

A *Nature Communications*-grade revision needs all seven items in [`01-revision-1-review.md`](./01-revision-1-review.md) "Specific, minimum revision to reach NC standard":

1. New G13 for training-paradigm pluralism, traced through goals / IR / validation.
2. Per-training-backend capability matrix in §8.2.
3. Worked §10.x training round-trip across backends.
4. Reproduced published benchmark via single-line backend swap.
5. Expanded §1.1 prior art including EXODUS, hxtorch.snn, Lava-DL, NeuroML/LEMS, SONATA, NMODL/NESTML, NIR.
6. New SPEC-NNN codes for IR × training-backend mismatches.
7. Protocol / Dataset / minimal Optimizer abstractions to make comparative-study reproducibility achievable (the load-bearing form of original C7).

The remaining concerns from the original report (C2, C3, C4, C5, C6, C8, C9, C10, C11 in [`04-concerns.md`](./04-concerns.md)) are valid as secondary revision items but are not the NC blocker. The training-paradigm-pluralism claim's evidence is the blocker.

## Closing note

The technical work shown here remains competent and frequently elegant, and the §1.1 rewrite is sharp and well-judged in direction. The remaining gap is no longer editorial positioning — that has been substantially fixed — it is *evidence*: the IR and the backends must visibly mechanize the claim that §1.1 makes, and a single end-to-end comparative study must demonstrate it. With those in place, this becomes a credible *Nature Communications* methods/systems submission and arguably one of the more important SNN-tooling contributions of the year.
