# Final Editorial Recommendation

> Part of the editorial report on [`../network-spec-dsl/`](../network-spec-dsl/). See [README](./README.md) for navigation.

**Decision: Major revision required for *Nature Communications*.**

The spec-design surface is mature. The expressive surface a comparative-study workflow would need — signals, schedules, cross-projection eligibility traces, DAG composability, noise, build-time variables, plus spatial / morphological / plasticity-six-layers / DAG-temporal-offset — is in place. The architectural discipline (immutable IR, declared build-time variables, backends as peer top-level modules under `brainpy.state`) is sharp and well-reasoned.

What remains is **evidence**, not design: the IR must visibly mechanize the §1.1 training-paradigm-pluralism claim, the per-shipped-training-backend capability matrix must be written out, and a single end-to-end comparative-study notebook must demonstrate the headline claim.

## Suitability by venue

- ***Nature Communications* / *Nature Methods* / *Nature Machine Intelligence*:** **not yet suitable.** The §1.1 claim is expressively supported but not mechanically demonstrated. See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) *Essential* items 1–7.
- ***Neuroinformatics* / *Frontiers in Neuroinformatics* / *JOSS* / *SoftwareX*:** **likely sufficient as it stands.** The design plus reference implementation plus protocol (schedules) plus canonical worked example (§3.15 cortex-striatum loop) clears the bar at this tier. The remaining gaps are NC-tier evidence questions.
- **Internal `brainpy.state` engineering adoption:** **ready.** Begin implementation. The IR is well-specified, the determinism contract is end-to-end, the validation catalog is comprehensive, and the `Builder` / `NetSpec` coexistence story is sharp.

## Minimum path to NC standard

In order of priority:

1. **`G13` line + IR markers for training-paradigm requirements.** Trace through Chapter 2 (IR), Chapter 6 (`BackendCapabilities` per-backend instantiation), and Chapter 10 (SPEC-NNN codes for IR × training-backend mismatch). See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #1.
2. **Per-shipped-training-backend capability matrix.** Concrete rows for `clock` / `event` / `bptt` / `eprop` / `eventprop` / `ppprop` in §6.1. See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #2.
3. **Reproduce one published comparative-study benchmark** by single-line backend swap (event-prop vs BPTT on YinYang or N-MNIST, matching Wunderlich & Pehle 2021 within reported error bars). Ship as a notebook + CI smoke test. See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #3.
4. **Engage the missing prior art** — Lava-DL, hxtorch.snn, EXODUS on the multi-paradigm axis; NeuroML / LEMS, SONATA, NMODL / NESTML, NIR-as-upstream-IR on the declarative-spec axis. See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #4.
5. **Training-round-trip §5.x walkthrough.** `bptt.build → train → archive → eventprop.build → train`, including how trained weights are bound. See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #5.
6. **Hardware story positioning.** Either an end-to-end exemplar (Loihi / SpiNNaker) or honest-scoped G11 ("graph-level NIR export; deployment is consumer-toolchain responsibility"). See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #6.
7. **Versioning / schema-evolution policy.** See [`10-improvement-suggestions.md`](./10-improvement-suggestions.md) #7.

## Closing note

The technical work is competent and frequently elegant. The §1.1 framing is sharp and well-judged. The gap to NC is no longer editorial positioning or expressive design — both are in place — it is evidence: the IR and the backends must visibly mechanize the §1.1 claim, and a single end-to-end comparative study must demonstrate it. With those, this becomes a credible *Nature Communications* methods/systems submission and arguably one of the more important SNN-tooling contributions of the year.
