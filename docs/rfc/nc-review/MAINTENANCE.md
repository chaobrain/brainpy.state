# Maintenance Guide — extending and updating this assessment

This directory is designed to evolve as the spec at [`../network-spec-dsl/`](../network-spec-dsl/) is revised. **Assessment files always reflect the current state of the spec.** No revision history, no resolved items, no annotations of "addressed in revision N." When the spec changes, each file is updated in place: closed concerns are deleted, addressed missing features are deleted, novelty claims that no longer hold are deleted.

## Per-file editing conventions

| File                              | Editing rule                                                                                                                                                                                            |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `README.md`                       | Current verdict, reading order, and editing rules. No revision-dates table; no per-concern history. Update whenever the verdict or reading order changes.                                              |
| `02-executive-summary.md`         | Current high-level framing of the contribution. Update when the central claim or central remaining gap changes.                                                                                        |
| `03-strengths.md`                 | Currently-valid strengths. Delete strengths that no longer hold; add new strengths as they emerge.                                                                                                     |
| `04-concerns.md`                  | Currently-open concerns. Delete concerns the spec has addressed. Use stable codes (`C1`, `C2`, …) but **do not retain resolved codes** — when an item is closed, its code retires with the file edit. |
| `05-novelty-assessment.md`        | Current per-claim novelty. Delete claims that no longer reflect the spec. Add new claims as new design surface lands.                                                                                  |
| `06-significance.md`              | Current audience-significance calculus. Update when audiences shift.                                                                                                                                   |
| `07-semantic-gaps.md`             | Currently-open conceptual / semantic gaps. Delete gaps the spec has closed; add new gaps as they emerge.                                                                                               |
| `08-missing-features.md`          | Currently-open missing features. Delete features the spec has added; add new gaps as they emerge.                                                                                                      |
| `09-prior-art-comparison.md`      | Current capability matrix and audience overlap. Add columns / rows for new comparators; update cells when the spec's relative position changes.                                                        |
| `10-improvement-suggestions.md`   | Currently-open improvement suggestions. Delete suggestions that have been incorporated. Use stable numbers but **do not retain incorporated numbers** — incorporated items are removed entirely.       |
| `11-final-recommendation.md`      | Current verdict and minimum path to NC. Update whenever the verdict or minimum-path list changes.                                                                                                       |
| `MAINTENANCE.md`                  | These conventions. Update when the conventions themselves change.                                                                                                                                       |

## Rationale for the no-history policy

A revision-history record is useful for understanding *how the spec evolved*; it is not useful for understanding *the current state*. Mixing both into one set of files makes both harder to read. The git history of this directory preserves the revision record; the file contents preserve the current state. Reviewers reading the files do not need to mentally subtract "addressed in revision N" annotations to see what is open today.

When a substantial revision lands, the typical edit pattern is:

1. Read the diff of the spec.
2. Open `04-concerns.md`, `07-semantic-gaps.md`, `08-missing-features.md`, `10-improvement-suggestions.md`. Delete every item the revision addresses.
3. Add new items the revision introduces (new design surface that needs critique, or new gaps that the revision exposed).
4. Update `03-strengths.md` — delete superseded strengths, add new ones.
5. Update `05-novelty-assessment.md` — delete superseded novelty rows, add new ones.
6. Update `09-prior-art-comparison.md` cells if the spec's relative position shifted.
7. Update `README.md` verdict block and `11-final-recommendation.md` if the central verdict or minimum-path list changed.
8. Update `02-executive-summary.md` and `06-significance.md` only if the high-level framing changed.

The result is a small, current snapshot — never a layered archaeological record.

## Cross-references

File names use a numeric prefix (`02-`, `03-`, …) to enforce reading order. The prefix is part of the cross-reference; if you renumber a file, update every reference to it. Section anchors inside files (`#c1-...`) use stable code prefixes where possible so cross-file links survive minor reordering.

## Verdict-tier vocabulary

The assessment uses a stable vocabulary for venue suitability — keep these consistent across edits:

- **NC tier** — *Nature Communications*, *Nature Methods*, *Nature Machine Intelligence* (top-tier methods venues with high evidence bars: comparative benchmarks, multi-axis demonstrations).
- **Comp-neuro / SoftwareX tier** — *Neuroinformatics*, *Frontiers in Neuroinformatics*, *PLOS Computational Biology* methods track, *JOSS*, *SoftwareX* (mid-tier: well-designed system + reference implementation, comparative study not strictly required).
- **Internal-engineering tier** — adoption-ready for `brainpy.state` users; not yet a research contribution.

`11-final-recommendation.md` always states suitability for all three tiers.
