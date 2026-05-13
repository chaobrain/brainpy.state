# Maintenance Guide — extending and updating this assessment

This directory is designed to evolve as `network-spec-dsl.md` is revised. This guide describes the conventions for keeping the assessment coherent across multiple revision cycles.

## File-level conventions

| File | When to edit | Notes |
|---|---|---|
| `README.md` | Every revision; whenever the verdict, status table, or reading order changes | Single source of truth for the *current* verdict and per-concern status |
| `01-revision-N-review.md` | Append a new file per spec revision (`01-revision-2-review.md`, `01-revision-3-review.md`, …) | One file per spec revision. Older revision reviews are retained — they are the historical record |
| `02-executive-summary.md` | Rarely. Only when the high-level framing of the contribution shifts substantively | If a revision changes the framing materially, update the executive summary AND keep the original phrasing in the corresponding `01-revision-N-review.md` |
| `03-strengths.md` | When new strengths emerge in a revision | Append new items at the end with `(added in revision N)`; keep historical items intact unless they become factually wrong |
| `04-concerns.md` | When a concern is addressed, partially addressed, expanded, or new | Use the existing inline-status pattern (see C1 for the model). Do not delete historical concerns; mark them resolved with `**Status after revision N: RESOLVED.**` and a short note |
| `05-novelty-assessment.md` | When the per-claim novelty changes | The table is keyed by *claim*; add rows for new claims; update the `Net:` line if the central thesis shifts |
| `06-significance.md` | When the audience-significance calculus shifts | Usually only updated alongside a major framing change |
| `07-semantic-gaps.md` | When a gap is closed in a revision OR a new gap is uncovered | Mark closed gaps with `~~strikethrough~~` and a note pointing to the spec section that closed it. Do not renumber |
| `08-missing-features.md` | When the spec adds or removes a feature | Update the table cells; add a `Status` column if the table starts tracking resolution |
| `09-prior-art-comparison.md` | When new comparators emerge (new framework released; new comparative-study paper) | The capability matrix is the source of truth — update cells, add columns; update `Bottom line on differentiation` if it shifts |
| `10-improvement-suggestions.md` | When suggestions are incorporated by the manuscript | Mark incorporated items with ✓ and a pointer to the spec section that addressed them; keep the numbering stable so cross-references remain valid |
| `11-final-recommendation.md` | Every revision | Should always reflect the *current* recommendation. The reasoning lives in the most recent `01-revision-N-review.md`; this file just states the verdict |
| `MAINTENANCE.md` | When the conventions themselves change | Rare |

## Adding a new revision review

When `network-spec-dsl.md` ships a new revision:

1. **Create `01-revision-N-review.md`** (e.g. `01-revision-2-review.md`) following the template below. Do not modify prior revision reviews — they are the historical record of how the manuscript evolved.
2. **Update `README.md`:**
   - Bump the `Manuscript:` line count
   - Update `Revision-N re-review:` date
   - Update the `Verdict (post-revision-N)` block
   - Update the *Status of original concerns* table (the C1–C11 / N1–N6 status columns)
   - Add the new revision file to the reading order
3. **Update `04-concerns.md`** — for each concern the revision addressed, partially addressed, or made more acute, add a `> **Status after revision N: ...**` block at the top of the concern section (see C1 for the model)
4. **Update `11-final-recommendation.md`** to the new verdict; the file should always state the *current* recommendation
5. **Selectively update** `02`–`10` per the table above

### Template for `01-revision-N-review.md`

```markdown
# Revision N — re-review of the [section / decision] rewrite

> See [README](./README.md) for navigation. This is the revision-N re-review.
> Prior re-reviews: [`01-revision-(N-1)-review.md`](./01-revision-(N-1)-review.md).

## What was changed

[diff summary: which sections / decisions were modified, line counts, the
 essence of the change]

## What this fixes

[which prior concerns (C-codes, N-codes from earlier revisions) this addresses,
 partially addresses, or moves]

## What still blocks NC acceptance

[new N-codes if applicable, numbered continuing from prior revision: N7, N8, ...]

## Specific, minimum revision to reach NC standard

[updated minimum-revision list — supersedes prior revision reviews' lists]

## Updated final verdict for this revision

[concise verdict; mirrored in `11-final-recommendation.md`]
```

## When the manuscript changes substantively

If a revision is large enough that the entire assessment needs rewriting (not just incremental additions), prefer:

1. Branch the directory: `nc-review-archive-rN/` for the old state, fresh `nc-review/` for the new
2. The README of the new directory should explicitly point at the archive

Avoid this if at all possible — incremental updates with clear `Status after revision N` annotations are easier to follow than parallel directories.

## Cross-reference discipline

- Concern codes (`C1`–`C11`) and revision-N concern codes (`N1`–`N6`, then `N7`–`Nk` etc.) are stable identifiers. **Never reuse a number.** When a concern is resolved, mark it resolved — don't recycle the code.
- File names use a numeric prefix (`01-`, `02-`, …) to enforce reading order. The prefix is part of the cross-reference; if you renumber a file, update every reference to it.
- Improvement-suggestion numbers (1–21 in `10-improvement-suggestions.md`) are also stable identifiers. Add new suggestions at the end with the next available number.

## Verdict-tier vocabulary

The assessment uses a stable vocabulary for venue suitability — keep these consistent across revisions:

- **NC tier** — *Nature Communications*, *Nature Methods*, *Nature Machine Intelligence* (top-tier methods venues with high evidence bars: comparative benchmarks, multi-axis demonstrations)
- **Comp-neuro / SoftwareX tier** — *Neuroinformatics*, *Frontiers in Neuroinformatics*, *PLOS Computational Biology* methods track, *JOSS*, *SoftwareX* (mid-tier: well-designed system + reference implementation, comparative study not strictly required)
- **Internal-engineering tier** — adoption-ready for `brainpy.state` users; not yet a research contribution

Each `11-final-recommendation.md` revision should state suitability for all three tiers.
