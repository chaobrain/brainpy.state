# /goal — 00 · Shared live-NEST validation harness

> Self-contained prompt. Open at repo root. Stream: both (foundation). Depends on: nothing.

## Session start

1. Create an isolated worktree branch **`nest-goal/00-validation-harness`** from `origin/main` (use `superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — it is authoritative on architecture and conventions. Then read:
   - `docs/nest-status/internal/numerical-validation-gap.md` (the harness this implements; tolerance categories A–E, multi-seed protocol).
   - `docs/nest-status/internal/index.md` P0 #1 and #4 (acceptance criteria).
   - The 5 existing parity tests you will refactor: `brainpy_state/_nest/_validation/{brunel_alpha,brunel_delta,brunel_exp_multisynapse,brunel_siegert,device_parity}_test.py`.
3. Write spec + plan under `` (gitignored). **State your approach and wait for approval before writing code** (`CLAUDE.md` #1).

## Goal

Build the shared harness every later cluster's parity test will import, so "did it match NEST?" is asked the same way everywhere. Today `_validation/` is just ad-hoc per-example tests with implicit, undocumented tolerances.

## Scope

Create under `brainpy_state/_nest/_validation/`:

- **`nest_compare.py`** — the comparison engine. At minimum:
  - a `requires_nest` skip decorator/marker (skip cleanly when `import nest` fails);
  - helpers to run a callable in **live NEST** and in **brainpy.state** and compare a named metric (V_m trace, firing rate, weight trajectory, PSC-amplitude train, F-I curve) with an explicit tolerance and an explicit comparison **mode**;
  - the **two comparison modes** from `CONTEXT.md`/§3.2: `trace` (deterministic drive → tight per-sample / max-abs-error) and `distributional` (PRNG-divergent drive → compare statistics over multiple seeds, never per-sample).
- **`tolerance_conventions.py`** — the A–E tolerance categories + the multi-seed protocol from `numerical-validation-gap.md`, as importable constants + a one-screen module docstring. This is the single source of truth for "what tolerance for what kind of model."
- **`../../README.md`** — how to write a parity test with this harness; the carve-out rules (distributional / docs-only / blocked→skipped-placeholder); how to run with and without `nest`.

Then **refactor the 5 existing parity tests** onto the harness (same assertions, now using the shared compare + documented tolerances) to prove the API and lock the convention.

## Constraints

- Per `CONTEXT.md` Part 2/4. The harness is test-only infrastructure; do not touch model code.
- Keep the brainpy.state side driving the **`Simulator` API** (NEST-flavored), matching the existing Brunel tests.
- NumPy-style docstrings on every public helper (doctest `Examples` in `.. code-block:: python`).

## Edge cases to cover with tests

Multi-seed averaging reproducibility; a `trace` comparison that *should* fail flags a clear diff; `requires_nest` actually skips (simulate `nest` absent); tolerance constants are unit-aware; zero-variance distributional input doesn't divide-by-zero.

## Definition of done

- [ ] `nest_compare.py` + `tolerance_conventions.py` + `../../README.md` exist; `requires_nest` registered.
- [ ] 5 Brunel/device parity tests refactored onto the harness and passing (with `nest`) / skipping (without).
- [ ] >90% coverage on the harness; docstrings complete.
- [ ] Lessons entry appended to `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`** (Part 5 template): the `nest_compare` API shape the next clusters will import, the tolerance categories chosen, any NEST-setup gotchas.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → open PR (body: scope + that the 5 tests still pass) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove the worktree.
