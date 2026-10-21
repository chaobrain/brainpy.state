# Spec — 00 · Shared live-NEST validation harness

> Gitignored design doc (CLAUDE.md #7). Implements `2026-06-11-nest-goal-00-validation-harness.md`.
> Branch: `nest-goal/00-validation-harness` (worktree). Authoritative context: `CONTEXT.md`,
> `docs/nest-status/internal/numerical-validation-gap.md`, `.../index.md` P0 #1/#4.

## 1. Problem

`brainpy_state/_nest/_validation/` is today a set of ad-hoc per-example parity tests. Every test
re-implements the same glue (`try: import nest`, `unittest.skipUnless`, `nest.ResetKernel()`,
`abs(bp-nest)/nest < 0.05`) with **implicit, undocumented tolerances**. There is no shared answer to
"did it match NEST?", and the ~18 downstream `/goal` clusters (synapse rebuild + example ports) will
each invent their own. This cluster builds the shared harness they all import, and proves it by
refactoring the 5 existing Brunel/device parity tests onto it.

## 2. Constraints (from CONTEXT.md / gap docs)

- **Test-only infrastructure.** Do not touch model code (`CONTEXT.md` Part 2/4). The 5 tests keep
  their NEST run logic and brainpy `Simulator` drive **verbatim**; only skip + compare/assert glue moves
  to the harness ("thin glue" — confirmed).
- **brainpy side drives the `Simulator` API** (NEST-flavored), matching the existing Brunel tests.
- **NumPy-style docstrings** on every public helper, with a doctest `Examples` block in
  `.. code-block:: python` (CLAUDE.md docstring rules).
- `skipUnless(nest)` semantics: skip cleanly when `import nest` fails. **NEST is live & importable in
  this environment** (NEST 3.x), so the refactored tests must actually *run and pass*, not just skip.
- **Functional helpers only** (confirmed) — no per-family base classes (YAGNI until a cluster needs one).

## 3. Design decisions (resolved)

| # | Decision | Choice | Rejected alternatives |
|---|---|---|---|
| D1 | Tolerance categories A–E | **By model/compare kind** (table §5). Keeps the doc's A/B/C numbers; adds principled D/E. | "numbers-first tiers" (reorders the doc's letters); "minimal/literal" (leaves Siegert mean-field rate homeless). |
| D2 | Refactor scope | **Thin glue**: run logic verbatim; replace only `skipUnless`→`requires_nest` and the rel-error/assert block→`compare_*`. | DRY shared helpers (larger diff, tighter coupling, risks the "same assertions" requirement). |
| D3 | Public surface | **Functional helpers** + a `nest_compare(mode=…)` umbrella. | Per-family `NEST*ComparisonCase` base classes (doc §6 suggestion; none of the 5 tests need them). |
| D4 | Spec/plan location | `` (gitignored, **not committed**) — CLAUDE.md #7 overrides the brainstorming skill's "commit the design doc". | `` committed. |
| D5 | Core comparators take **values**, not callables | `compare_trace`/`compare_distributional` accept already-computed metrics → pure, NEST-free, trivially unit-testable (drives >90% coverage). The `nest_compare` umbrella wraps callables. | callables-only (couples coverage to a live NEST run). |
| D6 | `requires_nest` reads `HAS_NEST` at **decoration call time** | testable by patching the module global before decorating a dummy. Also tags `pytest.mark.requires_nest`. | capturing a module-level bool that can't be patched. |
| D7 | Tolerance pass test = numpy-allclose form `|a−b| ≤ atol + rtol·|ref|` | no division ⇒ zero-reference/zero-variance safe; reproduces both pure-abs (V_m, atol-only) and pure-rel (rate, rtol-only) existing assertions exactly. | explicit `|a−b|/ref` (divides by zero on silent generators / zero variance). |
| D8 | Omit a `nest_kernel` ResetKernel helper for now | thin-glue keeps `ResetKernel` inside each `_nest_*` fn; doc §5's "setUp helper" deferred to a cluster that needs it. README documents the convention. | ship it now (untested surface or NEST-coupling for no current consumer). |

## 4. Public API

### 4.1 `tolerance_conventions.py` — single source of truth

Three small frozen dataclasses (one purpose each) + named constants + sim defaults. One-screen module
docstring summarising the A–E table and the multi-seed protocol.

```python
@dataclass(frozen=True)
class TraceTolerance:          # categories A, B, C (deterministic per-sample / max-abs)
    atol: float | u.Quantity   # absolute tol; unit-aware for V_m (mV), plain for rates
    rtol: float                # relative tol (dimensionless)
    align_steps: int = 0       # allow ±N integer-sample shift search (recorder offset)
    label: str = ""; note: str = ""

@dataclass(frozen=True)
class DistributionalTolerance: # category D (multi-seed statistic)
    rate_rtol: float           # rel tol on the seed-aggregated mean (network rate parity, 5%)
    mean_diff_pct: float       # tighter single-cell mean tol (2%) — convention, future use
    autocorr_max_diff: float   # abs tol on autocorrelation (0.05) — convention, future use
    n_seeds: int               # minimum seeds (≥4–5)
    label: str = "D"; note: str = ""

@dataclass(frozen=True)
class SpikeTimeTolerance:      # category E (event-count / spike-time)
    max_count_diff: int        # |Δ event count| (2)
    max_peak_step_diff: int    # |Δ peak step| (1 ≈ dt)
    label: str = "E"; note: str = ""

# sim defaults (numerical-validation-gap.md §6)
T_DEFAULT = 1000.0 * u.ms;  DT_DEFAULT = 0.1 * u.ms;  N_SEEDS_DEFAULT = 5
```

Constants: `CAT_A`, `CAT_B`, `CAT_B_ALIGNED`, `CAT_C`, `CAT_C_RATE`, `CAT_D`, `CAT_E` (values in §5).

### 4.2 `nest_compare.py` — comparison engine

```python
HAS_NEST: bool                                   # True iff `import nest` succeeded at import

def requires_nest(obj):                          # decorator for TestCase class OR method
    """Skip when NEST absent (reads HAS_NEST at call time) + tag pytest.mark.requires_nest."""

@dataclass
class ComparisonResult:
    passed: bool; error: float; bound: float; metric: str; detail: str
    def assert_(self) -> None: ...               # raise AssertionError(detail) if not passed

def compare_trace(reference, candidate, *, tol: TraceTolerance,
                  metric: str = "trace") -> ComparisonResult:
    """Deterministic per-sample / max-abs comparison (A/B/C). Scalars are 0-d traces.
    Optional ±tol.align_steps integer-shift search picks the best overlap (recorder offset)."""

def compare_distributional(reference_samples, candidate_samples, *,
                           tol: DistributionalTolerance, metric: str = "rate",
                           statistic="mean") -> ComparisonResult:
    """Multi-seed statistical comparison: aggregate each side (mean) then compare;
    NEVER per-sample. Zero-variance / zero-mean safe (guarded denominator)."""

def nest_compare(nest_fn, brainpy_fn, *, mode: str, tol, metric: str = "metric",
                 seeds=None, statistic="mean") -> ComparisonResult:
    """Run nest_fn (live NEST) + brainpy_fn (brainpy.state) and compare.
    mode='trace'        -> compare_trace(nest_fn(), brainpy_fn(), …)
    mode='distributional' (needs seeds) -> compare_distributional([nest_fn(s)…],[brainpy_fn(s)…], …)"""
```

`reference`/`candidate` accept plain floats/arrays **or** saiunit Quantities. When `tol.atol` is a
Quantity, both are converted to its unit; when plain, mantissas are taken. Canonical roles:
`reference` = NEST (or analytic ground-truth, e.g. the configured Poisson rate); `candidate` = brainpy.

### 4.3 `conftest.py`

Registers the marker (`config.addinivalue_line("markers", "requires_nest: needs a live NEST install")`)
so `pytest -m requires_nest` selects and there is no unknown-mark warning.

## 5. Tolerance categories A–E

Categories span the comparison space the goal names (V_m trace, firing rate, weight trajectory,
PSC-amplitude train, F-I curve). A/B/C = deterministic traces by integration kind (faithful to
`numerical-validation-gap.md` §6); D = distributional; E = spike-time.

| Cat | Kind | Models / metrics | atol | rtol | Mode | Maps which of the 5 |
|---|---|---|---|---|---|---|
| **A** | adaptive numerical-integrator trace | aeif/HH/izhikevich V_m; weight & PSC & F-I via RKF45 | `1e-3 mV` | `1e-3` | trace | — (cluster 02/04+) |
| **B** | analytic exact-propagator trace | linear `iaf_psc_*` V_m / PSC (near-exact) | `1e-6 mV` | `1e-6` | trace | — (cluster 02) |
| **B*** | B with recorder alignment | same, allow 1-step multimeter offset | `5e-2 mV` | `1e-3` (`align_steps=1`) | trace | — (`iaf_psc_alpha_parity`) |
| **C** | conductance / coupled trace | `iaf_cond_*`, multicompartment | `1e-3 mV` | `1e-3` | trace | — |
| **C(rate)** | mean-field rate fixed point | `siegert_neuron` asymptotic rate (Hz, deterministic) | `0.0` | `5e-2` | trace | **brunel_siegert** |
| **D** | distributional / PRNG-divergent | network firing rate, ISI CV, generator rate; multi-seed | rate `5%`, mean `2%`, autocorr `0.05`, n_seeds ≥4 | — | distributional | **brunel_alpha/delta/exp_multisynapse**, **device** |
| **E** | spike-time / event-count | precise `*_ps`, PSC peak timing, event counts | `|Δcount|≤2`, `|Δpeak|≤1` | — | trace-ish | — (`iaf_psc_alpha_parity` future) |

Multi-seed protocol (D): run N≥`N_SEEDS_DEFAULT` (=5) seeds per side, compare the **seed-mean**
(never per-sample); single-realization legacy tests use n_seeds=1 (documented limitation).

## 6. Comparison modes

- **`trace`** — deterministic drive ⇒ tight per-sample / max-abs-error (categories A/B/C, incl. scalar
  fixed points like Siegert). Optional integer-step alignment for recorder offset.
- **`distributional`** — PRNG-divergent drive ⇒ compare a statistic (mean) over seeds, never per-sample
  (category D).

## 7. Mapping the 5 tests onto the harness (thin glue)

| Test | Run logic (kept verbatim) | New compare call | Skip |
|---|---|---|---|
| `brunel_alpha_test` | `build()`+`sim.simulate()`; `_nest_rates()` | `compare_distributional([nest], [bp], tol=CAT_D, metric="exc rate").assert_()` (n=1) | `@requires_nest` |
| `brunel_delta_test` | same | same | `@requires_nest` |
| `brunel_exp_multisynapse_test` | seed loop (4 seeds) | `compare_distributional(nest_list, bp_list, tol=CAT_D, metric="exc rate").assert_()` | `@requires_nest` |
| `brunel_siegert_test` | `run()`; `_nest_rates()`→(rex,rin) | per (exc,inh): `compare_trace(ns, bp, tol=CAT_C_RATE, metric=f"{l} mean-field rate").assert_()` | `@requires_nest` |
| `device_parity_test` | poisson gen (bp eager); nest parrot+recorder; recorder-stamp unit test | bp & nest rate: `compare_distributional([1000.0], [rate], tol=CAT_D, metric="poisson rate").assert_()`; recorder-stamp **unchanged** | `@requires_nest` only on the NEST sub-test |

All preserve the existing `< 5%` / abs assertions exactly (D7 allclose form is equivalent). The
`device` recorder-stamp test is a pure brainpy unit test (no parity) and is left untouched.

## 8. Edge cases → harness unit tests (NEST-free, drive coverage)

`tolerance_conventions_test.py` + `nest_compare_test.py`:

1. **Multi-seed reproducibility** — `compare_distributional` / `nest_compare(distributional)` with
   fixed seeds yields a deterministic aggregate across repeated runs.
2. **Failing `trace` flags a clear diff** — mismatched arrays ⇒ `passed=False`, `detail` names the
   max-abs error + argmax index; `.assert_()` raises `AssertionError` with that detail.
3. **`requires_nest` actually skips** — patch `nest_compare.HAS_NEST=False`, decorate a dummy ⇒
   `__unittest_skip__ is True` with the reason; patched True ⇒ not skipped.
4. **Tolerance constants unit-aware** — `CAT_A/B/C.atol` are saiunit Quantities in mV;
   `T_DEFAULT`/`DT_DEFAULT` carry ms.
5. **Zero-variance distributional ≠ divide-by-zero** — identical per-seed samples and all-zero
   reference ⇒ finite result, no `ZeroDivisionError`/`NaN`.
6. Plus: `compare_trace` alignment picks the best shift; allclose-form reproduces pure-abs & pure-rel;
   `nest_compare` mode dispatch (trace & distributional) with dummy fns; unit-aware vs plain inputs.

## 9. Files

```
brainpy_state/_nest/_validation/
  tolerance_conventions.py        (new)  dataclasses + A–E constants + defaults
  nest_compare.py                 (new)  HAS_NEST, requires_nest, ComparisonResult, compare_*, nest_compare
  conftest.py                     (new)  register requires_nest marker
  README.md                       (new)  how-to + carve-outs + run with/without nest
  tolerance_conventions_test.py   (new)  unit tests (unit-aware, constants)
  nest_compare_test.py            (new)  unit tests (edge cases 1-6; NEST-free)
  brunel_alpha_test.py            (edit) thin-glue refactor
  brunel_delta_test.py            (edit) thin-glue refactor
  brunel_exp_multisynapse_test.py (edit) thin-glue refactor
  brunel_siegert_test.py          (edit) thin-glue refactor
  device_parity_test.py           (edit) thin-glue refactor (NEST sub-test + rate sub-tests)
```

`brunel_alpha_evolution_strategies_test.py` and `iaf_psc_alpha_parity_test.py` are **out of scope**
(not in the 5); left untouched, but the API is designed to fit them later (ES→D, iaf V_m→B*, spike→E).

## 10. Out of scope / YAGNI

Per-family base classes; `nest_kernel`/setUp helper; autocorr/CV distributional path (constants
recorded, comparator computes mean only); CI matrix; coverage badge; tightening existing 5% tolerances
(goal requires "same assertions"). These are later clusters' work.

## 11. Definition of done (from the goal)

- [ ] `nest_compare.py` + `tolerance_conventions.py` + `../../README.md` exist; `requires_nest` registered.
- [ ] 5 tests refactored onto the harness; pass with NEST, skip cleanly without.
- [ ] >90% coverage on the harness; NumPy docstrings (doctest Examples) complete.
- [ ] Lessons entry appended to `CONTEXT.md`; git lifecycle done.
