# Numerical validation — NEST parity status (cross-cutting)

_As of 2026-06-16._

## 1. Scope

Cross-cutting status of NEST-comparison test coverage across all ported modules
in `brainpy_state/_nest/`. This page used to be a proposal to *build* a
validation harness; that harness has since shipped and the cluster backlog
(00–28) that filled it out has fully merged. The page now records the shipped
reality: what the harness is, what it covers, the tolerance conventions it
enforces, and the genuine residual gaps that remain.

Upstream reference: not applicable — this is a repo-internal classification.

Evidence basis (verified 2026-06-16):
- `brainpy_state/_nest/_validation/` exists and holds **140 files**, of which
  **120** carry `@requires_nest` (live-NEST parity) and/or analytic checks.
- The harness ships its own unit tests (`nest_compare_test.py`,
  `tolerance_conventions_test.py`) that run with or without NEST installed.
- **122** non-test modules in `brainpy_state/_nest/`; **75** ports under
  `examples/nest_like/`, each backed by a `_validation` parity test.
- Cluster backlog 00–28 all merged (cluster 18, e-prop, moved to the sibling
  `braintrace` package and is out of scope here).

## 2. Parity summary

NEST-comparison coverage is now broad rather than bimodal. The shared harness
(`nest_compare.py` + `tolerance_conventions.py`) is the single comparison path,
and nearly every ported model has a `_validation` parity test that imports it.
AdEx, the IAF psc / cond / specialized families, GIF, GLIF, HH (incl. the
gap-junction variants), MAT/AMAT, Izhikevich, binary stochastic neurons, rate
models, the full device set (generators / recorders / detectors), and the
synapse + plasticity families (static, STDP, STP/Tsodyks, quantal STP, Clopath,
Urbanczik, Jonke, Vogels-Sprekeler) all have parity tests under
`_validation/`.

Tolerance, comparison mode, and the multi-seed protocol are no longer
undocumented per-test conventions: they live in `tolerance_conventions.py`
(categories A–E) and are imported by every parity test, so the question "did it
match NEST, and within what tolerance?" is asked the same way everywhere.

| Bucket | Status | Notes |
|---|---|---|
| Shared comparison harness | **shipped** | `nest_compare.py`, `tolerance_conventions.py`, `conftest.py`, `README.md` + self-tests |
| Tolerance conventions (A–E) | **documented + importable** | `tolerance_conventions.py` is the single source of truth |
| `@requires_nest` marker + skip-guard | **shipped** | registered in `conftest.py`; defined in `nest_compare.py` |
| Per-model parity tests | **~all ported models** | 120 of 140 `_validation` files carry `@requires_nest` |
| Genuine residual | **small** | see §6 — chiefly `pp_psc_delta`; do **not** claim 100 % parity |

## 3. The shipped harness

`brainpy_state/_nest/_validation/` is the live-NEST parity harness. It is the
shared answer to "did it match NEST?": every parity test imports the same
comparison engine and the same documented tolerances.

- **`nest_compare.py`** — the comparison engine:
  - `requires_nest` — decorator/skip-guard that skips a `TestCase` class or
    method when `import nest` fails and tags the `requires_nest` pytest marker
    (so `pytest -m requires_nest` selects the live-NEST tests).
  - `compare_trace(reference, candidate, *, tol, metric)` — deterministic
    per-sample / max-abs comparison (categories A/B/C). Pass test is the
    division-free numpy-allclose form `max|a − b| ≤ atol + rtol·max|ref|`, with
    an optional `±align_steps` integer-shift search to absorb a recorder
    one-step offset.
  - `compare_distributional(reference_samples, candidate_samples, *, tol,
    metric, statistic)` — multi-seed statistical comparison (category D). Never
    compares per-sample; aggregates each side (`"mean"`, `"cv"`, or `"autocorr"`)
    and compares the aggregate.
  - `nest_compare(nest_fn, brainpy_fn, *, mode, tol, seeds, statistic)` — the
    umbrella convenience that *runs* the two callables and dispatches to the
    right comparator.
  - `ComparisonResult` with `.assert_()` — the comparators return a result
    object; `.assert_()` raises `AssertionError(detail)` on failure.
- **`tolerance_conventions.py`** — the single source of truth for *what
  tolerance for what kind of model* (categories A–E + the multi-seed protocol +
  sim defaults). Importable, unit-aware constants.
- **`conftest.py`** — registers the `requires_nest` pytest marker (no
  `PytestUnknownMarkWarning`).
- **`README.md`** — how to write a parity test.
- **Self-tests** — `nest_compare_test.py` and `tolerance_conventions_test.py`
  exercise the engine and the constants. The core comparators take
  already-computed metric values (plain floats/arrays or `brainunit`
  quantities), so they are pure and unit-testable *without* NEST installed.

### Two comparison modes

| Mode | When | How it compares | Category | Helper |
|---|---|---|---|---|
| `trace` | deterministic drive (same dt, fixed input, analytic / mean-field) | per-sample / max-abs error, optional ±1-step recorder alignment | A, B, C | `compare_trace` |
| `distributional` | PRNG-divergent drive (Poisson, random connectivity, stochastic neurons) | seed-**aggregated** statistic (the mean), **never** per-sample | D | `compare_distributional` |

NEST and JAX draw from independent PRNG streams, so anything stochastic is
compared distributionally — averaged over seeds, never spike-by-spike.

## 4. Tolerance conventions (implemented)

These were once a proposal; they now live as importable, unit-aware constants
in `tolerance_conventions.py` and are consumed by `nest_compare.py`. Categories
A–E span the comparison space the parity harness covers (V_m trace, firing
rate, weight trajectory, PSC-amplitude train, F-I curve / spike timing).

| Cat | Kind | Example metric | Tolerance | Constant(s) |
|---|---|---|---|---|
| A | adaptive numerical integrator | aeif / HH / izhikevich `V_m` | `atol 1e-3 mV`, `rtol 1e-3` | `CAT_A` |
| B | analytic exact propagator | linear `iaf_psc_*` `V_m` / PSC | `atol 1e-6 mV`, `rtol 1e-6` (near-exact) | `CAT_B`, `CAT_B_ALIGNED` |
| C | conductance / coupled / mean-field | `iaf_cond_*`, `siegert_neuron` rate | `1e-3 mV` trace / `5 %` rate | `CAT_C`, `CAT_C_RATE` |
| D | distributional (PRNG-divergent) | network firing rate, ISI CV | mean `5 %`, `≥ 4` seeds | `CAT_D` |
| E | spike-time / event-count | `*_ps`, PSC peak timing, event counts | `|ΔN| ≤ 2`, `|Δstep| ≤ 1` | `CAT_E` |

- **CAT_A** — `TraceTolerance(1e-3 mV, 1e-3)`. Adaptive RKF45 integrator V_m /
  state trace.
- **CAT_B** — `TraceTolerance(1e-6 mV, 1e-6)`. Analytic exact-propagator trace
  (near-exact); the propagator should match NEST to round-off.
- **CAT_B_ALIGNED** — same analytic family but `align_steps=1` and a looser
  `5e-2 mV` atol, to tolerate a one-step multimeter recorder offset (the
  pipeline-latency case).
- **CAT_C** — `TraceTolerance(1e-3 mV, 1e-3)`. Conductance / coupled
  deterministic trace (`iaf_cond_*`, multi-compartment).
- **CAT_C_RATE** — `TraceTolerance(0.0, 5e-2)`. Mean-field rate fixed point in
  Hz, compared purely relatively.
- **CAT_D** — `DistributionalTolerance(rate_rtol=5e-2, mean_diff_pct=2e-2,
  autocorr_max_diff=5e-2, n_seeds=N_SEEDS_DEFAULT)`. PRNG-divergent multi-seed
  statistic; compare the aggregate, never per-sample.
- **CAT_E** — `SpikeTimeTolerance(max_count_diff=2, max_peak_step_diff=1)`.
  Spike-time / event-count; `±1` step ≈ `dt`.

Simulation defaults are also exported: `T_DEFAULT = 1000 ms`, `DT_DEFAULT =
0.1 ms`, `N_SEEDS_DEFAULT = 5`.

**Multi-seed protocol (category D).** Because NEST and JAX draw from independent
PRNG streams, PRNG-divergent drives are never compared per-sample. The
convention is to run `N ≥ N_SEEDS_DEFAULT` seeds per side and compare the
seed-**mean** (a handful of legacy single-realization tests use one seed, a
documented limitation).

`TraceTolerance.atol` is unit-aware: a `brainunit` Quantity in mV for voltage
traces, and a plain `float` for dimensionless / rate metrics. The `compare_*`
engine consumes these constants; `tolerance_conventions.py` itself holds no
logic.

## 5. Writing a parity test

A parity test is a plain `unittest.TestCase`. Run the NEST side and the
brainpy.state `Simulator` side however you like, hand the **computed metric** to
a comparator, and call `.assert_()`:

```python
import unittest
from brainpy_state._nest._validation.nest_compare import (
    requires_nest, compare_trace, compare_distributional,
)
from brainpy_state._nest._validation.tolerance_conventions import CAT_C_RATE, CAT_D


@requires_nest                       # skips cleanly when `import nest` fails
class TestMyModelParity(unittest.TestCase):
    def test_meanfield_rate(self):                 # deterministic -> trace mode
        bp = run_brainpy()                         # Simulator run -> a rate (Hz)
        ns = run_nest()                            # live-NEST run -> a rate (Hz)
        compare_trace(ns, bp, tol=CAT_C_RATE, metric="exc rate").assert_()

    def test_network_rate(self):                   # PRNG-divergent -> distributional
        bp = [run_brainpy(seed=s) for s in range(4)]
        ns = [run_nest(seed=s + 1) for s in range(4)]   # offset to decorrelate streams
        compare_distributional(ns, bp, tol=CAT_D, metric="exc rate").assert_()
```

Carve-outs the harness recognizes: docs-only items are exercised by doctest (no
live-NEST run); items blocked on an unlanded API ship a
`@unittest.skip("blocked on <api>")` placeholder rather than a missing test.

## 6. Genuine remaining gaps

Coverage is broad but **not** 100 %. The real residual:

- **`pp_psc_delta`** — the model is present (`brainpy_state/_nest/pp_psc_delta.py`)
  but has **no dedicated `_validation` parity test** with a live-NEST trace
  comparison. This is the headline residual: its sibling point-process model
  `pp_cond_exp_mc_urbanczik` *is* validated (cluster-21, via
  `_validation/urbanczik_synapse_parity_test.py`), but `pp_psc_delta` is not.
- **Any model whose only coverage is a unit / law test** (no live-NEST trace or
  distributional comparison). These have an analytic or self-consistency check
  but no `@requires_nest` parity test, so they are not on the same footing as
  the trace-validated families. Audit before claiming a family is "fully
  validated."

Do **not** over-claim. The harness exists and nearly every P0 from the old
roadmap has shipped, but the page should keep naming the residual above rather
than asserting universal parity.

## 7. Permanently out of scope

These are deliberate non-goals, not gaps to close:

- **Bit-exact RNG reproduction vs NEST.** NEST and JAX use independent PRNG
  streams; spike-by-spike equality of stochastic drives is not a goal. Anything
  PRNG-divergent is validated distributionally (category D) — seed-mean firing
  rate, ISI CV, covariance functions — never per-event equality.
- **MPI / distributed determinism.** Parity is validated single-process; MPI /
  distributed bit-for-bit determinism is not in scope.

## 8. Methodology note — measuring harness coverage

To measure coverage of the harness itself, use a directory-scoped run:

```bash
coverage run -m pytest \
    brainpy_state/_nest/_validation/tolerance_conventions_test.py \
    brainpy_state/_nest/_validation/nest_compare_test.py
coverage report --include='brainpy_state/_nest/_validation/*'
```

Do **not** use a dotted `--source=` / `--cov=<module>` form: pre-importing the
package under coverage's C tracer double-initializes jaxlib/absl and SIGABRTs.
Scope the report with `--include=` after a plain `coverage run` instead.

Running the suite:

```bash
# everything in the package (harness self-tests run; NEST tests run if nest present, else skip)
python -m pytest brainpy_state/_nest/_validation/ -q

# only the live-NEST parity tests
python -m pytest brainpy_state/_nest/_validation/ -m requires_nest -q
```

Without NEST installed, the `@requires_nest` tests skip cleanly and the harness
self-tests still pass.
