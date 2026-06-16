# `_validation/` — the live-NEST parity harness

The shared answer to **"did it match NEST?"** Every parity test in this package
imports the same comparison engine and the same documented tolerances, so the
question is asked the same way everywhere.

- **`tolerance_conventions.py`** — the single source of truth for *what tolerance
  for what kind of model* (categories **A–E** + the multi-seed protocol + sim
  defaults). Importable, unit-aware constants.
- **`nest_compare.py`** — the comparison engine: `requires_nest`, `compare_trace`,
  `compare_distributional`, the `nest_compare(mode=…)` umbrella, and
  `ComparisonResult.assert_()`.
- **`conftest.py`** — registers the `requires_nest` pytest marker.

## Two comparison modes

| Mode | When | How it compares | Category | Helper |
|---|---|---|---|---|
| **`trace`** | deterministic drive (same dt, fixed input, analytic/mean-field) | per-sample / max-abs error, optional ±1-step recorder alignment | A, B, C | `compare_trace` |
| **`distributional`** | PRNG-divergent drive (Poisson, random connectivity, stochastic neurons) | seed-**aggregated** statistic (the mean), **never** per-sample | D | `compare_distributional` |

NEST and JAX draw from independent PRNG streams, so anything stochastic is
compared distributionally — averaged over seeds, never spike-by-spike.

## Tolerance categories A–E

See `tolerance_conventions.py` for the authoritative values (and units). Summary:

| Cat | Kind | Example metric | Tolerance | Constant |
|---|---|---|---|---|
| A | adaptive numerical integrator | aeif/HH/izhikevich `V_m` | `atol 1e-3 mV` | `CAT_A` |
| B | analytic exact propagator | linear `iaf_psc_*` `V_m`/PSC | `atol 1e-6 mV` (`CAT_B_ALIGNED`: 5e-2 mV, 1-step) | `CAT_B` |
| C | conductance / coupled / mean-field | `iaf_cond_*`, `siegert_neuron` rate | `1e-3 mV` trace / `5 %` rate | `CAT_C`, `CAT_C_RATE` |
| D | distributional (PRNG-divergent) | network firing rate, ISI CV | mean `5 %`, `≥4` seeds | `CAT_D` |
| E | spike-time / event-count | `*_ps`, PSC peak timing | `|ΔN|≤2`, `|Δstep|≤1` | `CAT_E` |

## Writing a parity test

Plain `unittest.TestCase`. Keep the NEST run and the brainpy.state `Simulator`
run as you like; hand the **computed metric** to a comparator and call `.assert_()`.

```python
import unittest
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_C_RATE, CAT_D


@requires_nest                       # skips cleanly when `import nest` fails
class TestMyModelParity(unittest.TestCase):
    def test_meanfield_rate(self):                 # deterministic -> trace mode
        bp = run_brainpy()                         # your Simulator run -> a rate (Hz)
        ns = run_nest()                            # your live-NEST run -> a rate (Hz)
        compare_trace(ns, bp, tol=CAT_C_RATE, metric="exc rate").assert_()

    def test_network_rate(self):                   # PRNG-divergent -> distributional mode
        bp = [run_brainpy(seed=s) for s in range(4)]
        ns = [run_nest(seed=s + 1) for s in range(4)]     # offset to decorrelate streams
        compare_distributional(ns, bp, tol=CAT_D, metric="exc rate").assert_()
```

`nest_compare(nest_fn, brainpy_fn, mode=…, tol=…, seeds=…)` is a convenience that
*runs* the two callables for you and dispatches to the right comparator.

Metrics may be plain floats/arrays or `brainunit` quantities. For a unit-aware
tolerance (`CAT_A/B/C`, mV) a plain array is assumed already in that unit; a
quantity is converted. The pass test is the division-free allclose form
`|a−b| ≤ atol + rtol·|ref|`, so a zero reference never divides.

## Carve-outs

- **Distributional** — where the drive is PRNG-divergent, compare the seed-mean
  with `compare_distributional` (category D). Do **not** assert per-sample.
- **Docs-only** items — exercised by doctest; no live-NEST run. Exempt from a
  parity test.
- **Blocked** items (API not yet landed, e.g. spatial / e-prop) — ship a
  `@unittest.skip("blocked on <api>")` placeholder test, not a missing test.

## Running

```bash
# everything in the package (harness unit tests run; NEST tests run if nest present, else skip)
python -m pytest brainpy_state/_nest/_validation/ -q

# only the live-NEST parity tests
python -m pytest brainpy_state/_nest/_validation/ -m requires_nest -q

# without NEST installed: the @requires_nest tests skip cleanly; the harness unit tests still pass

# coverage of the harness itself (measure by directory — a dotted --cov pre-imports
# NEST under coverage's C tracer and can abort)
python -m pytest brainpy_state/_nest/_validation/tolerance_conventions_test.py \
                 brainpy_state/_nest/_validation/nest_compare_test.py \
                 --cov=brainpy_state/_nest/_validation --cov-report=term-missing -q
```
