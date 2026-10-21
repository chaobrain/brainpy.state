# v0.1.1 goal 01 — correctness fixes

_Date: 2026-10-21_

Part of [roadmap](2026-10-21-roadmap.md) Phase 0. One worktree branch `v011/01-correctness`; may be
split into one PR per section if it grows. **Test-first (CLAUDE.md #4).**

## A. Reproduced bugs (all fail on `main` @ 3153f74)

Each test below fails today; move it into the matching `*_test.py` and fix until green.

| # | bug | location | expected |
|---|---|---|---|
| 1 | `*Ref` models with default `spk_reset='soft'` fire on every step of the refractory window (22 consecutive spikes in the repro) — V is held at a soft-reset value that can stay above `V_th` | `_brainpy/lif.py` (`ExpIFRef` ~586-621, `AdExIFRef` ~980-1018, every `*Ref`) | ≤ 1 spike per `tau_ref`; hold V at `V_reset` while refractory (or default `spk_reset='hard'` for the `*Ref` models) |
| 2 | `MgBlock` defaults (`E`, `V_offset`, `alpha`) are unitless → `TypeError` with unit-carrying V | `_brainpy/synouts.py:285-289` | defaults `0*mV`, `alpha=0.062/mV` |
| 3 | `dist.LogNormal.sample` has no unit handling → `TypeError` | `_dist.py:60` | **design decision needed**: `mean`/`std` are log-space parameters; either reject Quantities with a clear error or add `unit=` so samples are `exp(...) * unit` |
| 4 | `spike_generator` merges duplicate spike times in one step into one spike | `_nest_device/spike_generator.py:286+` | emit multiplicity (NEST: one event per duplicate); with weights, sum them |
| 5 | Autapse exclusion uses slice-local indices: `pop[0:5] → pop[5:10]` with `allow_autapses=False` drops 5 real edges | `connectivity.py:95-96,117-121,154-158,190-196,221-223`; `_nest_spatial/rule.py:79-80`; `simulator.py:1484,1505,797,1222,1283,1385` | compare global neuron ids (pass slice offsets into samplers) |
| 6 | `get_connections` on the dense path rebuilds edges via `np.nonzero(W)` → zero-weight edges vanish (0 instead of 6), multapses collapse | `connection_introspection.py:256-260`, `event_proj.py:186` | keep the sampled `(pre, post)` edge list on the projection |
| 7 | `BRAINPY_STATE_ALLOW_DIRECT_IMPORT=0` counts as "allowed" | `_namespace.py:100` | accept only `1/true/yes` |

```python
# repro tests (as run on 2026-10-21)
import jax.numpy as jnp, numpy as np, brainstate, brainunit as u
from brainpy_state import (ExpIFRef, MgBlock, dist, spike_generator, Simulator,
                           iaf_psc_exp, all_to_all)
from brainpy_state import _namespace

def test_expifref_soft_reset_no_burst_in_refractory():
    brainstate.environ.set(dt=0.1 * u.ms)
    n = ExpIFRef(1, tau_ref=2. * u.ms); n.init_state()
    def step(i):
        with brainstate.environ.context(t=i * 0.1 * u.ms):
            return n.update(50. * u.mA)
    s = np.asarray(brainstate.transform.for_loop(step, jnp.arange(200))).ravel() > 0
    run = max(len(r) for r in ''.join('1' if x else '0' for x in s).split('0'))
    assert run <= 1

def test_mgblock_default_with_unit_voltage():
    MgBlock(E=0. * u.mV).update(1. * u.nS, -65. * u.mV)

def test_lognormal_with_units():
    dist.LogNormal(1. * u.nS, 0.5 * u.nS).sample((3,), brainstate.random.split_key())

def test_spike_generator_duplicate_times():
    brainstate.environ.set(dt=0.1 * u.ms)
    sg = spike_generator(spike_times=[1.0, 1.0] * u.ms); sg.init_state()
    with brainstate.environ.context(t=1.0 * u.ms):
        assert float(np.asarray(sg.update()).ravel()[0]) == 2.0

def test_autapse_exclusion_on_disjoint_slices():
    sim = Simulator(dt=0.1 * u.ms); pop = sim.create(iaf_psc_exp, 10)
    proj = sim.connect(pop[0:5], pop[5:10], rule=all_to_all, weight=1. * u.pA,
                       allow_autapses=False, comm='dense')
    assert len(np.asarray(proj.realized_edges().source)) == 25

def test_get_connections_keeps_zero_weight_edges():
    sim = Simulator(dt=0.1 * u.ms); pop = sim.create(iaf_psc_exp, 3)
    sim.connect(pop, pop, rule=all_to_all, weight=0. * u.pA,
                allow_autapses=False, comm='dense')
    assert len(sim.get_connections()) == 6

def test_allow_env_zero_is_not_allowed():
    assert not _namespace._is_internal_access(
        ["__main__"], {}, {_namespace.ALLOW_DIRECT_IMPORT_ENV: "0"})
```

## B. High-confidence (write the repro first; drop the item if it passes)

| bug | location |
|---|---|
| AMPA/GABAa/BioNMDA saturation `(1 * u.get_unit(g) - g)` depends on the unit chosen for `g` (1 mS vs 1 nS) | `_brainpy/synapse.py:262,541,545` |
| Wiener/OU inputs use `sigma*sqrt(dt_mag)` with the unit dropped → amplitude changes ms vs s | `_brainpy/inputs.py:1010,1058` |
| `align_pre_projection`/`align_post_projection` call `fun(*x)` on plain `State`s (not callable); use `call_module`; also give `super().__init__()` a name | `_brainpy/projection.py:664,678,768,781` |
| `PoissonInput` docstring example passes `neuron.V` (a State) where a `Prefetch` is required | `_brainpy/inputs.py:~524,559` |
| `fixed_total_number`: `allow_multapses=False` dedups to < N edges; autapse removal does one resample pass then raises | `_nest_network/connectivity.py:190-208` |
| `LeakySpikeReadout` caches `decay` from dt at construction | `_brainpy/readout.py:121` |
| `synaptic_projection.py` validates with `assert` (stripped by `python -O`) | `_brainpy/synaptic_projection.py` |
| `validate_receptor_type` truncates via `int()` silently | `_nest_base/plastic_base.py` |

## Edge cases to cover
Batched states (`batch_size`); unit-less vs unit-carrying inputs; under `jit`/`for_loop`
(not just eager); slices that overlap partially (`pop[0:5] → pop[2:7]`); concatenated
views; `comm='sparse'` and `'dense'` agreeing; `spike_generator` with weights + duplicates.
