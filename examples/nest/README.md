# NEST-style ports

Reference networks from the [NEST simulator](https://www.nest-simulator.org/)
ported onto **brainpy.state**'s explicit `Simulator` API. The goal is twofold:
demonstrate a NEST-flavored network-construction syntax that drives the real
NEST-compatible models (`iaf_psc_alpha`, `poisson_generator`, `spike_recorder`,
…), and validate those models by reproducing published benchmarks against a live
NEST run.

## The `Simulator` API

The API mirrors NEST's vocabulary — there is no global kernel; a `Simulator`
owns the populations, devices, and connections:

```python
import saiunit as u
from brainpy_state import (
    Simulator, fixed_indegree, all_to_all,
    iaf_psc_alpha, poisson_generator, spike_recorder,
)

sim = Simulator(dt=0.1 * u.ms)
ne = sim.create(iaf_psc_alpha, 800, params=npar)     # population (NodeView)
ni = sim.create(iaf_psc_alpha, 200, params=npar)
noise = sim.create(poisson_generator, rate=p_rate * u.Hz)
esr = sim.create(spike_recorder)

sim.connect(noise, ne, weight=J_ex * u.pA, delay=1.5 * u.ms, rule=all_to_all)
sim.connect(ne, ne + ni, weight=J_ex * u.pA, delay=1.5 * u.ms,   # population algebra
            rule=fixed_indegree(80), seed=1)
sim.connect(ne[:50], esr)                            # slice + record

res = sim.simulate(1000.0 * u.ms)
print(res.rate(esr.segments[0].population))          # spks/s
```

Key pieces:

- **`NodeView` algebra** — `ne + ni` concatenates populations and `ne[:50]`
  slices one, so a single `connect` can target a combined or partial population.
- **Connection rules** — `all_to_all`, `one_to_one`, `fixed_indegree(K)`.
- **`connect(..., weight=, delay=)`** — weights are synaptic currents in pA
  (signed: positive excitatory, negative inhibitory), delivered as delayed delta
  events; `delay=` is a homogeneous axonal delay.
- **Generators fan out** to one independent train per target neuron, matching
  NEST; recorders are read as stacked arrays after the run.

## Examples

- **`brunel_alpha.py`** — Brunel (2000) random balanced network with alpha
  synapses, a port of NEST's `brunel_alpha_nest.py`. Run it directly:

  ```bash
  python examples/nest/brunel_alpha.py
  ```

  It prints the excitatory/inhibitory rates and writes
  `brunel_alpha_raster.png`. It defaults to NEST's native `order=2500`, so the
  first run spends ~1–2 min sampling connectivity before simulating.

## `order` and `comm`

`order` sets the network size (`NE = 4·order`, `NI = order`). `build(order=...,
comm=...)` accepts `comm='sparse'` (the default) or `comm='dense'`:

- `comm='sparse'` routes the recurrent `fixed_indegree` connectivity through a
  `brainevent` CSR event matmul, so memory stays light (~1.9 GB at `order=2500`)
  and the flagship runs at NEST's native size.
- `comm='dense'` materialises a full weight matrix — fine for small networks,
  but the `order=2500` recurrent matrices would need several GB.

Both paths are built from the same sampler and seed, so they produce
bit-identical results. Construction cost is dominated by the `fixed_indegree`
sampler, which is `O(NE + NI)`; that is what makes the large-`order` build slow,
not the sparse comm.

## Validation

Live-NEST parity tests live in
[`brainpy_state/_nest/_validation/`](../../brainpy_state/_nest/_validation) and
skip automatically when `nest` is not importable. The Brunel network test
asserts the excitatory rate is within 5 % of live NEST. The committed test runs
at `order=200` (**56.9 vs 57.0 spks/s, 0.21 %**); a manual `order=2500` check
lands at **28.8 vs 28.5 spks/s (0.91 %)** — the lower rate is a genuine
finite-size effect that NEST reproduces.
