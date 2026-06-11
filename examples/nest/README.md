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
  `brunel_alpha_raster.png`.

## A note on `order`

`order` sets the network size (`NE = 4·order`, `NI = order`). The default
`order=400` keeps the dense event projection memory-light. The dynamics are
scaled (à la Brunel) so the mean rates are essentially order-independent, which
is what the live-NEST parity test exploits (it compares at `order=200`). Running
at NEST's native `order=2500` waits on the sparse-connectivity communication
path; until then, prefer small-to-moderate `order`.

## Validation

Live-NEST parity tests live in
[`brainpy_state/_nest/_validation/`](../../brainpy_state/_nest/_validation) and
skip automatically when `nest` is not importable. The Brunel network test
asserts the excitatory rate is within 5 % of live NEST — it currently lands at
**56.9 spks/s vs 57.0 spks/s (0.21 %)** at `order=200`.
