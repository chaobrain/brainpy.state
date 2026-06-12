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
- **Analog recording** — a `voltmeter`/`multimeter` is connected in NEST's
  *reversed* direction (`connect(voltmeter, neuron)`, the recorder observes the
  neuron) and read back with `res.trace(rec, 'V_m')` → `(T, N)` and `res.times`.
- **Current devices** — `noise_`/`dc_`/`step_`/`ac_generator` inject a current
  (pA) through the neuron's current ring buffer (a NEST-faithful one-step delay),
  while `poisson_generator` and other spike sources deliver delayed delta events.

## Examples

Each script ports one of NEST's Brunel variants and is paired with a live-NEST
parity test (see [Validation](#validation)). Run any of them directly, e.g.:

```bash
python examples/nest/brunel_alpha.py
```

- **`brunel_alpha.py`** — Brunel (2000) random balanced network with alpha
  synapses (`iaf_psc_alpha`), a port of `brunel_alpha_nest.py`. Prints the
  excitatory/inhibitory rates and writes `brunel_alpha_raster.png`. Defaults to
  NEST's native `order=2500`, so the first run spends ~1–2 min sampling
  connectivity before simulating.
- **`brunel_delta.py`** — the delta-synapse variant (`iaf_psc_delta`), a port of
  `brunel_delta_nest.py`. Synaptic weights are membrane-voltage jumps, so they are
  given in `u.mV` rather than pA.
- **`brunel_exp_multisynapse.py`** — the multi-receptor variant
  (`iaf_psc_exp_multisynapse`), a port of `brunel_exp_multisynapse_nest.py`. Each
  neuron exposes 100 receptor ports with time constants spanning 0.1–1.09 ms, and
  every connection is routed to a uniformly-drawn port via
  `connect(..., receptor_type='uniform')`.
- **`brunel_siegert.py`** — the mean-field analysis (`siegert_neuron`), a port of
  `brunel_siegert_nest.py`. Rather than simulating spikes, it integrates three
  rate nodes (excitatory, inhibitory, drive) in pseudo-time to the self-consistent
  firing-rate fixed point and writes `brunel_siegert_relaxation.png`. This network
  is wired by hand (the spiking `Simulator` does not apply to rate units).
- **`brunel_alpha_evolution_strategies.py`** — a Natural Evolution Strategies
  optimizer (Wierstra et al. 2014) tuning `g` and `eta` of the alpha network
  toward target rate / CV / correlation, a port of
  `brunel_alpha_evolution_strategies_nest.py`. The optimizer and spike-statistics
  analysis are model-agnostic; only `simulate()` builds the `Simulator` network.

## Single- and few-neuron demos (§3.2)

Seven of NEST's single-/few-neuron tutorials, each a faithful port driven by a
live-NEST parity test (`brainpy_state/_nest/_validation/<name>_test.py`). Run any
directly, e.g. `python examples/nest/one_neuron.py`.

- **`one_neuron.py`** — an `iaf_psc_alpha` driven by a constant `I_e = 376 pA`,
  observed by a `voltmeter`. The minimal analog-recording demo.
- **`one_neuron_with_noise.py`** — the same neuron driven by a 2-channel
  `poisson_generator` (80 kHz / 15 kHz) with signed per-channel weights
  `[1.2, -1.0] pA`.
- **`twoneurons.py`** — an `I_e`-driven `iaf_psc_alpha` connected to a second
  through a static synapse (`w = 20 pA`, `d = 1 ms`); both `V_m` recorded.
- **`testiaf.py`** — charge → spike → refractory → recovery, swept over the
  resolutions `dt ∈ {0.1, 0.5, 1.0}` ms (rebuild-per-trial).
- **`balancedneuron.py`** — SciPy `bisect` tunes the inhibitory Poisson rate so
  the target neuron fires at the excitatory rate (≈ 5 Hz), re-simulating per
  trial.
- **`if_curve.py`** — the I-F surface of an `aeif_cond_exp` population driven by a
  white-noise current (`noise_generator`) across an `(I_mean, I_std)` grid.
- **`vinit_example.py`** — passive relaxation of an `iaf_cond_exp_sfa_rr` from
  several initial membrane voltages.

These ports drive four `Simulator` extensions reused by later clusters:

- **A — analog recording.** `voltmeter`/`multimeter` State taps, read via
  `res.trace(rec, recordable)` and `res.times` (all demos except as noted).
- **B — current-injecting devices.** `noise_/dc_/step_/ac_generator` inject pA
  through the neuron's current ring buffer (`if_curve`).
- **C — sweep / rebuild-per-trial.** `simulate()` re-inits all state each run, so
  a fresh `build()` + `simulate()` per trial sweeps a parameter (`testiaf`,
  `balancedneuron`, `if_curve`, `vinit_example`).
- **D — per-generator weight vectors.** `create(poisson_generator, k,
  rate=[…])` → a `k`-segment view; `connect(gen, neuron, weight=[…])` applies one
  signed weight per channel (`one_neuron_with_noise`, `balancedneuron`).

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
skip automatically when `nest` is not importable. Each test builds the same
network in live NEST and in brainpy.state and asserts the firing rate is within
5 % — a statistical comparison (the RNG streams differ), never a per-neuron
match. Representative figures:

| Port | brainpy vs NEST | rel. |
|---|---|---|
| `brunel_alpha` (`order=200`) | 56.9 vs 57.0 spks/s | 0.21 % |
| `brunel_delta` (`order=200`) | 58.5 vs 58.2 spks/s | 0.55 % |
| `brunel_exp_multisynapse` (`order=200`, full pop, 4 seeds) | 25.8 vs 24.8 spks/s | ≈4 % |
| `brunel_siegert` (mean-field, `order=2500`) | 32.03 vs 32.03 spks/s | 0.00 % |
| `brunel_alpha_evolution_strategies` (`simulate`, `N=1000`) | 51.5 vs 51.5 spks/s | 0.08 % |

The `exp_multisynapse` rate is a steep function of each neuron's randomly-drawn
receptor time constant, so its population mean has the widest spread; recording
the full excitatory population and averaging over four seeds keeps it inside the
5 % bound. The `siegert` mean-field solves the same self-consistent equation in
both simulators, so the asymptotic rate matches exactly. A manual `brunel_alpha`
check at `order=2500` lands at **28.8 vs 28.5 spks/s (0.91 %)** — the lower rate
is a genuine finite-size effect that NEST reproduces.

### Single- and few-neuron demos (§3.2)

Deterministic ports are compared per-sample against live NEST (category B,
one-step recorder alignment); Poisson/noise-driven ports are compared as a
seed-mean statistic (category D, 5 %).

| Port | metric | brainpy vs NEST |
|---|---|---|
| `one_neuron` | `V_m` charge, max\|Δ\| | 2.8e-14 mV |
| `twoneurons` | `V_m` (neuron_1 / neuron_2), max\|Δ\| | 2.8e-14 / 5.7e-14 mV |
| `testiaf` (`dt∈{0.1,0.5,1.0}`) | `V_m` max\|Δ\| ; spike count | 2.8e-14 mV ; 16 = 16 |
| `vinit_example` (5 initial `V_m`) | relaxation, worst max\|Δ\| | 1.4e-14 mV |
| `one_neuron_with_noise` (4 seeds) | firing rate | 45.8 vs 46.0 spks/s (0.5 %) |
| `if_curve` (`I=700,σ=0`) | rate, deterministic | 9.0 vs 9.0 spks/s |
| `if_curve` (noisy, 4 seeds) | rate (e.g. `I=900,σ=200`) | 23.8 vs 23.9 spks/s |
| `balancedneuron` | bisected inhibitory rate | 20.81 vs 20.81 Hz |

The deterministic single-neuron traces match NEST to ~1e-14 mV because
`iaf_psc_alpha`'s exact propagator and the analog State tap are bit-faithful; the
`balancedneuron` objective is steep near the root, so the bisected inhibitory rate
is identical to NEST despite the PRNG-divergent Poisson drive.
