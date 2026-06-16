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
import brainunit as u
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
  `brunel_siegert_nest.py`. Rather than simulating spikes, it relaxes three rate
  nodes (excitatory, inhibitory, drive) to the self-consistent firing-rate fixed
  point and writes `brunel_siegert_relaxation.png`. The nodes are coupled by
  `diffusion_connection` (drift → μ, diffusion → σ²) and relaxed end-to-end
  through the `Simulator` — one compiled `for_loop` over the rate dynamics, not a
  Python step loop. Six convergent edges (drive/ex/in into each of ex/in,
  including the ex→ex and in→in population self-coupling) accumulate into each
  node's μ / σ², matching NEST's `diffusion_connection`.
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

## Plasticity demos (§3.3)

All five of NEST's plasticity tutorials, each a faithful port driven by a
live-NEST parity test (`brainpy_state/_nest/_validation/<name>_test.py`). Run any
directly, e.g. `python examples/nest/evaluate_tsodyks2_synapse.py`.

- **`clopath_synapse_spike_pairing.py`** — the canonical voltage-based STDP
  (Clopath 2010) protocol: a pre train paired with a post train across pairing
  frequencies 10–50 Hz in both orderings, onto an `aeif_psc_delta_clopath`
  through a single `clopath_synapse`. The stored weight is read with
  `res.weight_trace` and the *normalised weight change* is plotted against pairing
  frequency. `aeif_psc_delta_clopath` is a **delta** neuron, so the weight is in
  `u.mV`.
- **`evaluate_tsodyks2_synapse.py`** — the deterministic Tsodyks-Markram rule
  (`tsodyks2_synapse`) in a depression (`U=0.67`) and a facilitation (`U=0.1`)
  regime. A 50 ms-ISI burst plus a recovery pair drives a single edge onto a
  linear, never-spiking `iaf_psc_exp` post (`V_th = 1e4` mV), so the post `V_m`
  **is** the PSC-amplitude train — shrinking under depression, growing under
  facilitation.
- **`evaluate_quantal_stp_synapse.py`** — the *stochastic* quantal variant
  (`quantal_stp_synapse`): each edge has `n` release sites and every spike
  releases `Binomial(available, u)` of them. The per-run `seed` is forwarded to
  `connect` and keys the release PRNG, so realizations differ across seeds and the
  seed-mean tracks the deterministic `tsodyks2` envelope (`weight = n·w`, plotted
  as the limit line).
- **`clopath_synapse_small_network.py`** — a small all-to-all recurrent
  `aeif_psc_delta_clopath` population (no autapses) whose neurons are spike-clamped
  to fire in a staggered order (`0→1→2`) each cycle. Each directed edge then sees
  the spike-pairing protocol, so forward edges (pre before post) potentiate and
  backward edges depress; the recurrent weight **matrix** evolution is recorded
  with `res.weight_trace` and shows a feedforward chain emerging.
- **`urbanczik_synapse_example.py`** — the Urbanczik-Senn dendritic prediction-error
  rule (Urbanczik & Senn 2014, Fig. 1B). A two-compartment
  `pp_cond_exp_mc_urbanczik` neuron has its *soma* driven by a time-varying
  conductance teacher while a fixed Poisson pattern drives the *dendrite* through
  plastic `urbanczik_synapse` edges; the dendritic weights adapt so the dendritic
  prediction `V_W*` reproduces the somatically-imposed signal — the rate prediction
  error `|phi(U) - phi(V_W*)|` shrinks over training. The rule reads the post
  neuron's **dendritic** prediction error `delta_Pi` per edge through the post-state
  reader (primitive #2), validated against live NEST in
  `urbanczik_synapse_parity_test.py`; the demo's end-to-end learning is asserted by
  `urbanczik_synapse_example_test.py`.

These ports add two reusable seams on top of the §3.2 / §3.4 vocabulary:

- **F — plastic projections + weight recording.** `connect(..., synapse=<plastic
  rule>)` dispatches to the plastic-projection primitives — `EventPlasticProj`
  (event-driven; the two STP rules) and `VoltageCoupledPlasticProj` (the
  post-state reader). The post-state reader pulls **named** post States each step:
  Clopath reads the `aeif_psc_delta_clopath` analog voltages, and Urbanczik reads
  the `pp_cond_exp_mc_urbanczik` *dendritic* prediction error `delta_Pi` (the
  rule declares `post_state_reads`, so the same primitive serves a somatic and a
  dendritic-compartment reader without change). `record_weight(proj)` taps the
  per-edge weight, read post-run as `res.weight_trace(proj)` → `(T, E)` in CSR
  (sorted-by-pre) order. The Clopath and Urbanczik demos record the weight
  directly; the STP demos read the plasticity through the post `V_m` (the
  PSC-amplitude train, seam **A**).
- **G — stochastic seed threading.** `connect(..., seed=k)` keys the per-edge
  release PRNG and survives `simulate`'s `init_all_states`, so a stochastic rule
  (`quantal_stp_synapse`) is reproducible per seed; parity is then **distributional**
  (seed-mean, category D).

In NEST a plastic synapse cannot be driven by a device, so a `parrot_neuron`
relays the train; the `Simulator` `spike_generator` drives the plastic edge
directly with a one-step (0.1 ms) holder lag, so the NEST reference's relay delay
is set to the matching **0.1 ms** (the `RELAY_D` convention) to align delivery
before comparison.

## Recording & device demos (§3.4)

Five of NEST's recording/device tutorials, each paired with a live-NEST parity
test (`brainpy_state/_nest/_validation/<name>_test.py`). Run any directly, e.g.
`python examples/nest/recording_demo.py`.

- **`multimeter_file.py`** — a `multimeter` records three analog recordables
  (`V_m`, `I_syn_ex`, `I_syn_in`) from an `iaf_psc_exp` driven by two
  `spike_generator`s (excitatory `+80 pA`, inhibitory `−40 pA`). The upstream
  records `V_m`/`g_ex`/`g_in` from a conductance `iaf_cond_alpha` to an `ascii`
  file. The spike→conductance routing seam is now **unblocked**: the conductance
  family (`iaf_cond_alpha` and its seven siblings) accepts spike-driven
  `receptor_type=1`→`g_ex` / `=2`→`g_in` input through the `w_by_rec` multi-receptor
  bridge (goal 25), so this demo could now port to its exact upstream conductance
  `iaf_cond_alpha` (out of scope here). Only the file backend (`devices-gap.md` P2)
  is still missing, so this stays the **in-memory, current-based** equivalent —
  same demo shape, traces read with `res.trace(mm, name)`.
- **`recording_demo.py`** — the recording-API tour: a `poisson_generator` (1 MHz,
  refractory-saturating) drives an `iaf_psc_exp` into a `spike_recorder` and a
  `multimeter`. NEST's `record_to` backend axis (ascii vs memory) collapses to
  in-memory; the `time_in_steps` axis is reproduced post-hoc by `read_spikes`,
  which returns spikes as integer step indices or in ms.
- **`cross_check_mip_corrdet.py`** — a `mip_generator` (one shared parent Poisson,
  per-child copy probability) emits two correlated trains whose cross-correlogram
  is computed **two independent ways** — the built-in `correlation_detector` and a
  hand-written `corr_spikes_sorted` reference — and cross-checked bit-for-bit. Both
  devices are imperative host devices, so the demo runs **eagerly** (post-hoc, no
  `for_loop`).
- **`correlospinmatrix_detector_two_neuron.py`** — two coupled binary neurons (a
  `ginzburg_neuron` driving a `mcculloch_pitts_neuron`) whose spin trains feed a
  `correlospinmatrix_detector`; the demo reads back per-channel mean activities and
  the `2×2` covariance matrix. The neurons run in one `for_loop` (the `n1→n2`
  coupling reads `n1`'s pre-update spin); the detector is driven eagerly from the
  recorded trains.
- **`precise_spiking.py`** — the grid model `iaf_psc_exp` vs its precise twin
  `iaf_psc_exp_ps` under the same DC drive across resolutions `dt ∈ {0.1, 0.5,
  1.0}` ms. The grid model fires *on* the resolution grid; the precise model
  resolves the spike **between** grid points (read from `last_spike_time`), so its
  firing period barely moves with `dt`.

Two further §3.4 demos extract realized connectivity **after** wiring, via
`Simulator.get_connections` — the `GetConnections` / `SynapseCollection` idiom
(enumerate realized synapses and read or write per-edge
`weight`/`delay`/`source`/`target` without holding each `Projection` handle):

- **`plot_weight_matrices.py`** — an E/I network wired with `fixed_indegree`
  (excitatory weights `Normal(20, 0.5)` pA, inhibitory `−g` times as large); for
  each of the four `E→E / E→I / I→E / I→I` population pairings it enumerates the
  realized edges with `get_connections(source, target)` and scatters
  `W[source, target] += weight` into a dense weight matrix. Population-local
  `source` / `target` indices replace NEST's global-node offset; multapses sum into
  a cell exactly as NEST's `W[i, j] += w`.
- **`synapsecollection.py`** — the `SynapseCollection` introspection tour:
  `one_to_one`, an `all_to_all` block with `Uniform(0.5, 4.5)` pA weights, and a
  five-rule complex network, queried by `get_connections()` (every edge),
  `get_connections(source, target)` (a population slice) and
  `get_connections(synapse=model)` (one synapse model), with
  `get(['source', 'target', 'weight'])` batch reads and `set('weight', …)` per-edge
  write-backs.

These ports add two reusable seams on top of the §3.2 vocabulary:

- **E — eager imperative devices.** `mip_generator`, `correlation_detector`, and
  `correlospinmatrix_detector` are NumPy-RNG / Python-loop host devices that cannot
  enter a JAX `for_loop`. The pattern is to obtain the spike data first (a device
  `.simulate(n_steps)` multiplicity matrix, or a State-tapped binary train) and
  then drive the detector **post-hoc**, feeding only event-carrying steps and
  stamping each at `step + 1` (NEST's one-step delivery latency, which cancels in
  the lag difference). Nothing imperative runs inside the `for_loop`.
- **F — post-hoc connection introspection.** `get_connections(source, target,
  synapse)` returns a lazy `SynapseCollection` over the realized edges of the
  matching projections, re-reading `weight` / `delay` live on each `get` and writing
  them back on `set`. One uniform `source` / `target` / `weight` / `delay` view spans
  every edge-storage family (dense matrix, sparse CSR, per-receptor scatter, the
  `one_to_one` scalar, and the plastic CSR projections); per-edge weight writes are
  refused on homogeneous-weight or rule-managed (weight-evolving) projections, and a
  set delay grid-rounds to the resolution as NEST stores it.

## Single-neuron model demos (§3.5)

Sixteen of NEST's single-neuron model tutorials (NEST §3.5): most are faithful
ports driven by a live-NEST trace-parity test
(`brainpy_state/_nest/_validation/<name>_test.py`); the stochastic and mean-field
ones use a documented analytic or distributional carve-out (where the PRNG streams
diverge, the closed form or the seed-averaged statistics are the ground truth).
Run any directly, e.g. `python examples/nest/glif_cond_neuron.py`.

- **`hh_psc_alpha.py`** — the Hodgkin–Huxley neuron (`hh_psc_alpha`) under a
  sub-rheobase bias current, recording `V_m` and the `m`/`h`/`n` gating variables
  (NEST `Act_m`/`Inact_h`/`Act_n`), plus a supra-threshold F–I curve sweep.
- **`hh_phaseplane.py`** — a phase-plane **analysis** carve-out: the reduced
  `(V, n)` vector field of the HH neuron with `m` frozen, its nullclines (checked
  against the closed-form `n_inf(V)`), and a relaxation trajectory to the resting
  fixed point. An analysis demo, not a spike-train port.
- **`aeif_cond_beta_multisynapse.py`** — the adaptive-exponential neuron with four
  beta-function conductance receptors at distinct reversal potentials; the first
  demo to exercise the multi-receptor routing seam `connect(receptor_type=k)`.
- **`gif_cond_exp_multisynapse.py`** — the generalized-IF neuron with two
  exponential-conductance receptors of opposite reversal potential, recording the
  `E_sfa`/`I_stc` adaptation variables.
- **`glif_cond_neuron.py`** — the Allen-Institute conductance-based GLIF neuron at
  all five mechanism levels (`lif`, `lif_r`, `lif_asc`, `lif_r_asc`, `lif_r_asc_a`)
  under four stimulation paradigms (400 pA step current, excitatory/inhibitory
  receptor spikes, and a 15 kHz Poisson window relayed through a `parrot_neuron`).
- **`glif_psc_neuron.py`** — the current-based GLIF counterpart (exact propagator
  matrices, alpha-PSC synapses), the same five levels and four paradigms (the
  150 kHz Poisson window exercises spike multiplicity through the parrot),
  recording the injected current `I` and summed synaptic current `I_syn`.
- **`glif_psc_double_alpha_neuron.py`** — `glif_psc` with a *double* alpha synaptic
  kernel (a fast alpha plus `amp_slow`×a slow alpha), comparing the single- vs
  double-alpha synaptic-current shape across three receptor ports.
- **`iaf_tum_2000_short_term_depression.py`** / **`…_facilitation.py`** — two
  `iaf_tum_2000` neurons with Tsodyks–Markram STP integrated *presynaptically*: a
  DC-driven presynaptic neuron relays its graded released efficacy `weight·(u·x)`
  through `receptor_type=1` to a post neuron whose sub-threshold `V_m` is recorded.
  A large `U` with `tau_fac=0` depresses successive EPSPs; a small `U` with a
  non-zero `tau_fac` facilitates them. First demo to use the STP-emission seam.
- **`izhikevich.py`** — the Izhikevich two-variable neuron in the four canonical
  regimes (`RS`/`IB`/`CH`/`FS`) under a constant current, recording `V_m` and `U_m`.
- **`mat_psc_exp.py`** — the multi-timescale adaptive-threshold neuron
  (`mat2_psc_exp`, plus an `amat2_psc_exp` config with the active voltage-dependent
  `V_th_v` component), recording the non-resetting `V_m` and the composite moving
  threshold `V_th = omega + V_th_1 + V_th_2 [+ V_th_v]`.
- **`mc_neuron.py`** — the three-compartment `iaf_cond_alpha_mc` driven through all
  nine receptors: per-compartment current pulses (receptors 7-9), per-compartment
  excitatory/inhibitory spike trains (receptors 1-6), and a somatic rheobase;
  records per-compartment `V_m.{s,p,d}` and `g_ex/g_in.{s,p,d}`. Exercises the full
  device→compartment-receptor routing seam.
- **`CampbellSiegert.py`** — a mean-field **analysis** carve-out: an `iaf_psc_alpha`
  population under Poisson drive whose free-membrane mean/variance (Campbell's
  theorem) and firing rate (Siegert's approximation) are computed analytically and
  cross-checked against the Simulator's empirical statistics.
- **`BrodyHopfield.py`** — a distributional **phase-locking** carve-out: 1000
  `iaf_psc_alpha` neurons under a shared 35 Hz sub-threshold oscillation, independent
  noise, and a per-neuron DC bias ramp; spikes synchronize to the oscillation
  (measured by vector strength), and the synchrony vanishes without it.
- **`gif_population.py`** — a microscopic GIF network: 100 `gif_psc_exp` neurons with
  recurrent `fixed_indegree(30)` coupling and a Poisson drive, run in one compiled
  `for_loop`; spike-frequency adaptation drives a population-rate oscillation that
  recurrence sharpens.
- **`gif_pop_psc_exp.py`** — the same finite two-population GIF network simulated two
  ways — mesoscopically (the host-side `gif_pop_psc_exp` population-rate model) and
  microscopically (an 800+200 `gif_psc_exp` network on the Simulator) — shown to give
  the same population activity `A_N(t)`. The mesoscopic half runs in a host-side loop
  (`gif_pop_psc_exp` is host-side NumPy, not JAX-traceable).

These ports add four seams on top of the §3.2/§3.4 vocabulary:

- **F — multi-receptor routing.** `connect(pre, neuron, receptor_type=k)` deposits a
  spike train into a specific 1-based receptor port `k` of a multi-receptor neuron.
  Conductance-based models (`aeif_cond_beta_multisynapse`,
  `gif_cond_exp_multisynapse`, `glif_cond`) receive the input through a per-port
  `w_by_rec` bridge whose unit (`nS`) is taken from the model; current-based GLIF
  models (`glif_psc`, `glif_psc_double_alpha`) pull each port with a keyed
  `sum_delta_inputs(label='receptor_k')` (`pA`). Per-port recordables (`g_1..g_4`,
  `I_syn`, `I`, `threshold`/`threshold_spike`/`threshold_voltage`, `ASCurrents_sum`)
  resolve by name through a recordable-alias table (tuple-of-candidates or a
  derived-value callable).
- **G — spike-multiplicity relay.** `parrot_neuron` repeats every incoming spike
  *including multiplicity* — a high-rate `poisson_generator` whose per-bin count
  exceeds one is relayed as that count, not collapsed to a single event — so a
  Poisson window drives a chosen receptor port at NEST-faithful magnitude via
  `poisson_generator → parrot_neuron → connect(receptor_type=k)`.
- **H — presynaptic-STP emission.** When a presynaptic model carries the class attr
  `_emission_attr` (e.g. `iaf_tum_2000`'s `spike_offset`) and is wired with
  `connect(pre, post, receptor_type=1)` (its `TSODYKS` receptor), the Simulator
  delivers the *graded released efficacy* `weight·(u·x)` as a plain pA delta input —
  not the binary spike — so the short-term plasticity, which lives in and is gated by
  the presynaptic neuron, reaches the post at NEST-faithful magnitude. A plain
  (receptor-0) connection from the same neuron still delivers the binary spike.
- **I — device→compartment-receptor routing.** `connect(device, mc, receptor_type=k)`
  routes a current or spike device into a specific compartment-receptor of a
  multi-compartment neuron (`iaf_cond_alpha_mc`): 1-based `k ∈ 1..6` are the
  per-compartment excitatory/inhibitory spike ports, `k ∈ 7..9` the per-compartment
  current ports. Per-compartment recordables (`V_m.{s,p,d}`, `g_ex/g_in.{s,p,d}`)
  resolve by compartment index through the recordable-alias table.

## Network demos (§3.6)

Six of NEST's spiking-network tutorials (NEST §3.6), each a faithful port driven
by a live-NEST **distributional** parity test — population rate, synchrony,
perturbation divergence, or winner-take-all decision within a documented band —
plus a no-NEST companion that runs in CI. Run any directly, e.g. `python
examples/nest/brette_et_al_2007.py`.

- **`repeated_stimulation.py`** — a `poisson_generator` gated to an active window
  (`start`/`stop`) drives a neuron across repeated trials (NEST's `origin` shift);
  the per-trial spike count inside the window tracks NEST (≈ `rate·(stop−start)`),
  and a zero-rate trial is silent.
- **`artificial_synchrony.py`** — a population of `iaf_psc_alpha` neurons fanned
  out across the sub-threshold band and recurrently coupled; the Golomb–Rinzel
  synchrony measure `Σ = var_t(mean_n V) / mean_n(var_t V)` rises monotonically with
  coupling strength. The brainpy port is the fixed-`dt` **grid** branch of NEST's
  demo, so it reproduces NEST's grid (not precise) synchrony curve.
- **`sensitivity_to_perturbation.py`** — a Brunel-style sparse balanced
  (`iaf_psc_delta`) network in the asynchronous-irregular state, run twice from the
  same seed with a **single extra spike** injected at `t_stim`; the two runs are
  identical before the perturbation and decorrelate across most of the network
  after it — the spiking-network signature of deterministic chaos.
- **`ei_clustered_network.py`** — a Litwin-Kumar/Rostami `iaf_psc_exp` balanced
  network whose E and I populations are split into `Q` clusters with in-cluster
  synapses potentiated (`J+`) and out-cluster depressed (`J−`). Above a clustering
  threshold the clusters spontaneously wax and wane (winner-take-all), raising
  across-cluster rate heterogeneity and irregularity over the homogeneous control.
- **`brette_et_al_2007.py`** — the integrate-and-fire benchmarks 1 (**COBA**,
  `iaf_cond_exp`) and 2 (**CUBA**, `iaf_psc_exp`) of the Brette et al. (2007)
  simulator review (Vogels–Abbott self-sustained E/I network), one consolidated
  script building both. A brief Poisson kick ignites the network, which then
  self-sustains an asynchronous-irregular state after the kick ends. The
  Hodgkin–Huxley variant (benchmark 3) is the sibling
  `examples/brainpy_like/106_COBA_HH_2007.py`.
- **`wang_decision_making.py`** — Wang's (2002) two-population winner-take-all
  decision network of `iaf_bw_2001` conductance neurons: two selective excitatory
  pools (A/B) with strong recurrent NMDA+AMPA self-excitation (`w+`) and mutual /
  background depression (`w−`), a shared inhibitory pool, and a coherence-modulated
  stimulus (two `inhomogeneous_poisson_generator`s with rates `μ0·(1 ± c)`). After
  the stimulus the winning pool latches into a high-rate attractor while the loser
  is suppressed; the winner follows the coherence bias and is noise-driven (either
  pool can win) at zero coherence.

These ports add a conductance-LIF receptor seam, a recurrent presynaptic-gated NMDA
path, and a Bernoulli connection rule on top of the Brunel-family population
vocabulary:

- **COBA receptor routing.** `connect(E, ne+ni, receptor_type=1)` /
  `connect(I, ne+ni, receptor_type=2)` route excitation into `g_ex` and inhibition
  into `g_in` of `iaf_cond_exp` through the multi-receptor `w_by_rec` bridge (the
  §3.5 seam F, now extended to a conductance LIF). The inhibitory **conductance**
  weight is a *positive* magnitude (`67 nS`); the reversal does the sign. CUBA's
  `iaf_psc_exp` instead splits excitation/inhibition by weight **sign** internally
  (inhibitory weight negative, no receptor) — matching NEST on both sides.
- **`pairwise_bernoulli(p)`.** A named connection rule wiring each pre→post pair
  independently with probability `p` — the random-balanced-network primitive behind
  the clustered and perturbation demos.
- **Recurrent presynaptic-gated NMDA.** `iaf_bw_2001`'s NMDA gate is driven by a
  *presynaptic* graded emission (`spike_offset = k0 + k1·s_NMDA_pre`), not the binary
  spike. `connect(pool, pool, receptor_type=NMDA, comm='dense')` routes that graded
  emission through the dense matmul into each post's NMDA delta channel, while AMPA/GABA
  edges from the same pre stay binary on their own channels. A live-NEST recurrent
  micro-parity (`iaf_bw_2001_recurrent_nmda_parity_test.py`) confirms this reproduces
  NEST's recurrent `Connect(pool, pool, {receptor_type: 3})` gate to machine precision
  across asymmetric per-neuron routing — so no bespoke offset-aware event projection is
  needed (`comm='sparse'` is rejected, as it would binarize the graded value).

Parity is **distributional by construction**: these networks are chaotic, balanced,
or metastable, so their PRNG streams diverge from NEST's and a per-neuron match is
meaningless. Each test compares a seed-**mean** (or, for `ei_clustered`, a seed-
**median** robust to a rare globally-synchronized seed) of a population observable
within a documented band, and asserts the qualitative law the demo exists to show.

### Rate-based network demos

NEST's §3.6 **rate-based network** demos are ported too (cluster 17). Unlike the spiking
demos above, rate neurons couple **continuously** — each emits a graded `rate` that
connections deposit (`comm='dense'`) into the post's input sum along the continuous-emission
seam (the rate-neuron families + instantaneous/delayed `rate_connection_*` landed in cluster
15a). Because the linear dynamics have an analytic fixed point, parity here is **tighter**
than the spiking demos: closed-form and deterministic-NEST anchors, not only distributional
bands.

- **`lin_rate_ipn_network.py`** — an excitatory (`NE = 4·order`) and an inhibitory
  (`NI = order`) population of `lin_rate_ipn` neurons with **delayed excitatory** and
  **instantaneous inhibitory** connections, relaxed end-to-end through the `Simulator` (one
  compiled `for_loop`). With `sigma > 0` the net fluctuates about its mean-field fixed point
  `r* = (λI − W)⁻¹μ`; a small deterministic (`sigma = 0`) instrument net matches that closed
  form **and** live NEST (`use_wfr=False`) tightly, and the per-neuron trajectory matches
  NEST once `align_steps` absorbs the uniform pipeline+delay offset. NEST's `fixed_outdegree`
  wiring is mapped to the mean-field-equivalent `fixed_indegree` (`K_in = N_src·K_out/N_tgt`).
- **`rate_neuron_dm.py`** — two mutually-inhibiting `lin_rate_ipn` units form a rectified
  winner-take-all decision circuit. Evidence is each unit's mean input `μ`; a positive bias
  `dE` selects the higher-`μ` unit (winner → `10·μ_win`, loser rectified to 0), and at
  `dE = 0` the input noise breaks the tie. The deterministic decision matches NEST exactly;
  the noisy decision is matched **distributionally** (direction, winner-loser contrast,
  zero-bias balance over seeds), since the WTA attractor amplifies any PRNG divergence.

Only NEST's `ht_neuron` intrinsic-currents demos (`intrinsic_currents_*`) remain unported —
they need a single-neuron intrinsic-currents primitive, not continuous network coupling.

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

## Generator-pattern demos (§3.7)

Three of NEST's stimulation-device tutorials (NEST §3.7), each a faithful port with a
live-NEST **distributional** parity test plus a no-NEST companion that runs in CI. Run
any directly, e.g. `python examples/nest/pulsepacket.py`.

- **`sinusoidal_poisson_generator.py`** — an inhomogeneous Poisson train whose rate
  oscillates, `λ(t) = max(0, dc + ac·sin(2πft + φ))`, across a bank of `N` channels. The
  population PSTH tracks `λ(t)` and the per-bin spike-count autocorrelation carries the
  modulation period; the `individual_spike_trains` flag selects the noise mode
  (independent channels vs one train broadcast to all, perfectly synchronous).
- **`sinusoidal_gamma_generator.py`** — the same rate profile, but spikes are drawn from
  a **gamma renewal process of order `m`**. The headline is the gamma-regularization law:
  the ISI coefficient of variation `CV → 1/√m`, so `m = 1` recovers Poisson (`CV → 1`) and
  large `m` gives an increasingly clock-like train, while the mean rate profile is
  unchanged.
- **`pulsepacket.py`** — a `pulsepacket_generator` emits Gaussian-jittered synchronous
  spike packets (`activity` spikes per center, times `~ N(pulse, sdev²)`). The packet
  width *is* the jitter `sdev`; replaying the packet through `iaf_psc_alpha` neurons
  recovers the neuron-averaged membrane excursion, which matches the analytical
  Gaussian⊛PSP convolution of Diesmann.

The two sinusoidal generators are `for_loop`-traceable, so each is driven **directly** by
`brainstate.transform.for_loop` over a single `in_size=N` instance — the same loop
primitive `Simulator.simulate` uses internally — which both exercises the
`individual_spike_trains` flag and lowers the whole rollout into one compiled program.
`pulsepacket_generator` is instead **host-side** (NumPy `default_rng` + per-train `deque`
queues, so its `update()` is not JAX-traceable); it is driven by an explicit host loop,
and its precomputed packet is then replayed through the `Simulator` as a `SpikeTime`
population (`one_to_one`) for the membrane drive.

Parity is **distributional**: NEST's per-thread RNG and the JAX/NumPy streams diverge
sample-by-sample, so each test compares a seed-aggregated statistic within a documented
band — the spike-count autocorrelation (Poisson), the seed-mean ISI-CV (gamma), and the
packet width plus per-step count profile (pulsepacket) — alongside the qualitative law
each demo exists to show. The pulsepacket membrane excursion is checked NEST-free against
the analytical solution.

> NEST's §3.8 **astrocyte** demos (`astrocytes/`): the neuron↔astrocyte SIC-loop machinery
> **landed in cluster 15d** (the last bucket-3 *model* cluster; only the Siegert
> `diffusion_connection` remains queued → 15c). `astrocyte_lr_1994` emits its
> slow-inward current through the seam-(H) continuous-emission path, and a one-way
> `sic_connection` deposits `weight·SIC` into `aeif_cond_alpha_astro`'s `'I_SIC'` current
> channel via an `as_current` `EventProjection`; the whole bidirectional loop lowers under
> `Simulator.simulate` and matches live NEST near-exactly
> (`_validation/astrocyte_sic_test.py`).
>
> **Cluster 17b** ports the two substrate-ready single-cell demos:
> [`astrocyte_single.py`](astrocyte_single.py) (one astrocyte, Poisson-driven, IP3/Ca + a
> downstream SIC) and [`astrocyte_interaction.py`](astrocyte_interaction.py) (the tripartite
> loop: pre-neuron → {post-neuron EPSP, astrocyte IP3} → SIC back to post), each with a
> live-NEST parity test (`V`/IP3/Ca/`I_SIC`). Porting `astrocyte_interaction`'s default
> Poisson drive surfaced — and fixed — a latent gap: `aeif_cond_alpha_astro` could not
> receive excitatory/inhibitory **spike** input into its synaptic conductance (it self-pulled
> a delta channel the Simulator never populated, so a presynaptic spike left `V_m` pinned at
> `E_L`). It now exposes the `n_receptors=2`/`w_by_rec` multi-receptor bridge
> (`receptor_type=1`→`g_ex`, `=2`→`g_in`, positive nS = NEST's weight-sign routing), with its
> own conductance parity test (`_validation/aeif_cond_alpha_astro_test.py`); the same fix is
> tracked for the sibling conductance neurons in `neurons-gap.md`.
>
> **Cluster 24** closes §3.8: it adds the `Simulator`-level
> [`tripartite_connect`](../../brainpy_state/_network/_simulator.py) +
> `third_factor_bernoulli_with_pool` astrocyte-pool rule. One realized primary
> `pre→post` sample is shared across all three arms — primary (`pre→post`),
> `third_in` (`pre→astro`, delta IP3) and `third_out` (`astro→post`, the
> `sic_connection`) — reusing the merged static + SIC paths with **no new deposit
> primitive**, validated against live NEST by a micro-parity GATE
> (`_validation/tripartite_connect_test.py`: block bit-identical / random
> distributional). The three pool-rule demos now ship as **real ports**:
> [`astrocyte_small_network.py`](astrocyte_small_network.py) (deterministic per-sample
> parity on IP3/Ca/`I_SIC` + driver `V_pre`) and the two
> [`astrocyte_brunel_*`](astrocyte_brunel_bernoulli.py) variants
> (`pairwise_bernoulli` / `fixed_indegree` primary rule; **connectivity-distributional**
> parity on the `pre→post`/`pre→astro`/`astro→post` edge counts — a balanced AI rate
> needs near-full scale, so rate parity is not asserted at the dense-friendly test
> scale). Both arms use **static** (not `tsodyks`) synapses — connectivity-neutral, the
> documented goal-24 divergence. **§3.8 complete.**

## Spatial-network demos (§3.9)

NEST's `spatial/` tutorials place neurons at **coordinates** and wire them with
distance-dependent rules (`nest.spatial.*` / `nest.spatial_distributions.*`). brainpy.state
gains a sibling [`brainpy.state.spatial`](../../brainpy_state/_nest_spatial) namespace mirroring
that surface: position layers (`grid`, `free` in 2-D/3-D), the `distance` sentinel and the
`gaussian` kernel, masks (`circular`/`spherical`/`box`), the `spatial_pairwise_bernoulli`
connection rule, and query helpers (`center_element`, `Distance`, `target_nodes`,
`target_positions`). Coordinates carry `brainunit` length units.

The seam is deliberately small. `Simulator.create(model, positions=spatial.grid/free(...))`
attaches coordinates to a population (a `grid` / concrete `free` layer derives the size from
its coordinates; a distribution-backed `free` layer draws `size` of them) and stores them under
the population; `Simulator.get_position` (NEST `GetPosition`) reads them back.
`spatial_pairwise_bernoulli(p=..., mask=...)` is an **ordinary `ConnRule`** — it rides the
existing `Simulator.connect(..., rule=...)` with no signature change. At connect time the
Simulator binds the connect's sliced pre/post coordinates onto a pure rule clone, so every
downstream path (static, plastic, even diffusion/gap/sic) samples one coordinate-bound rule;
the whole `(n_pre, n_post)` distance + Bernoulli draw is vectorized (no Python pair loop).

Four faithful ports plus one documented placeholder (run any directly, e.g.
`python examples/nest/spatial_gaussex.py`):

- **`spatial_grid_iaf.py`** (NEST `grid_iaf`) — a 4×3 `iaf_psc_alpha` grid. The layout is
  *exactly* NEST's (column-slow/row-fast, `x` left→right, `y` top→bottom), asserted
  element-for-element against live NEST `GetPosition`.
- **`spatial_gaussex.py`** (NEST `gaussex`) — two 30×30 grids connected with a Gaussian
  distance kernel `p(d)=exp(-d²/2σ²)` clipped to a circular mask; the central neuron's
  realized footprint is read back via `target_positions`.
- **`spatial_3d_gauss.py`** (NEST `test_3d_gauss`) — 1000 neurons at uniform-random 3-D
  positions, a Gaussian kernel with **no autapses** clipped to a cubic `box` mask, and a
  target-distance histogram of the central footprint.
- **`spatial_csa.py`** (NEST `csa_spatial_example`) — the CSA Gaussian connectivity
  (`csa.random * (csa.gaussian(σ, cutoff) * d)`) expressed **natively** as
  `spatial_pairwise_bernoulli(p=gaussian(distance, std=σ), mask=circular(cutoff))` — no
  `libneurosim`.
- **`csa_example.py`** — a **documented placeholder**: the CSA/`conngen` *mechanism*
  (libneurosim) is intentionally not ported, but the connectivity it describes
  (`csa.random(0.1)`) is shown to map to the native `pairwise_bernoulli(0.1)`.

Parity is **tiered**. Grid coordinates and the centre element are deterministic, so they match
live NEST exactly. The probabilistic samples diverge under independent PRNGs, so the kernel
demos are validated **distributionally** — the empirical connection fraction binned by distance
must track the analytic Gaussian *and* live NEST's empirical curve, plus a seed-mean edge count
(`_validation/spatial_{grid,gaussian_kernel,3d}_test.py`). Each example also runs standalone in
CI without NEST.

## Pedagogical demos (§3.10)

NEST's §3.10 "pedagogical / advanced" group — single-cell AdEx figures, multi-compartment
dendrites, and a reinforcement-learning game — each a faithful port with a live-NEST parity
test plus a NEST-free law/behaviour companion. Run any directly, e.g.
`python examples/nest/two_comps.py`.

### AdEx figures (Brette & Gerstner 2005)

- **`brette_gerstner_fig_2c.py`** — **spike-frequency adaptation** in an adaptive exponential
  integrate-and-fire neuron (`aeif_cond_alpha`). Two DC pulses (500 pA, then 800 pA) drive a
  single cell whose inter-spike intervals lengthen as the adaptation current `w` accumulates —
  the SFA hallmark of Figure 2C.
- **`brette_gerstner_fig_3d.py`** — **post-inhibitory rebound** (`aeif_cond_exp`; `a = 80 nS`,
  `b = 80.5 pA`, `tau_w = 720 ms`). An 800 pA *inhibitory* step hyperpolarises the membrane; on
  release the adaptation current drives a rebound burst back through threshold (Figure 3D).

### Compartmental models (`cm_default`)

Both build a soma+dendrite tree on NEST's multi-compartment `cm_default`, exercising the
committed Simulator seam for per-compartment state, receptors, and device routing.

- **`two_comps.py`** — **active vs passive dendrites.** The same soma+dendrite is built twice —
  once with a passive dendrite, once with active Na/K channels — and driven identically through
  `AMPA_NMDA` receptors on soma and dendrite; the active dendrite amplifies and sharpens the
  dendritic response relative to the passive cable.
- **`receptors_and_current.py`** — **multiple receptor types + steady current.** A passive
  soma+2-dendrite tree carries `GABA` on the soma, `AMPA` on dendrite 1, and `AMPA_NMDA` on
  dendrite 2, each driven through its receptor index, plus a 1 pA DC injection into dendrite 1 —
  showing the distinct AMPA / NMDA / GABA signatures attenuating electrotonically across
  compartments.

### Pong — reinforcement learning (Wunderlich et al. 2019)

A pure-Python game plus two spiking learners that map the ball's y-cell to a paddle move,
trained turn-by-turn on the **persistent-rollout substrate added for this group**
(`Simulator.cont` + `host_drive`, below).

- **`pong.py`** — the pure-`numpy` `GameOfPong` (ball, paddles, 32×20 grid); no spiking
  machinery, so it is stepped from the host loop between turns.
- **`pong_networks.py`** — the two learners over a 20-cell input→motor map. **`PongNetRSTDP`**
  rewrites static-synapse weights on the host after each 200 ms turn by the reward-modulated
  STDP rule (`learning_rate · calculate_stdp · reward`, a per-edge spike-timing correlation);
  **`PongNetDopa`** instead lets `stdp_dopamine_synapse` edges evolve online *inside* the
  rollout, driven by an actor–critic circuit (striatum → ventral pallidum → dopaminergic
  neurons → `volume_transmitter`) whose reward current the host injects each turn.
- **`pong_run.py`** — the `AIPong` host loop (a faithful port of NEST's `run_simulations.py`)
  pits two learners head-to-head and emits a learning-curve comparison. Because each network
  owns its own `Simulator` + `volume_transmitter`, any pairing trains — including two
  dopaminergic players, which NEST's single global transmitter forbids. Run
  `python examples/nest/pong_run.py --quick`.

**Substrate added for pong (reusable).** Driving a model in host-interleaved chunks needs two
new primitives, both promoted into the package:

- **`Simulator.cont(duration)`** — a non-re-initialising sibling of `simulate()`: state persists
  across calls (biological time accumulates), so a host loop can read recordings, rewrite a
  `host_drive` schedule, or overwrite static weights between chunks while the compiled per-chunk
  `for_loop` is reused (no recompile). `reset_rollout()` starts a fresh rollout at `t = 0`.
- **`host_spike_drive` / `host_current_drive`** — State-backed input devices holding a
  `(window, n)` per-step schedule; `set_schedule()` rewrites the State *contents* (fixed shape)
  between `cont()` chunks, so changing which input fires this turn — or how large the reward
  current is — never retraces the rollout.

> NEST's §3.10 **`sudoku/`** demo (a noise-driven WTA constraint-satisfaction network) was
> investigated and found intractable to bring to solve-rate parity on the current substrate; it
> is carried as a documented TODO (`CONTEXT.md`, `examples-gap.md` §3.10).

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

### Plasticity demos (§3.3)

Deterministic rules are compared per-sample against live NEST; the stochastic
quantal rule is compared as a seed-mean statistic (category D, 5 %). The two
Clopath demos read the stored weight, which carries a documented online-vs-NEST
band (instantaneous post-state read vs NEST's deferred-history `weight_recorder`):
backward/LTD edges are near-exact, forward/LTP within 5 % (growing with pairing
frequency). Weights are bare mV mantissas (`aeif_psc_delta_clopath` is a delta
neuron, init `0.5`).

| Port | metric | brainpy vs NEST |
|---|---|---|
| `evaluate_tsodyks2_synapse` (dep / fac) | post `V_m` PSC-amplitude train, max\|Δ\| | 9.4e-16 mV (`CAT_B`, 2-step align) |
| `evaluate_quantal_stp_synapse` (8 seeds) | seed-mean `V_m` (dep / fac) | 2.08 vs 2.11 (1.8 %) ; 2.57 vs 2.49 (2.9 %) — within `CAT_D` |
| `clopath_synapse_spike_pairing` (10 trains) | stored weight per train, in clopath band | LTP rel ≤ 3.3 % ; LTD \|Δ\| ≤ 0.0022 mV |
| `clopath_synapse_small_network` (6 edges) | per-edge final weight, in clopath band | LTP rel ≤ 2.0 % ; LTD \|Δ\| ≤ 0.0007 mV |

`tsodyks2` is an exact analytic-propagator rule, so once the parrot relay delay is
matched to the generator holder lag (`RELAY_D = 0.1`) the two PSC trains agree to
machine precision (the 2-step search only absorbs the multimeter recorder step).
The quantal rule draws a single `jax.random.binomial` where NEST draws one
Bernoulli per site — distributionally identical but on independent PRNG streams —
so parity is the seed-mean `V_m` (the example uses `n_sites = 100`, `n·w` fixed,
to tighten the estimate). Both Clopath demos are the same spike-pairing physics:
the small network is just that protocol replicated across every directed edge, so
its per-edge band is tighter (≤ 2 %) than the up-to-50 Hz pairing sweep (≤ 3.3 %).

### Recording & device demos (§3.4)

Deterministic recordings are compared per-sample against live NEST; PRNG-driven
detectors are compared as a seed-mean statistic (category D, 5 %); the precise
spiking contrast is compared as an onset-aligned spike sequence (category E);
connection introspection is compared as exact realized-edge counts (and per-edge
`set` round-trips) plus seed-mean weights (category D).

| Port | metric | brainpy vs NEST |
|---|---|---|
| `multimeter_file` | `V_m`/`I_syn_ex`/`I_syn_in` trace, max\|Δ\| | machine precision (`CAT_B_GEN`, 2-step align) |
| `recording_demo` (4 seeds) | firing rate | refractory-saturated → identical |
| `cross_check_mip_corrdet` (5 seeds) | normalized cross-correlogram, max\|Δ\| | within `CAT_D` (5e-2) |
| `correlospinmatrix_..._two_neuron` (5 seeds) | mean activities ; covariance | means \|Δ\| ≤ 0.013 ; cov max\|Δ\| ≈ 0.014 |
| `precise_spiking` (`dt∈{0.1,0.5,1.0}`) | onset-aligned spike times ; count | ≤ 1 step ; 10 = 10 |
| `plot_weight_matrices` (5 seeds) | per-edge `Normal` weight mean ; `fixed_indegree` structure | `w_ex`/`w_in` mean within `CAT_D` ; in-degree exact |
| `synapsecollection` (5 seeds) | `one_to_one`/`all_to_all` counts + per-edge `set` ; `Uniform` weight mean ; `stdp` model-filter count | counts/`set` exact ; weight mean within `CAT_D` |

`multimeter_file`'s three recordables are exact analytic-propagator curves, so
they match to machine precision once the constant two-step generator-delivery
offset is aligned (`CAT_B_GEN`). `recording_demo`'s 1 MHz drive pins the neuron to
its refractory-saturated rate, so the rate is identical across seeds despite the
PRNG-divergent Poisson stream. The two correlation detectors are imperative host
devices that mirror NEST bit-for-bit; the parity that PRNG-diverges is the
*input* train, so `cross_check_mip_corrdet` and `correlospinmatrix` are compared
as seed-mean correlograms / activities (`CAT_D`). `precise_spiking` is fully
deterministic but the DC drive lands after a constant connection-delay onset
(NEST's default 1 ms vs the Simulator one-step convention), so parity is asserted
on the **onset-aligned** spike sequence — exact spike *count* and spike times
*relative to the first spike* (the physically meaningful firing period and its
resolution dependence). The two introspection demos split into an *exact* part and
a *distributional* part: deterministic-count rules (`one_to_one`, `all_to_all`,
`fixed_indegree`, `fixed_total_number`) realize the same edge counts and per-target
in-degrees as NEST and a per-edge `set` round-trips identically on both sides, so
those are asserted exactly; the `Normal` / `Uniform` weight *draws* agree only as a
seed-mean (`CAT_D`), and the random topology of `pairwise_bernoulli` /
`fixed_total_number` differs between the PRNGs, so only their counts (not the
realized pairs) are compared.

### Single-neuron model demos (§3.5)

Deterministic ports are compared per-sample against live NEST with the standard
one-step recorder alignment (`t=0` sample dropped candidate-side, `align_steps=1`
for the remaining shift); spiking GLIF levels additionally assert exact spike
**counts**. Exact-propagator models (`glif_psc*`) match to machine precision;
RKF45 models (`hh`, `aeif`, `glif_cond`) match their smooth sub-threshold traces
to category A.

| Port | metric | brainpy vs NEST |
|---|---|---|
| `hh_psc_alpha` | subthreshold `V_m` + `Act_m`/`Inact_h`/`Act_n` ; F–I spike count | `CAT_A` (~1e-3 mV) ; counts match |
| `hh_phaseplane` | `n`-nullcline vs analytic `n_inf(V)` ; trajectory relaxation | within one grid step (NEST-free) |
| `aeif_cond_beta_multisynapse` | `V_m` ; `g_1..g_4`, max\|Δ\| | ~1e-6 mV ; machine precision |
| `gif_cond_exp_multisynapse` | subthreshold `V_m`, max\|Δ\| | machine precision |
| `glif_cond_neuron` (5 levels) | `g_1`/`g_2` full trace ; subthreshold `V_m`/`threshold` ; spike count | machine precision ; ~1e-13 mV ; exact |
| `glif_psc_neuron` (5 levels) | `I_syn`/`I` full trace ; subthreshold `V_m` ; spike count | ~2e-15 pA ; ~0.03 mV (`CAT_B_ALIGNED`) ; exact |
| `glif_psc_double_alpha_neuron` (3 cfgs) | `V_m` + `I_syn` full trace (sub-threshold) | ~1e-13 mV / ~1e-15 pA |
| `iaf_tum_2000` (depression, facilitation) | post `V_m` full trace (sub-threshold) | machine precision after a constant ~8-step align |
| `izhikevich` (RS/IB/CH/FS) | `V_m`+`U_m` full trace ; spike count | `CAT_A` (~1e-3 mV), zero shift ; exact |
| `mat_psc_exp` (mat2, amat2) | `V_m` + composite `V_th` full trace ; spike count | float-noise floor, zero shift ; exact |
| `mc_neuron` (`iaf_cond_alpha_mc`, 9 receptors) | per-compartment `V_m.{s,p,d}` ; `g_ex/g_in.*` ; spike count | ~0.05–0.08 mV (RKF45) ; machine precision ; exact |

The GLIF per-port conductances (`g_k`) and synaptic currents (`I_syn`, `I`) are
linear filters of the fixed external spike trains, so they match NEST over the
whole trace independent of the neuron's own spike jitter; the membrane potential
is compared in the sub-threshold window (before the two integrators diverge on
spike timing) and the spiking behaviour by exact count. `glif_psc`'s `V_m` carries
a ~0.03 mV residual — `V` is computed from the *pre*-propagation PSC while the
recorded `I_syn` is the *post*-propagation `y2`, so the two sit one step apart
relative to NEST (which reports both from the same `y2`) — hence `CAT_B_ALIGNED`
rather than the machine-precision band the linear currents and the spike-free
`double_alpha` traces enjoy. The 150 kHz `glif_psc` Poisson window is excluded
from per-sample parity (independent PRNG streams) but tracks NEST in aggregate
once the parrot relays spike multiplicity (≈59/88/27/26/26 vs 59/89/27/27/26
spikes across the five levels).

Four §3.5 demos are **carve-outs** rather than per-sample NEST trace parity: their
drive is a PRNG stream (or the comparison is against analytic theory), so the ground
truth is the closed form or the seed-averaged statistics, never a per-sample match.

| Port | comparison | result |
|---|---|---|
| `CampbellSiegert` | sim vs analytic Campbell μ/σ², Siegert rate | μ <0.01 mV ; σ² ~2–3 % ; rate ~35 % (low-count Siegert approximation) |
| `BrodyHopfield` | seed-mean vector strength / rate / phase-histogram vs NEST (5 seeds) | R ~2 % ; rate ~0.3 % ; phase-hist max\|Δ\| ~0.006 |
| `gif_population` | seed-mean rate / binned-rate autocorrelation vs NEST (`CAT_D`) | rate ~1.4 % ; autocorr max\|Δ\| ~0.024 |
| `gif_pop_psc_exp` | mesoscopic vs microscopic `A_N` ; meso driver vs NEST (uncoupled) | window-mean rate ~11 % (mean 0.3 %) ; step jump ×3.4 / ×2.7 |

`BrodyHopfield`, `gif_population` and `gif_pop_psc_exp` are stochastic spiking
populations whose NEST and JAX PRNG streams diverge, so they are compared
distributionally (seed-aggregated, category D); `CampbellSiegert` is checked against
its own closed-form theory. `gif_pop_psc_exp`'s mesoscopic half is the host-side
`gif_pop_psc_exp` population-rate model (NumPy + `RandomState`, not JAX-traceable),
driven by a host-side Python loop — the documented exception to the
`brainstate.transform` lowering rule for untraceable models.

### Network demos (§3.6)

These networks are chaotic, balanced, or metastable, so they are compared
**distributionally**: a seed-mean (or seed-median) population observable within a
documented band, plus the qualitative law the demo exists to demonstrate — never a
per-neuron match (the RNG streams diverge from NEST's). Bands are wider than the
single-neuron 5 % because a *balanced* rate sits on a near-cancellation of large
E/I currents and is hypersensitive to sub-percent current scatter.

| Port | metric | brainpy vs NEST |
|---|---|---|
| `repeated_stimulation` | per-trial active-window spike count | within `CAT_D` 5 % ; zero-rate → silent |
| `artificial_synchrony` | synchrony Σ vs coupling | uncoupled baseline exact ; Σ↑ monotone both sims ; sensitive strengths in ~10 % band |
| `sensitivity_to_perturbation` | AI-state rate ; perturbation divergence | 14.95 vs 15.17 Hz (1.45 %) ; > 0.9 of net decorrelates after `t_stim`, 0 before (both sims) |
| `ei_clustered_network` | rep=1 median E/I rate ; median ISI-CV ; rep=6 signature | rate ≤ 12 % (meas. ~1–3 %) ; CV ≤ 8 % (meas. < 4 %) ; `std6>3·std1` & `CV6>CV1` both |
| `brette_et_al_2007` COBA | steady-state E / I rate (3-seed mean) | 13.77 / 14.03 vs 15.12 / 14.44 Hz (E 8.9 % / I 2.8 %, band 15 %) |
| `brette_et_al_2007` CUBA | steady-state E / I rate (3-seed mean) | 3.97 / 3.99 vs 4.03 / 4.01 Hz (E 1.5 % / I 0.5 %, band 12 %) |
| `iaf_bw_2001` (Wang neuron) | AMPA+GABA `V_m`/gating ; 2-neuron NMDA `s_NMDA`/`I_NMDA`/`V_m` | machine precision (direct align) |
| `iaf_bw_2001` recurrent NMDA | per-neuron `s_NMDA`/`V_m`, asymmetric 3-cell pool drive | machine precision ~5e-15 (pipeline-latency aligned) |
| `wang_decision_making` | WTA decision direction & contrast (3 seeds × ±/0 coherence) | distributional: ±coh→A/B both sims ; winner > 2.5× loser (< 4 Hz) ; unbiased at 0 |
| `lin_rate_ipn_network` | deterministic fixed point `(λI−W)⁻¹μ` ; per-neuron trajectory | FP = closed form **and** NEST to atol 1e-3 (`use_wfr=False`) ; trajectory matches NEST (`align_steps=12`) |
| `rate_neuron_dm` | deterministic WTA winner/loser ; noisy decision direction & contrast | winner 11.0 / loser 0 = NEST exactly ; +/-bias 5/5 & 0/5 both sims ; winner ≫ loser ; unbiased at dE=0 |

`brette_et_al_2007` compares the **second-half (steady-state) rate** over
`[simtime/2, simtime]` — the kick-independent observable; the full-window rate is
dominated by an ignition transient whose magnitude differs between simulators (a
kick-response detail, not the benchmark's self-sustained-rate point). The COBA
receptor seam is therefore validated at network scale, not just single-cell.
`ei_clustered_network` uses the **median** over seeds because a rare seed falls into
a globally synchronized state (CV≈0) that NEST does not share at that seed; the
median reports the typical AI-state realization while the mean would chase that
outlier. `sensitivity_to_perturbation`'s chaotic divergence is asserted only
qualitatively (the perturbed neuron and connectivity PRNG-differ, so the per-seed
divergence trajectory is not matched — only that *both* simulators decorrelate the
bulk of the network after `t_stim` and neither before). `wang_decision_making` is a
winner-take-all attractor whose positive NMDA feedback amplifies the per-neuron
integrator and PRNG differences, so the winner's *absolute* rate differs between
simulators (brainpy A~12 vs NEST A~7 Hz on +bias); its parity is therefore
**distributional/behavioural** — strong ±coherence selects A/B on both sims, the
winner's late rate exceeds the loser's by > 2.5× (loser suppressed < 4 Hz), and the
choice is unbiased at zero coherence. The recurrent-NMDA *coupling* underneath it
matches NEST to machine precision (`iaf_bw_2001_recurrent_nmda_parity_test.py`), so
the divergence is genuine attractor amplification, not a wiring or coupling error.

The two **rate-based** demos break the distributional-by-construction pattern of the spiking
demos above. `lin_rate_ipn_network` is linear, so it has an analytic fixed point: a small
deterministic (`sigma = 0`) instrument net is matched to the closed form `(λI − W)⁻¹μ` **and**
to live NEST (`use_wfr=False`) at `atol 1e-3`, and the per-neuron trajectory is matched once
`align_steps` absorbs the uniform pipeline+delay offset (the demo's full random net is covered
by a smoke run, since random connectivity PRNG-diverges). `rate_neuron_dm` has a tight
**deterministic** anchor too — at `sigma = 0` the winner relaxes to `10·μ_win` (11.0) and the
loser rectifies to exactly 0 on *both* simulators — and is otherwise compared
distributionally over seeds (a strong bias drives D1 on 5/5 seeds and against it on 0/5 on
both sims; the winner ≫ loser; the decision is unbiased at `dE = 0`), because the WTA
attractor amplifies PRNG divergence just as Wang's does. `rate_neuron_dm` is also the goal-17
arbiter that confirmed `rectify_output` behaves correctly inside a *recurrent* rate loop.

### Spatial-network demos (§3.9)

Grid coordinates are deterministic and match live NEST **exactly**; the probabilistic kernel
samples PRNG-diverge, so they are compared **distributionally** — the empirical connection
fraction binned by distance vs both the analytic Gaussian and live NEST's empirical curve,
plus a seed-mean edge count. Figures are against NEST 3.9.0.

| Port | metric | brainpy vs NEST |
|---|---|---|
| `spatial_grid_iaf` | grid coordinates (4×3, 3×3×3) ; centre element | exact, element-for-element (`GetPosition` / `FindCenterElement`) |
| `spatial_gaussex` | empirical `p(d)` per distance bin ; edge count | max\|bp−NEST\| ≈ 0.016 ; 21083 vs 20980 edges (0.5 %) |
| `spatial_3d_gauss` | 3-D empirical `p(d)` per bin ; box cutoff ; autapses ; edge count | max\|bp−NEST\| ≈ 0.008 ; hard box ; zero autapses ; 129437 vs 130758 edges (≈1 %) |
| `spatial_csa` (native CSA) | Gaussian-kernel footprint | same kernel family as `gaussex` (smoke) |

The grid layout was pinned empirically against live NEST (`x = c−L/2 + (col+0.5)·L/n`,
`y = c+L/2 − (row+0.5)·L/n`, column slow / row fast). Both Gaussian demos confirm the
substrate's `p(d)` tracks the analytic law and NEST's realized curve bin-by-bin; the 3-D demo
additionally checks the `box` mask is a hard per-axis cutoff and `allow_autapses=False` removes
every self-edge. Each demo's NEST-free companion (the structural class) runs in CI.

### Pedagogical demos (§3.10)

The AdEx figures and compartmental demos are **deterministic** ports, compared per-sample
against live NEST; pong is a **reinforcement-learning** demo whose game trajectory
PRNG-diverges, so its parity is component-deterministic plus behavioural (as for
`wang_decision_making` above), never per-sample.

| Port | metric | brainpy vs NEST |
|---|---|---|
| `brette_gerstner_fig_2c` | AdEx sub-threshold `V_m` (500 pA, spike-free), `CAT_A` ; spike pattern `CAT_E` | < 1e-3 mV ; \|ΔN\| ≤ 2, first spike ≤ 1 step |
| `brette_gerstner_fig_3d` | hyperpolarised-plateau `V_m`, `CAT_A` ; rebound burst `CAT_E` | < 0.02 mV ; \|ΔN\| ≤ 2, first spike ≤ 1 step |
| `two_comps` | soma/dend `v_comp` ; Na/K `m,h,n` ; AMPA/NMDA `g_r,g_d` | soma AP ~0.03 mV (active-tip residual ~0.56 mV) ; gating 6e-3 ; conductance ~1e-15 nS |
| `receptors_and_current` | DC-only `V_m` (`CAT_C`) ; synaptic-drive `V_m` (`CAT_C`) | ~1e-13 mV ; ~1e-2 mV |
| `pong` `calculate_stdp` | R-STDP correlation vs NEST's own `PongNetRSTDP.calculate_stdp` | bit-for-bit (pinned, e.g. 67.6377405226) |
| `pong` dopamine pathway | reward current → dopaminergic firing → input→motor weight change | potentiation (reward) vs depression (no reward), sign-correct ; weights bounded `[Wmin, Wmax]` |

The compartmental residuals are the integrator-and-tap precision floor: the `two_comps`
active-dendrite *tip* carries a ~0.56 mV sub-step-peak sensitivity (the Na/K spike is sharper
than `dt` and `cm_default` has no reset to re-anchor it), while the soma AP and the
deterministic double-exponential conductances match to ~float noise. For pong, the host
`calculate_stdp` correlation reproduces NEST's method bit-for-bit, the dopaminergic
reward→potentiation pathway is sign-checked against a zero-reward control, and a bounded
fixed-seed head-to-head run is asserted only to be well-formed (weights inside `[Wmin, Wmax]`,
paddles tracking within the field, the reward baseline rising off zero). The `Simulator.cont`
persistent rollout and `host_drive` clamped input underneath the turn loop are themselves
oracle-tested against one long `simulate()` and against chunked live-NEST accumulation.
