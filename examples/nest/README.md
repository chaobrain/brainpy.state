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

## Plasticity demos (§3.3)

Four of NEST's plasticity tutorials, each a faithful port driven by a live-NEST
parity test (`brainpy_state/_nest/_validation/<name>_test.py`). Run any directly,
e.g. `python examples/nest/evaluate_tsodyks2_synapse.py`.

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

The fifth §3.3 demo is **blocked** and ships as a skipped placeholder that raises
`NotImplementedError` with the gap reason:

- **`urbanczik_synapse_example.py`** — the Urbanczik-Senn dendritic rule trains
  plastic inputs onto the *dendritic* compartment of a two-compartment
  `pp_cond_exp_mc_urbanczik` neuron. It needs a dendritic post-compartment reader
  on a plastic projection (the rule reads a named compartment voltage + prediction
  error, but `VoltageCoupledPlasticProj` exposes only the somatic `V`) and a
  validated multi-compartment point-process post (`synapses-plasticity-gap.md` §3,
  `neurons-gap.md` §3). It will be ported once those land.

These ports add two reusable seams on top of the §3.2 / §3.4 vocabulary:

- **F — plastic projections + weight recording.** `connect(..., synapse=<plastic
  rule>)` dispatches to the plastic-projection primitives — `EventPlasticProj`
  (event-driven; the two STP rules) and `VoltageCoupledPlasticProj` (the
  post-state reader; Clopath reads the `aeif_psc_delta_clopath` analog voltages
  every step). `record_weight(proj)` taps the per-edge weight, read post-run as
  `res.weight_trace(proj)` → `(T, E)` in CSR (sorted-by-pre) order. The two Clopath
  demos record the weight directly; the STP demos read the plasticity through the
  post `V_m` (the PSC-amplitude train, seam **A**).
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
  file; brainpy.state has neither a file backend (`devices-gap.md` P2) nor a
  spike→conductance routing seam (a documented follow-up), so this is the
  **in-memory, current-based** equivalent — same demo shape, traces read with
  `res.trace(mm, name)`.
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

Two further §3.4 demos are **blocked** and ship as skipped placeholders that
raise `NotImplementedError` with the gap reason:

- **`plot_weight_matrices.py`** and **`synapsecollection.py`** both need post-hoc
  connection introspection — `GetConnections` / `SynapseCollection` to enumerate
  realized synapses and read per-edge weights (`network-api-gap.md` §3.1, §3.8) —
  which the explicit `Simulator` does not expose. They will be ported once the
  planned `nest_compat` facade lands.

These ports add one reusable seam on top of the §3.2 vocabulary:

- **E — eager imperative devices.** `mip_generator`, `correlation_detector`, and
  `correlospinmatrix_detector` are NumPy-RNG / Python-loop host devices that cannot
  enter a JAX `for_loop`. The pattern is to obtain the spike data first (a device
  `.simulate(n_steps)` multiplicity matrix, or a State-tapped binary train) and
  then drive the detector **post-hoc**, feeding only event-carrying steps and
  stamping each at `step + 1` (NEST's one-step delivery latency, which cancels in
  the lag difference). Nothing imperative runs inside the `for_loop`.

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

Five of NEST's spiking-network tutorials (NEST §3.6), each a faithful port driven
by a live-NEST **distributional** parity test — population rate, synchrony, or
perturbation divergence within a documented band — plus a no-NEST companion that
runs in CI. Run any directly, e.g. `python examples/nest/brette_et_al_2007.py`.

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

A sixth, **`wang_decision_making.py`**, is a **documented deferred placeholder**.
Its `iaf_bw_2001` neuron is fully validated against live NEST (single-cell AMPA+GABA
and the two-neuron NMDA presynaptic-offset coupling match to machine precision), and
the script ships a runnable demonstration of that validated recurrent-NMDA building
block — but the full competing-populations decision network awaits a Simulator seam:
recurrent NMDA deposits `weight · sender_spike_offset` (a *presynaptic*-state-gated
synapse), which the generic `weight · spike` event projection cannot yet express.

These ports add a conductance-LIF receptor seam and a Bernoulli connection rule on
top of the Brunel-family population vocabulary:

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

Parity is **distributional by construction**: these networks are chaotic, balanced,
or metastable, so their PRNG streams diverge from NEST's and a per-neuron match is
meaningless. Each test compares a seed-**mean** (or, for `ei_clustered`, a seed-
**median** robust to a rare globally-synchronized seed) of a population observable
within a documented band, and asserts the qualitative law the demo exists to show.

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
spiking contrast is compared as an onset-aligned spike sequence (category E).

| Port | metric | brainpy vs NEST |
|---|---|---|
| `multimeter_file` | `V_m`/`I_syn_ex`/`I_syn_in` trace, max\|Δ\| | machine precision (`CAT_B_GEN`, 2-step align) |
| `recording_demo` (4 seeds) | firing rate | refractory-saturated → identical |
| `cross_check_mip_corrdet` (5 seeds) | normalized cross-correlogram, max\|Δ\| | within `CAT_D` (5e-2) |
| `correlospinmatrix_..._two_neuron` (5 seeds) | mean activities ; covariance | means \|Δ\| ≤ 0.013 ; cov max\|Δ\| ≈ 0.014 |
| `precise_spiking` (`dt∈{0.1,0.5,1.0}`) | onset-aligned spike times ; count | ≤ 1 step ; 10 = 10 |

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
resolution dependence).

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
bulk of the network after `t_stim` and neither before). `wang_decision_making` ships
no network parity test — its neuron is validated by `iaf_bw_2001_nest_parity_test.py`
and the network is deferred pending an offset-aware NMDA event projection (the demo
documents the exact seam gap).
