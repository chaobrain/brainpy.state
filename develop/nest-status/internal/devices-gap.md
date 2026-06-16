# Devices — NEST parity gap

## 1. Scope

Stimulation generators (ac, dc, step current, step rate, Poisson variants,
inhomogeneous Poisson, sinusoidal Poisson + gamma, gamma-superimposed,
PPD-superimposed, MIP, pulse-packet, noise, spike), spike utilities
(`spike_generator`, `spike_train_injector`, `spike_dilutor`), recorders
(multimeter, spike_recorder, weight_recorder), correlation detectors
(`correlation_detector`, `correlomatrix_detector`, `correlospinmatrix_detector`),
binary-state detector (`spin_detector`).

Upstream reference:
<https://nest-simulator.readthedocs.io/en/stable/models/index.html> (devices
sections of catalog snapshot §§3-6: 25 devices total, excluding MUSIC proxies).

Lead implementations actually read for this analysis:
- `brainpy_state/_nest/multimeter.py` (lines 44-155 confirm full NEST device-
  timing parameters: `record_from`, `interval`, `offset`, `start`, `stop`,
  `origin`; start-exclusive / stop-inclusive gate semantics match NEST
  `(origin+start, origin+stop]`).
- `brainpy_state/_nest/spike_recorder.py` (lines 51-135 confirm identical
  gating semantics).
- `brainpy_state/_nest/poisson_generator.py` (lines 46-103 confirm `rate`,
  `start`, `stop`, `origin` with NEST-compatible bounds).
- Family extrapolation to remaining 22 devices.

## 2. Parity summary

All 25 NEST devices (excluding MUSIC proxies, which are unsupported per spec §7)
are ported and have NEST-comparison tests at the test-file level. The
`docs/nest-status/index.rst:93-94` self-disclosure flags "recording-device
fidelity" as not yet fully matching NEST's device model — meaning the
*parameter surface* matches NEST but the *output-buffering / event-emission
semantics* may diverge. Validate this concretely.

| Bucket | Count | Notes |
|---|---:|---|
| implemented | 0 | (no device test has documented tolerance + event-count parity convention) |
| nest_validated | 2 | `weight_recorder` (cluster-09 send-view audit) and `volume_transmitter` (`_validation/volume_transmitter_parity_test.py`) have live-NEST parity tests |
| unvalidated | 0 | All ported devices have `import nest` in their test |
| partial | 0 known | Per family-level structural check |
| divergent | 23 | All other ported devices: have NEST tests, but output-buffering / event-emission semantics flagged as divergent in `nest-status/index.rst:93-94` |
| missing | 0 | All in-scope NEST devices ported |
| unsupported | 7 | MUSIC proxies (catalog §7) |
| **total NEST devices surveyed (excl. MUSIC)** | **25** | per snapshot §§3-6 |

## 3. Evidence-backed mapping table

### Stimulation generators

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `ac_generator` | divergent | `brainpy_state/_nest/ac_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ac_generator.html> | `ac_generator_test.py` (Y, see line 348 for `nest.Connect` usage) | parameter surface ok; output cadence verify |
| `dc_generator` | divergent | `brainpy_state/_nest/dc_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/dc_generator.html> | `dc_generator_test.py` (Y) | |
| `gamma_sup_generator` | divergent | `brainpy_state/_nest/gamma_sup_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/gamma_sup_generator.html> | `gamma_sup_generator_test.py` (Y) | superposition of gamma processes; PRNG distributional |
| `inhomogeneous_poisson_generator` | divergent | `brainpy_state/_nest/inhomogeneous_poisson_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/inhomogeneous_poisson_generator.html> | `inhomogeneous_poisson_generator_test.py` (Y) | piecewise-constant rate; verify rate-change boundary handling |
| `mip_generator` | divergent | `brainpy_state/_nest/mip_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/mip_generator.html> | `mip_generator_test.py` (Y) | multiple-interaction-process; PRNG sensitive |
| `noise_generator` | divergent | `brainpy_state/_nest/noise_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/noise_generator.html> | `noise_generator_test.py` (Y) | Gaussian white-noise current — PRNG distributional only |
| `poisson_generator` | divergent | `brainpy_state/_nest/poisson_generator.py:46-103` | <https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator.html> | `poisson_generator_test.py` (Y) | start-exclusive / stop-inclusive bounds match NEST |
| `poisson_generator_ps` | divergent | `brainpy_state/_nest/poisson_generator_ps.py` | <https://nest-simulator.readthedocs.io/en/stable/models/poisson_generator_ps.html> | `poisson_generator_ps_test.py` (Y) | precise spike timing + dead time |
| `ppd_sup_generator` | divergent | `brainpy_state/_nest/ppd_sup_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/ppd_sup_generator.html> | `ppd_sup_generator_test.py` (Y) | Poisson processes with dead time, superposed |
| `pulsepacket_generator` | divergent | `brainpy_state/_nest/pulsepacket_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/pulsepacket_generator.html> | `pulsepacket_generator_test.py` (Y) | Gaussian pulse packets |
| `sinusoidal_gamma_generator` | divergent | `brainpy_state/_nest/sinusoidal_gamma_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_gamma_generator.html> | `sinusoidal_gamma_generator_test.py` (Y) | |
| `sinusoidal_poisson_generator` | divergent | `brainpy_state/_nest/sinusoidal_poisson_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/sinusoidal_poisson_generator.html> | `sinusoidal_poisson_generator_test.py` (Y) | |
| `spike_generator` | divergent | `brainpy_state/_nest/spike_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/spike_generator.html> | `spike_generator_test.py` (Y) | emits spikes from prescribed times — verify off-grid rounding convention |
| `step_current_generator` | divergent | `brainpy_state/_nest/step_current_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/step_current_generator.html> | `step_current_generator_test.py` (Y) | piecewise DC |
| `step_rate_generator` | divergent | `brainpy_state/_nest/step_rate_generator.py` | <https://nest-simulator.readthedocs.io/en/stable/models/step_rate_generator.html> | `step_rate_generator_test.py` (Y) | piecewise rate (for rate models) |

### Recorders

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `multimeter` | divergent | `brainpy_state/_nest/multimeter.py:44-155` | <https://nest-simulator.readthedocs.io/en/stable/models/multimeter.html> | `multimeter_test.py` (Y) | full NEST timing surface (`interval`, `offset`, `start`, `stop`, `origin`, `record_from`); verify per-step pending-event aggregation matches NEST |
| `spike_recorder` | divergent | `brainpy_state/_nest/spike_recorder.py:51-135` | <https://nest-simulator.readthedocs.io/en/stable/models/spike_recorder.html> | `spike_recorder_test.py` (Y) | NEST-matching gate `(origin+start, origin+stop]`; verify event buffer/flush semantics under jit |
| `weight_recorder` | nest_validated | `brainpy_state/_nest/weight_recorder.py` (imperative shim) + `brainpy_state/_network/_weight_recorder_view.py` (send-view seam) | <https://nest-simulator.readthedocs.io/en/stable/models/weight_recorder.html> | `weight_recorder_test.py` (Y), `_validation/weight_recorder_audit_test.py` (Y) | per-rule wiring validated (cluster-09): the send-view seam reproduces NEST's send-event series (count + timing + value) for all 13 plastic rules with no imperative hook — `synapses-plasticity-gap.md` §6 |

### Detectors

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `correlation_detector` | divergent | `brainpy_state/_nest/correlation_detector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/correlation_detector.html> | `correlation_detector_test.py` (Y) | binned cross-correlation between two spike sources |
| `correlomatrix_detector` | divergent | `brainpy_state/_nest/correlomatrix_detector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/correlomatrix_detector.html> | `correlomatrix_detector_test.py` (Y) | covariance matrix from N inputs |
| `correlospinmatrix_detector` | divergent | `brainpy_state/_nest/correlospinmatrix_detector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/correlospinmatrix_detector.html> | `correlospinmatrix_detector_test.py` (Y) | covariance from binary states |
| `spin_detector` | divergent | `brainpy_state/_nest/spin_detector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/spin_detector.html> | `spin_detector_test.py` (Y) | binary-state recording |

### Other devices

| NEST model | Status | brainpy.state location | NEST upstream | Tests (import nest?) | Notes |
|---|---|---|---|---|---|
| `spike_dilutor` | divergent | `brainpy_state/_nest/spike_dilutor.py` | <https://nest-simulator.readthedocs.io/en/stable/models/spike_dilutor.html> | `spike_dilutor_test.py` (Y) | per-spike Bernoulli relay; PRNG distributional |
| `spike_train_injector` | divergent | `brainpy_state/_nest/spike_train_injector.py` | <https://nest-simulator.readthedocs.io/en/stable/models/spike_train_injector.html> | `spike_train_injector_test.py` (Y) | acts as a neuron + injects prescribed spike train (cross-link `neurons-gap.md`) |
| `volume_transmitter` | nest_validated | `brainpy_state/_nest/volume_transmitter.py` | <https://nest-simulator.readthedocs.io/en/stable/models/volume_transmitter.html> | `volume_transmitter_rule_test.py` (N, law/unit), `_validation/volume_transmitter_parity_test.py` (Y, live-NEST) | listed here for completeness; primary classification in `synapses-plasticity-gap.md` |

### MUSIC proxies (unsupported — see spec §7)

`music_cont_in_proxy`, `music_cont_out_proxy`, `music_event_in_proxy`,
`music_event_out_proxy`, `music_message_in_proxy`, `music_rate_in_proxy`,
`music_rate_out_proxy`. Catalogued in `nest-catalog-snapshot.md` §7. Not gaps.

## 4. Missing or incomplete functionality

- **Nothing entirely missing** among in-scope NEST devices. All 25 are ported,
  all have `import nest` comparison tests.
- **Output-buffering / event-emission semantics divergence** is signalled in
  `docs/nest-status/index.rst:93-94`. Concretely this means:
  - NEST recorders maintain per-device event buffers that grow during
    `Simulate()` and are flushed to the chosen recording-backend
    (`memory`, `ascii`, `sionlib`, …) on schedule.
  - `brainpy.state`'s JAX implementation stores events in `brainstate`
    state containers; the timing of "when an event is observable" relative
    to a simulation step differs.
  - Concrete verification needed: does `mm.events['V_m']` after a JIT-compiled
    simulation match the array NEST writes to memory after the same
    `Simulate()` call, element-for-element, including the stamp-step
    indexing?
- **Recording backends.** NEST supports `memory`, `ascii`, `sionlib` (and others
  via dynamic modules). `brainpy.state` recorders only support in-memory
  PyTree storage. No file-backend equivalents. Classify each backend's status
  in the roadmap below.

## 5. Semantic & numerical risks

- **Stamp-step rounding for `start`/`stop`/`origin`.** Multimeter and
  `spike_recorder` use the NEST convention `(origin+start, origin+stop]` in
  step units (`multimeter.py:96-99`). Verify against NEST that the floor /
  ceil convention matches at boundary steps.
- **Pre/post-step recording order.** NEST records *after* the step update;
  some recorders take pre-update values. Verify `multimeter.py` records the
  post-update value of `V_m`, `g_ex`, etc. at the sample step — this is
  what NEST does for most recordables but not all (a few `record_from`
  variables sample pre-update).
- **`spike_generator` off-grid spike times.** NEST supports off-grid spike
  times via `precise_times`. Verify the repo's spike_generator either matches
  the off-grid convention or documents the divergence.
- **`spike_dilutor` PRNG.** Per-spike Bernoulli — distributional only
  (spec §7).
- **MIP and supposition generators PRNG.** Same — distributional only.
- **Noise generator (`noise_generator`).** Gaussian white-noise current:
  per-step amplitude scaling by `1/sqrt(dt)` is NEST's convention; verify the
  repo implementation matches scaling so that the spectral density is
  invariant to dt.
- **`weight_recorder` hookup per plasticity rule.** ✅ **Resolved (cluster-09).**
  Rather than an imperative *emit-on-update* hook, weight recording reuses the
  analog State-tap: a thin send-view (`brainpy_state._network.weight_recorder_events`
  / `send_steps_from_pre`) masks the per-step weight trajectory to its send
  (pre-spike) steps, reproducing NEST's `weight_recorder` event series. Validated
  for all 13 plastic rules in `_validation/weight_recorder_audit_test.py`
  (cross-cuts `synapses-plasticity-gap.md` §6).
- **Correlation-detector binning windows and accumulation interval.** The
  correlator devices store covariance counts in a `Tau_max`-wide window
  bucketed by `delta_tau`. Verify the bucketing and the post-`Simulate()`
  normalization match NEST.
- **`spike_train_injector` is a neuron + device hybrid.** Cross-link
  `neurons-gap.md`. Classification is divergent in both docs.

## 6. Validation gaps

- All 25 devices have `import nest` in their tests, with per-device parity
  protocols now anchored in `_validation/` (tolerance conventions in
  `tolerance_conventions.py`).
- ✅ **Resolved.** Multimeter stamp-step parity is covered:
  `_validation/multimeter_file_test.py::test_all_recordables_match_nest`
  compares every recordable (`V_m`, `I_syn_ex`, `I_syn_in`) against live NEST
  at category B (`CAT_B_GEN`) via `compare_trace`.
- ✅ **Resolved.** `spike_recorder` exact-event parity is covered:
  `_validation/device_parity_test.py` asserts the exact event count and
  stamp-step time against NEST (`test_counts_and_stamp_step`) plus the
  distributional mean-count match (`test_mean_count_matches_nest_within_tolerance`).
- No dedicated regression yet pins the **start-exclusive / stop-inclusive**
  boundary behavior at `t == start` and `t == stop` exactly (the stamp-step
  test exercises one boundary but not the full parametric sweep).
- **Genuinely open (P2).** No tests cover the **recording-backend** divergence
  (no file-backed `ascii` / `sionlib` recorders in repo).

## 7. Prioritized roadmap

- **P0 — Make recording-device parity concrete with a canonical test.** ✅ **Done.**
  The user-facing `nest-status/index.rst:93-94` flags recording semantics as
  divergent; recorder parity is now pinned concretely by two live-NEST tests:
  `brainpy_state/_nest/_validation/multimeter_file_test.py` runs an
  `iaf_psc_exp` + `multimeter` against live NEST and asserts every recordable
  (`V_m`, `I_syn_ex`, `I_syn_in`) matches at category B via `compare_trace`,
  and `brainpy_state/_nest/_validation/device_parity_test.py` asserts
  `spike_recorder` exact event count + stamp-step parity. (No standalone
  `recorder_parity_test.py` is needed — these two files cover it.)

- **P0 — Document the stamp-step gating convention.** [S]
  Rationale: `multimeter.py:96-99` and `spike_recorder.py:51-52` are precise
  about start-exclusive / stop-inclusive — but the convention is not in a
  user-facing doc. Acceptance: device-gating convention (with formula) appears
  in `docs/api/nest-devices.rst` or the future `docs/nest-guide/`.

- **P1 — Boundary regression for `start`/`stop`/`origin`.** [S]
  Acceptance: parametric test runs at `t == start`, `t == start + dt/2`,
  `t == stop`, `t == stop + dt`, and asserts inclusion/exclusion of events.

- **P1 — `weight_recorder` per-rule hookup test.** ✅ **Done (cluster-09).**
  `_validation/weight_recorder_audit_test.py` verifies event count + timing +
  value against NEST for all 13 plastic rules via the send-view seam (cross-link
  `synapses-plasticity-gap.md` P0 audit).

- **P1 — Noise generator dt-invariance test.** [S]
  Acceptance: same `mean`, `std` parameters with `dt = 0.1 ms` and `dt = 0.05 ms`
  produce equivalent spectral density (estimated via Welch over 10 s).

- **P1 — Correlation-detector window + normalization parity.** [M]
  Acceptance: each correlation-detector variant matches NEST's normalized
  covariance over a 5 s window for a known 2-source pattern.

- **P1 — `spike_generator` off-grid times convention.** [S]
  Acceptance: documented in docstring + test asserts repo behavior matches
  NEST `precise_times=True` semantics or documents the difference.

- **P2 — File-backed recording backends.** [L] (genuinely open)
  Rationale: NEST supports `ascii`, `sionlib`, etc. `brainpy.state` only
  supports in-memory PyTree storage — no file-backend equivalents exist. Most
  users of brainpy.state are JAX-pipeline users who don't need file backends,
  but porting NEST scripts that rely on them requires the equivalent. The
  ported scripts `examples/nest_like/multimeter_file.py` and
  `examples/nest_like/recording_demo.py` reproduce the trace content in memory but
  **skip the file-backend (`set_data_path` / `.dat` write) step**. Acceptance:
  at minimum an `ascii`-equivalent backend that writes to `.dat` files on
  `Simulate()` completion is available for `spike_recorder` and `multimeter`.

- **P2 — Stamp-step bit-exact recorder test.** [M]
  Rationale: confirm that for analytical-propagator neurons (Category B),
  recorder traces are bit-equal to NEST. Acceptance: `iaf_psc_alpha` + `mm`
  trace difference is exactly zero modulo float-ulp at every recorded step.
