# /goal — 15a · Rate-neuron core: de-queue + seam-(H) continuous `rate` emission + `InputDelay` rate connections

> Self-contained prompt. Open at repo root. Stream: synapse-substrate + `_nest` rate models. **Bucket-3 foundation** — first of the four continuous-coupling clusters (15a → 15b gap, 15c siegert+diffusion, 15d astro), then the demos (17 rate/gap, 17b astro). Establishes the *de-queue + seam-(H)-emits-continuous-`rate` + `InputDelay`-on-emission* template that 15b/c/d reuse. Depends on: 12 (seam H `_emission_attr`), 22 (seam H generalized to continuous presynaptic state), 23 (`InputDelay` `delay_seam`) — all merged.

## The reframe (read first — it overturns the old index framing)

The long-standing **"build one monolithic `ContinuousCoupledProj` primitive #3"** plan is **superseded.** A grilled design session established (and the codebase confirms) all of:

- The rate neurons are already **JAX-native** (`τ dX/dt = −X + μ + φ(h)`); their only non-substrate baggage is a **host-side dict-queue event emulator** (`_delayed_ex_queue`/`_delayed_in_queue`, keyed by `step_idx`) bolted alongside the standard `sum_delta_inputs`/`sum_current_inputs` reads.
- A rate connection `r_post = Σ w·r_pre` **is literally the seam-(H) graded-emission path** (`weight·emission`, `comm='dense'`) that **22 proved generalizes to continuous presynaptic state** (recurrent NMDA, micro-parity 5e-15).
- The substrate is already **explicit one-step-lagged** (phase-1 reads previous-step state). NEST's `rate_connection_instantaneous` is `SUPPORTS_WFR=True`; reading the **previous-step rate is exactly NEST's WFR seed** — it preserves fixed points and diverges only in the transient. `rate_connection_delayed` (`delay_steps ≥ d_min`) is NEST-exact under `InputDelay`.

So there is **no new primitive.** Rate coupling = de-queue + seam-(H) emit `rate` + the existing `InputDelay`. This cluster lands that template; 15b/c/d add one small additive deposit variant each (difference, dual-channel, one-way) on top of it.

## Session start

1. Create an isolated worktree branch **`nest-goal/15a-rate-core`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`22-wang-nmda-network` Lessons (seam H generalized)** — `_emission_attr`(+`_emission_receptor`) → `_connect_pair` builds a plain projection depositing `weight·emission`; **graded emission requires `comm='dense'`** (sparse binarizes); the **micro-parity-first discipline** (the test is the arbiter, not the prior text — 14's "new projection" turned out unnecessary); pipeline-latency `align_steps`.
   - the **`12` Lessons (seam H + `for_loop` discipline)** — the emission-holder timing; **every Simulator-run model needs a `for_loop`-lowering regression test** (eager stepping hides carry-shape bugs).
   - the **`23-connection-introspection` Lessons (`InputDelay`)** — `proj.delay_seam = InputDelay((n_pre_pop,), rounded)` with dt-rounding (sub-half-grid → 0 steps); the realized-edge enumeration order.
   - the **`14-network-demos` Lessons** — distributional/fixed-point network parity (seed-mean/median in a documented band + the qualitative law; steady-state on the 2nd half).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.6 (the `lin_rate`/`rate_neuron_*` rows) and `synapses-plasticity-gap.md` §5 (the rate-coupling / WFR risk notes).
   - `brainpy_state/_nest/lin_rate.py` — `_lin_rate_base`: the `_delayed_ex/in_queue` dict system (`_accumulate_instant_events`/`_schedule_delayed_events`/`_drain_delayed_queue`/`_parse_event`/`_coerce_events`/`_queue_add`), `_common_inputs`, and where `linear_summation`/`mult_coupling`/the transfer function are applied; the 8 rate neurons (`rate_neuron_ipn`, `rate_neuron_opn`, `lin_rate`, `gauss_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `tanh_rate`, `threshold_lin_rate`) + their `_test.py`; `rate_connection_instantaneous.py` / `rate_connection_delayed.py` (the `SUPPORTS_WFR`/`HAS_DELAY`/`delay_steps` specs).
   - The substrate: `brainpy_state/_network/_event_proj.py` (the seam-H deposit path), `_network/_simulator.py` `_connect_pair` (the `_emission_attr` branch + the `receptor_type` bridge + phase-1 previous-state read), `brainpy_state/_brainpy/_delay.py` (`InputDelay`), `_network/_connection_introspection.py` (23's `delay_seam` dt-rounding).
   - NEST upstream: `models/lin_rate.{h,cpp}`, `models/rate_neuron_ipn_impl.h`, `models/rate_connection_instantaneous.h`, `models/rate_connection_delayed.h`; the example `pynest/examples/lin_rate_ipn_network.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — design question A (below) is the crux; resolve it with a rate-coupling micro-parity test before rebuilding all 8 neurons.

## Goal

Rebuild the 8 rate neurons + the two rate connections on the substrate: **delete the host-side dict-queue**, route rate coupling through **seam-(H) continuous `rate` emission** (`comm='dense'`), **instantaneous = explicit one-step lag**, **delayed = `InputDelay(delay_steps)`** — validated by a rate-coupling micro-parity test and a rate-network fixed-point parity against live NEST.

## Scope + the core design question

### A. seam-(H) continuous `rate` emission — **resolve FIRST via micro-parity** (record in Lessons)

22 deposits `weight·emission` for continuous presynaptic state into a **named receptor** (`_emission_receptor='s_NMDA'`). Rate neurons have **no receptor** — a single transfer-function input `h`. **Try:** declare the rate neuron's `_emission_attr='rate'` with **no `_emission_receptor`**, so the deposit lands in the **default delta-input channel** the neuron reads as `h` (`sum_delta_inputs`). Build a **rate-coupling micro-parity test** (two rate neurons + one instantaneous connection; compare the driven neuron's `rate` trajectory vs live NEST, deterministic given matched input + IC). Decide:

- **(a)** emission-**without**-receptor → default delta channel **suffices** → the only substrate change is making `_emission_receptor` optional in the seam-H branch. **(check this first.)**
- **(b)** the delta channel can't carry it (units, the ex/in weight-sign split, or the `linear_summation` φ-placement) → the **minimal additive rate-deposit variant** in `_connect_pair`/`_event_proj`, unit-tested, primitives intact.

The **`linear_summation`** flag selects *which State the presynaptic neuron emits*: `rate` (True — φ applied by the post to its summed input `φ(Σw·r + μ)`) vs `φ(rate)` (False — the pre applies its own φ first, post sums `Σw·φ(r)`). Both are seam-(H); confirm both against NEST.

### B. De-queue the rate neurons

Delete `_delayed_ex_queue`/`_delayed_in_queue` + `_accumulate_instant_events`/`_schedule_delayed_events`/`_drain_delayed_queue`/`_parse_event`/`_coerce_events`/`_queue_add` from the base(s). The neuron keeps its JAX `τ dX/dt = −X + μ + φ(h)` dynamics, reading `h = sum_delta_inputs(0.0)` (the seam-H rate deposit) and `μ = sum_current_inputs(x, rate)`. `mult_coupling` (`g_ex·(θ_ex−r)`, `g_in·(θ_in+r)`) and the ex/in weight-sign split stay **neuron-side** (they are neuron params). The de-queue must **not** change the JAX dynamics.

### C. Rate connections

- `rate_connection_instantaneous` (`SUPPORTS_WFR=True`, `delay≡0`) → **explicit one-step lag** (phase-1 reads previous-step `rate` = NEST's WFR seed; fixed-point-preserving, transient-divergent — **document the divergence**).
- `rate_connection_delayed` (`HAS_DELAY=True`, `delay_steps ≥ d_min`) → `proj.delay_seam = InputDelay(delay_steps)` (23's pattern; dt-rounded). NEST-exact under lag.

## Constraints

- seam-(H) rides **`comm='dense'`** (22: `comm='sparse'` binarizes graded emission) — rate nets are small/structured, dense `x @ W` is fine; note the **sparse-graded-emission extension as explicitly out-of-scope** (a future cluster).
- Any neuron/substrate change **RED-first** (`superpowers:test-driven-development`) + a **`for_loop`-lowering regression test** (12). Drive every multi-step run with `brainstate.transform.for_loop`/`scan`, **never a Python loop** (CLAUDE.md #10).
- **No-coupling de-queue regression:** pin a single rate neuron's `τ dX/dt` trajectory (μ-drive only, no connection) vs NEST **before and after** the de-queue — proves the dict-queue removal didn't perturb the dynamics.
- `Simulator` API for the network tests; `saiunit` units; the two-namespace rule (inner source/tests import `brainpy_state`; any example uses `brainpy.state`).
- NumPy docstrings; >90 % coverage on touched `_nest`/`_network` code. Coverage via `coverage run --source=brainpy_state` + the coverage API (the `--cov=pkg.mod` form coredumps under jaxlib; the rtk proxy strips `--cov`).

## Validation (via the 00 harness)

`_nest/_validation/rate_core_test.py` + the rate-coupling micro-parity, `requires_nest`:

- **Micro-parity (precondition):** two rate neurons + one **instantaneous** connection; driven `rate` trajectory vs NEST (tight band; deterministic given matched input + IC). A **delayed**-connection variant: the response shifts by exactly `delay_steps`.
- **Network fixed point:** a small **linear-rate** net (`lin_rate`) relaxes to its analytic fixed point `r* = (I − W)⁻¹ μ`; assert vs **both** the closed form **and** NEST. A **nonlinear** net (`sigmoid_rate`/`tanh_rate`) steady-state (2nd-half mean) vs NEST (fixed-point/distributional band, 14 posture).
- **Per-transfer-function** unit tests (`gauss`/`sigmoid`/`tanh`/`threshold_lin`/`sigmoid_gg`) vs NEST's φ on a swept input.
- **No-NEST companion:** imports + a short headless `for_loop` rate-relaxation run produces the fixed point.

## Edge cases to enumerate + test

`linear_summation` True vs False (emit `rate` vs `φ(rate)` — same fixed point, different transient); `mult_coupling` on/off; an **instantaneous recurrent loop** (explicit-lag breaks the algebraic loop — fixed point matches NEST, transient differs; document); zero-weight / self-connection (no autapse per spec); delayed connection at `delay_steps = d_min` and `> d_min`; mixed-sign (ex/in) weights into one target; a μ-only net (no coupling) ≡ the de-queued single-neuron trajectory (the regression above); `rate_neuron_ipn` (input-noise) vs `rate_neuron_opn` (output-noise) placement; `dt`-invariance of the fixed point; standalone run without NEST; `for_loop` lowering of the full net.

## Definition of done

- [ ] Design question A resolved by the micro-parity test (receptor-optional seam-H into the delta channel suffices, **or** a minimal rate-deposit variant landed additive + unit-tested); decision + the `linear_summation` emit-selection recorded in Lessons.
- [ ] 8 rate neurons de-queued (dict event system deleted) + the JAX-dynamics-unchanged regression pinned; both rate connections on seam-H / `InputDelay`.
- [ ] Rate-coupling micro-parity + linear-rate fixed-point + nonlinear steady-state parity passing (live NEST) / skipping (without); a `for_loop`-lowering test.
- [ ] >90 % coverage on touched code; examples-gap §3.6 rate rows + README updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (receptor-optional seam-H continuous emission into the delta channel, or the minimal rate-deposit variant), the **`linear_summation` emit-`rate`-vs-`φ(rate)`** mechanism, the **de-queue** (exactly what was deleted + the JAX-dynamics-unchanged proof), the **instantaneous-lag = WFR-seed** fixed-point-preserving result + its documented transient divergence, the **delayed-`InputDelay`** wiring, and confirmation that **the rate-core template is ready for `15b` (gap difference-deposit), `15c` (diffusion dual-channel), `15d` (astro one-way SIC)** to build on. Note the sparse-graded-emission extension as deferred.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
