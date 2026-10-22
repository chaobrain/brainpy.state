# K0 T5: delays, network, compiler, runner, observers

_Date: 2026-10-22_ · Parent: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)
§4.4 (delays), §4.5 (RNG), §4.8 (network/compiler/runner), §5 row T5, §6 G1/G4/G9, §7.
Architecture doc §7 (step contract, §7.1 timing, §7.2 RNG).

## Modules

| module | contents |
|---|---|
| `_kernel/delays.py` | `delay_steps` (NEST rounding), `classify_delays -> DelayLayout`, `PreRing`, `PostRing` |
| `_kernel/network.py` | `Network`, `Population`, `PopulationView`, `Segment`, `Projection` |
| `_kernel/compiler.py` | `schedule`, `ScheduleError`, `compile_network -> Plan`, `PopulationPlan`, `ProjectionPlan`, `EdgeGroup`, `ObserverPlan` |
| `_kernel/runner.py` | `Program` (pure functional step/rollout), `Runner` (`simulate`, `cont`, `step`, `reset`) |
| `_kernel/observers.py` | `Observer`, `Recording`, `Results`, window accumulators |
| `_kernel/validation/brunel.py` | Brunel alpha builder (tests, benchmarks) |

## Delays

- `event` delays are whole steps: `floor(delay_ms * steps_per_ms + 0.5)`, NEST's
  `Time::delay_ms_to_steps` (`ld_round`). Inputs stored in float32 (x64 off) that lie within
  float32 resolution of a half step are taken as exact halves (NEST receives the decimal:
  `0.35 ms` is `3.5` steps there, `3.4999999` in float32).
- A delay of `d` steps delivers a spike emitted in step `n` (NEST stamp `(n+1) dt`) to the
  receiver's update in step `n + d`. A delay rounding to 0 is a zero-delay connection.
- Per projection, `classify_delays` gives `scalar`, `per_source` (constant per pre neuron)
  or `per_edge`. Placement (`delay_buffer='auto'`): scalar and per-source delays use a
  **pre-side ring** per `(population, output port)` of length `max delay` (shared by all
  projections reading that port); per-edge delays use a **post-side arrival ring**
  `(max delay, n_post)` per projection, with the edges grouped by delay value (one storage
  per group; zero-delay groups bypass the ring). `delay_buffer='post'` forces the post side,
  `'pre'` refuses per-edge delays.

## Step contract (per step `n`)

1. For each population in schedule order: **deliver** (pre-side ring read at `n - d`, or
   the post ring slot `n`, or, for `d = 0`, the current step's output of an already updated
   population), then **update** (`model.step(ctx, inputs)`).
2. **Emit**: write every pre ring slot `n % L`; for post rings clear slot `n % L`, then
   deposit each delay group `k >= 1` at `(n + k) % L`.
3. **Observe**.

Current ports are latched by the models (T3), so a current takes effect one step after it
is delivered. `ctx = StepContext(t=n dt, dt, step=n, key)`, with
`key = fold_in(population_key(seed, uid, n), trial)` (trial 0 when unbatched or `common`).

## Schedule

Insertion order, constrained by zero-delay edges (Kahn's algorithm, smallest index first).
A zero-delay self-connection or cycle raises `ScheduleError` at compile.

## Compiler

`net.compile(precision)` instantiates every model (`key = block_key(object_key(seed, uid),
INIT_STREAM)`), builds each projection's `EdgeSet` with `object_key(seed, uid)`, realises
weights (`WEIGHT_STREAM`) and delays (`DELAY_STREAM`) for distributions, chooses storage
(`make_storage`), the payload path (`payload_path`: binary only for
`Event(max_multiplicity=1)` sources) and the buffer placement. Weights live in
`ParamState` (`trainable=True`) or `ConstParamState`. The `Plan` is a
`brainstate.nn.Module`; `plan.describe()` prints the schedule, populations, projections
(edges, storage, payload path, delay layout, buffer side), buffers and memory, merges
(none until T6), observers and capabilities.

## Runner

- `Program(plan)`: `init_state()`, `read_params()`, `step(params, state, n, trial)`,
  `rollout(params, state, n0, trial, n_steps)` (a `brainstate.transform.scan`). Pure: all
  parameters and weights are arguments, so `jax.grad`/`vmap` apply directly.
- `Runner(plan, batch=None, rng='independent')`: `simulate(T)` resets then runs; `cont(T)`
  continues (state, step counter and RNG streams carry over); `step()` runs one step and
  returns the outputs. The rollout is `brainstate.transform.jit` with `n_steps` static:
  repeated `cont` of one duration never re-traces (`runner.n_traces`), and changing a
  parameter or weight value does not either. Batches use `jax.vmap` over state and trial.

## Observers

`net.observe(view, *recordables, interval=None)`: any state of every model in the view or
an `Event` output. With `interval = k` steps, events are summed over each window and
states sampled at the window's last step; times are the window end stamps. Run durations
must be multiples of every interval. `Results[observer]` gives a `Recording` with
`data[name]` `(n_samples, n)` (`(batch, n_samples, n)` when batched), `times`, `gids`,
`columns`, `by_population`, `spike_events`.

## Extension hooks (raise `NotImplementedError`)

`connect(synapse=...)`, `connect(plasticity=...)` (T6); `TimedEvent` ports (T7);
gated `Conductance` ports (T8); `plan.runner(learner=...)` (T9).

## Decisions

1. `check_kind_compatible` allows `Event -> Current/Conductance` when the port is
   `filtered` (the receiving model owns the kinetics, e.g. `iaf_cond_exp.ex`; T3 notes).
2. `poisson_generator` draws with `poisson_counts` (bit-identical to `jax.random.poisson`;
   a Knuth fast path under `lax.cond(all(lam < 10))`, because the traced-`lam` call
   evaluates both of JAX's branches, about 7x slower). *Superseded in T10:* the fast path
   is now CDF inversion (one uniform per element), exact in distribution but no longer
   bit-identical to `jax.random.poisson`, which stays the `lam >= 10` branch; see
   [`2026-10-22-k0-t10-gates.md`](2026-10-22-k0-t10-gates.md) §3.
3. `connect` also takes `source_port`, `delay_buffer`, `trainable` and `name`; defaults:
   `rule=AllToAll()`, `weight=1.0`, `delay=dt`.
4. `Network` is a plain class; `Plan` is the `brainstate` module. Models are built at
   compile time; each `compile()` gives fresh models and states.
5. For an empty source or target view, the weight keeps its own unit (nothing is
   delivered).

## Edge cases tested

Zero-size populations (both sides, observers); partially overlapping slices with
`allow_autapses=False` on global ids; concatenated views on both sides; zero-delay chains
inserted in reverse order (delivered in one step) and zero-delay cycles (error); delays
`0.04 ms -> 0`, `0.15 ms -> 2`, `0.25/0.35 ms` against NEST; per-edge vs scalar delays
equality; per-source delays; delay distributions; batch 1 vs 4 trial 0 for `independent`
and `common`; `simulate(T1+T2) == simulate(T1) + cont(T2)`; no re-trace on `cont` or a
parameter change; float32 and float64; interval observers; gradients through
`Program.rollout`; single-spike and DC timing against live NEST (`align_steps=0`, CAT_B).
