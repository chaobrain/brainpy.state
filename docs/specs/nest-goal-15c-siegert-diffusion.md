# /goal — 15c · Siegert + diffusion: dual-channel seam-H (`drift→μ`, `diffusion→σ²`) + leggauss-64 → JAX

> Self-contained prompt. Open at repo root. Stream: synapse-substrate + `_nest` Siegert neuron. **Bucket-3 child** of `15a` (rate-core). Adds **one** new deposit variant — the *dual-channel deposit* (one emission → two target channels with two factors) — and ports the Siegert mean-field quadrature to JAX so it runs inside the `for_loop`. Depends on: `15a` (seam-H continuous emission + de-queue + lag template), `22` (seam H), `23` (`InputDelay`, if delayed diffusion) — **run `15a` first**.

## What 15a established (build on it)
`15a` resolved seam-(H) continuous emission (emit `rate`, `comm='dense'`, default channel) + the de-queue + the instantaneous-lag posture. Diffusion connections reuse the emission but deposit into **two** channels (mean μ + variance σ²); the Siegert neuron is a **mean-field** node reading both to compute its rate via the LIF transfer Φ(μ,σ). `diffusion_connection` is `SUPPORTS_WFR=True` → explicit-lag, like instantaneous-rate.

## Session start
1. Create an isolated worktree branch **`nest-goal/15c-siegert-diffusion`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`15a` Lessons** — seam-H continuous emission (receptor-optional), the de-queue template, the instantaneous-lag = WFR-seed posture.
   - the **`22` Lessons (seam H generalized)** — `_emission_attr` deposits `weight·emission` for continuous presynaptic state; `comm='dense'`; **micro-parity-first** (the test is the arbiter, not prior text).
   - the **`23` Lessons (`InputDelay`)** — `proj.delay_seam = InputDelay(...)` dt-rounding (for delayed diffusion, if exercised).
   Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.6 Siegert/mean-field rows + `synapses-plasticity-gap.md` §5 (diffusion / WFR notes).
   - `brainpy_state/_nest/siegert_neuron.py` — the `scipy.integrate.quad`/`scipy.special` path **and** the host-side 64-point `np.polynomial.legendre.leggauss` path (erfcx = `exp(x²)·erfc(x)`, Dawson's integral, the asymptotic expansions), plus the same dict-queue to delete; `diffusion_connection.py` (`drift_factor`/`diffusion_factor`, `prepare_secondary_event` packaging drift + diffusion).
   - The substrate: `brainpy_state/_network/_simulator.py` `_connect_pair` (the seam-H branch), `_network/_event_proj.py`; how a neuron exposes **two** input accumulation channels (μ vs σ²).
   - NEST upstream: `models/siegert_neuron.{h,cpp}` (the `nest::siegert` algorithm), `models/diffusion_connection.h`, and any siegert/mean-field example or testsuite reference.
3. Write spec + plan under ``. **Approach-first; wait for approval before code** — design questions A (dual-channel) + B (JAX quadrature) are the sign-off; resolve A with a Siegert-driven-by-diffusion micro-parity, B against the scipy oracle.

## Goal
Make `siegert_neuron` a first-class JAX `for_loop` mean-field node (leggauss-64 quadrature ported to jnp, scipy as oracle), reading μ/σ² via a **dual-channel** `diffusion_connection` deposit, validated by the transfer-vs-scipy grid + a self-consistent mean-field rate against live NEST.

## Scope + the core design questions

### A. The dual-channel deposit — **resolve via micro-parity** (record in Lessons)
`diffusion_connection` carries two factors: it deposits `drift_factor·rate` into the target's **μ** channel and `diffusion_factor·rate` into the **σ²** channel. **Try** the simplest realization — **two parallel seam-(H) deposits** from the same pre emission (reusing 15a's single-channel path twice: one into μ, one into σ²) — and confirm it equals NEST. Build a **Siegert-driven-by-diffusion micro-parity** (one Siegert neuron + one diffusion connection with known drift/diffusion; compare its rate vs NEST). Decide:
- **(a)** two parallel seam-H deposits (μ, σ²) **suffice** → no new projection, just a second channel target. **(check this first.)**
- **(b)** they must be one atomic dual-channel projection (shared edge order / single secondary event) → the minimal dual-channel variant. **Prefer (a)**; (b) only if the micro-parity demands atomicity.
The Siegert neuron reads both channels (μ-accumulation + σ²-accumulation) and computes `rate = Φ(μ, σ)`.

### B. Port the leggauss-64 quadrature to JAX (scipy = oracle)
Re-express the 64-point Gauss-Legendre transfer (erfcx via `jax.scipy.special.erfc`; Dawson's integral in jnp; static nodes/weights) so Siegert lowers under `for_loop`. **Keep the scipy path as the host-side parity oracle** in tests. Validate JAX-Φ vs scipy-Φ across the (μ,σ) grid; **port the asymptotic expansions** if leggauss-64 underperforms in a regime (large |μ|, small σ). De-queue Siegert (delete its dict event system; same template as 15a).

## Constraints
- seam-(H) `comm='dense'` (mean-field nets are dense — fine). Delayed diffusion (if exercised) → `InputDelay` (23).
- The JAX quadrature must match scipy to a **documented tolerance** across the validated (μ,σ) range; assert it.
- **RED-first** + a **`for_loop`-lowering regression test** (12); `brainstate.transform.for_loop`/`scan`, never a Python loop (CLAUDE.md #10).
- **`clear_caches()` + float64 guard** (the Siegert quadrature is precision-sensitive; 21 discipline).
- NumPy docstrings; >90 % coverage. Coverage via `coverage run --source=brainpy_state` + the coverage API.

## Validation (via the 00 harness)
`_nest/_validation/siegert_diffusion_test.py` + the micro-parity, `requires_nest`:
- **Quadrature oracle (NEST-free + NEST):** JAX-Φ(μ,σ) vs the scipy path across a (μ,σ) grid (tight, documented tolerance); vs NEST's siegert rate where available.
- **Dual-channel micro-parity (precondition):** one Siegert + one diffusion connection (known drift/diffusion) → rate vs NEST (deterministic given matched input).
- **Self-consistent mean-field:** a population of Siegert neurons + diffusion connections relaxes to the mean-field fixed point; compare vs NEST's siegert **and** the numerical mean-field solution (e.g. the Brunel mean-field rate) — fixed-point band.
- No-NEST companion: imports + a headless Siegert relaxation to its fixed point.

## Edge cases to enumerate + test
`diffusion_factor=0` (drift-only → deterministic μ-driven rate); `drift_factor=0` (variance-only); the erfcx/Dawson regimes leggauss-64 struggles with (large |μ|, small σ → assert the asymptotic-expansion fallback); negative drift (sub-threshold → low rate); the self-consistent loop's existence/uniqueness on both sims; `dt`-invariance of the fixed point; `clear_caches()`/x64; standalone without NEST; `for_loop` lowering.

## Definition of done
- [ ] Design A resolved by the micro-parity (two parallel seam-H deposits suffice, **or** the atomic dual-channel variant); design B: JAX leggauss-64 (+asymptotics) matches scipy to a documented tolerance; both recorded in Lessons.
- [ ] Siegert de-queued + JAX-native (`for_loop`-lowered); `diffusion_connection` dual-channel wired; scipy retained as the oracle.
- [ ] Quadrature-oracle + dual-channel micro-parity + self-consistent mean-field parity passing (live NEST) / skipping; `for_loop`-lowering test; `clear_caches()`+x64.
- [ ] >90 % coverage; examples-gap §3.6 Siegert rows + README updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end
1. **Append a Lessons entry to `CONTEXT.md`**: the **A resolution** (dual-channel = two parallel seam-H deposits μ+σ², or the atomic variant), the **B resolution** (the JAX erfcx/Dawson leggauss-64 port + the asymptotic-expansion regimes + the scipy-oracle tolerance), the Siegert de-queue, and confirmation the mean-field path is ready for the `17`/§3.8 demos.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity numbers) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
