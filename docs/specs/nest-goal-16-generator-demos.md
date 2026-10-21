# /goal — 16 · §3.7 generator-pattern demos (sinusoidal Poisson/gamma, pulse-packet)

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 02 (Simulator/devices), 03 (recording + `compare_distributional` autocorr/CV + the eager-device "bypass don't gut" rule) — merged. Independent of 23. **§3.8 astrocytes are NOT in scope** (they need `sic_connection` + the astrocyte rate model ⇒ bucket-3; defer to the post-`15` astro cluster). This ports the bucket-3-**independent** §3.7 generators.

## Session start

1. Create an isolated worktree branch **`nest-goal/16-generator-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially:
   - the **`03-recording-demos` Lessons** — the eager-device **"tap State / run the detector post-hoc, never inside the `for_loop`"** rule, the completed **`compare_distributional` autocorr/CV** path (you reuse it for rate-modulation + ISI statistics), `CAT_D` for PRNG-divergent counts, and the recorder one-step offset.
   - the **`02` Lessons** — analog/spike recording (`res.trace`/`res.spikes`/`res.rate`), multi-channel generators, the `examples/nest/` house style.
   - the **`14` Lessons** — distributional network parity vocabulary (seed-mean/median in a documented band, qualitative-law assertions) for any demo that drives a population.
   Then read:
   - `docs/nest-status/internal/examples-gap.md` **§3.7** (`sinusoidal_poisson_generator`, `sinusoidal_gamma_generator`, `pulsepacket`) and `devices-gap.md` (the generator rows + any documented divergence).
   - The generator models (all present): `brainpy_state/_nest/{sinusoidal_poisson_generator,sinusoidal_gamma_generator,pulsepacket_generator,inhomogeneous_poisson_generator,ppd_sup_generator}.py` (+ their `_test.py`).
   - The house style for device demos: `examples/nest/{cross_check_mip_corrdet,recording_demo}.py` + `../../README.md`.
   - The NEST upstream scripts: `sinusoidal_poisson_generator.py`, `sinusoidal_gamma_generator.py`, `pulsepacket.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Port the §3.7 generator demos to `examples/nest/<name>.py` on the `Simulator` API, each with a live-NEST **distributional** parity test, exercising the rate-modulated / pulse-packet generators and driving any generator-device fix back into `_nest/<model>.py`.

## Scope

**Port now (all bucket-3-independent):**
- `sinusoidal_poisson_generator` → a rate-sinusoid `λ(t)=dc+ac·sin(ωt+φ)` Poisson drive into a recorder/parrot population; assert the **time-resolved rate** (PSTH) tracks `λ(t)` and the **spike-count autocorrelation** vs NEST (`compare_distributional`, `CAT_D`). Cover the multi-target **individual-vs-shared** noise modes if the model exposes them.
- `sinusoidal_gamma_generator` → gamma-process (order `m`) rate-modulated drive; assert the modulated rate **and** the **ISI CV → 1/√m** (the gamma regularization — the demo's headline), `compare_distributional` CV path.
- `pulsepacket` → `pulsepacket_generator` emits Gaussian-jittered synchronous spike packets (`sdev`, `pulse_times`, `a`); drive a population, assert the **packet shape** (output spike-time histogram width/amplitude) vs NEST distributionally.

## Constraints

- `Simulator` API only; `examples/nest/` house style; `saiunit` units.
- Generators stay imperative-but-bypassed: **tap State / post-process recorded arrays** (03 rule); nothing imperative inside the `for_loop`. If a generator's draw needs a PRNG key inside `for_loop`, use the `mcculloch_pitts` self-managed-key pattern (03) / the `bernoulli` shape-`(E,)` key idiom (06).
- Parity is **distributional** (NEST per-thread RNG vs JAX) — assert rate/ISI/packet **statistics** in a documented band + the qualitative law (rate tracks `λ(t)`, CV→1/√m, packet width ∝ `sdev`), never per-sample spike times. Set the NEST recorder `interval=dt`.
- Any generator-device bug fixed in `_nest/<model>.py` (reproduce first, `superpowers:systematic-debugging`); add a **`for_loop`-lowering regression test** for any generator the Simulator runs (12 discipline).
- NumPy docstrings; >90 % coverage on touched code. Coverage via `coverage run --source=brainpy_state` + the coverage API (the `--cov=pkg.mod` form coredumps; rtk proxy strips `--cov`).

## Validation (via the 00 harness)

`_nest/_validation/<name>_test.py`, `requires_nest`: rate-modulation PSTH + autocorr (`sinusoidal_poisson`), modulated rate + ISI-CV (`sinusoidal_gamma`), output-packet histogram (`pulsepacket`) — each `compare_distributional` in a documented band. Each example runs standalone (`PYTHONPATH=. python …`) without NEST (parity skips, demo still plots its PSTH/raster). README §3.7 rows.

## Edge cases to enumerate + test

`ac=0` (sinusoidal_poisson ≡ stationary Poisson at `dc`); `ac>dc` (rate clamped ≥0 — assert no negative-rate spikes); gamma order `m=1` (≡ Poisson, CV→1); `m→∞` (regular, CV→0); pulse-packet `sdev=0` (perfectly synchronous), multi-packet trains, `a=0` (empty); individual vs shared multi-target modulation; `dt` invariance of the statistics; standalone run without NEST.

## Definition of done

- [ ] 3 §3.7 generator demos ported + runnable standalone, each with a distributional live-NEST parity test (pass / skip).
- [ ] Any generator-device fix TDD-covered + `for_loop`-lowering-tested; §3.8 astro explicitly deferred to the post-bucket-3 astro cluster in README + Lessons.
- [ ] >90 % coverage on touched code; README §3.7 updated; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the generator-demo parity recipe (which statistic + band per generator — rate/CV/packet), any generator-device fix + its `for_loop` lowering, the multi-target individual-vs-shared mode handling, and confirmation that §3.7 is complete + §3.8 astro is the one deferred §3.7/3.8 item (waiting on bucket-3 `sic_connection`).
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
