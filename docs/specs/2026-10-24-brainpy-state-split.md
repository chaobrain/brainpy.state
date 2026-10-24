# Split `brainpy_state`: BrainPy-style API → `brainpy.state`, the rest → `brainspike`

Date: 2026-10-24. Source snapshot: `brainpy-state` `main` @ `0f2e3ab`.

## Goal

`brainpy_state` held two different things behind one `brainpy.state` namespace: the
BrainPy-style models (`_brainpy`, plus `_base` / `_mixin` / `_misc`) and the
NEST-compatible stack with the new kernel (`_nest_*`, `_kernel`). The two
distributions also depended on each other (`brainpy` → `brainpy_state>=0.0.3` →
`brainpy>=2.7.6`), held together by an import guard (`_namespace.py`) and a version
guard (`_compat.py`).

After the split:

- **`brainpy.state`** (BrainPy ≥ 2.9.0) is in-tree BrainPy code again: the
  BrainPy-style models, part of general-purpose brain dynamics programming.
- **`brainspike`** (this repo, 0.2.0, continuing the brainpy-state 0.1 line) is the
  BrainX package for point-neuron spiking networks: building, simulating and
  training them, unified by the kernel language. It sits beside braincell
  (detailed cells) and brainmass (neural mass).

## Decisions

| | Decision | Reason |
|---|---|---|
| S-1 | Name: `brainspike`. | Says what the package is for (spiking point neurons); free on PyPI and GitHub. |
| S-2 | Hard break: `brainpy.state` forwards no NEST, `Simulator`, `dist` or `spatial` names. | One home per name; forwarding would bring back the dependency cycle. |
| S-3 | brainspike keeps its own copy of the base classes it needs (`_base.py`; `InputDelay` in `_nest_network/delay.py`). It has **no runtime dependency on brainpy**. | Removes the cycle. `brainpy` is a dev-only dependency, used only by the kernel parity tests (`requirements-dev.txt`). |
| S-4 | **D-e is carried out early.** The BrainPy-style API leaves this repo now instead of being deprecated at K3 and deleted at K4. It is not deprecated in `brainpy.state`. | The redesigned kernel has its own `LIF` / `Expon` / … and never needed the BrainPy-style classes. Moving them out does what D-e planned and keeps them for BrainPy users. The K2 item "BrainPy-family models ported for use through `Network`" is unchanged: those are new kernel models. |
| S-5 | **§7.4 is settled:** at K3 the kernel exports to the `brainspike` top-level namespace (option (a)). | The BrainPy-style names no longer share the namespace, so most of the 18 collisions counted on 2026-10-22 are gone. The remaining NEST-name collisions are staged as in §7.4 (a). |
| S-6 | The BrainPy-style network path is deleted now: `Network` (old), `Builder`, `Recorder`, the eight rule-based `*Proj` classes, the Brunel builder, and their tests. | They exist only to serve the BrainPy-style models. The NEST `Simulator` path (`EventProjection` + `InputDelay`) stays until K4 (D-f). This shrinks the K1b inventory "every old `_nest_network` behaviour test re-expressed on `Network`": the deleted tests drop out of it. |
| S-7 | `SpikeTime` is copied into `brainspike` (`_nest_device/spike_time.py`). | The NEST examples need per-neuron spike times, and `spike_generator` broadcasts one list. |
| S-8 | No cross-references: neither package's docs, README or changelog describe how `brainspike` relates to `brainpy.state`. | Each package documents itself. The record of the split is this spec. |
| S-9 | Code examples use `import brainspike`, never an alias (`bsp`, `bp`, …). | One spelling everywhere; the script rewrites aliases on every run. |
| S-10 | The rename is a scripted, idempotent rewrite (`scripts/rename_to_brainspike.py`) in its own commit. | Open branches can rebase by re-running it. It skips `docs/specs/` and `changelog.md`: specs are dated records. |

Dependency graph (no cycle):

```
brainpy    → brainstate, brainunit, brainevent, braintools
brainspike → brainstate, brainunit, brainevent, braintools   (brainpy: dev only)
```

## Steps

1. BrainPy (`feat/state-in-tree`): copy `_brainpy/*`, `_base`, `_mixin`, `_misc` and
   their tests into `brainpy/state/`. Then rewrite the imports, add an explicit
   `__all__`, drop the `brainpy_state` dependency, bump to 2.9.0, move the
   BrainPy-style docs and examples, and add a changelog entry with a migration
   section.
2. This repo (`refactor/rename-brainspike`), in three commits:
   - Commit A: the rename script and its run.
   - Commit B: structural deletions and copies (S-3, S-6, S-7); `brainpy` out of
     the runtime dependencies.
   - Commit C: README, AGENTS.md rule 9, docs, CI.
3. Downstream (one branch each):
   - braintrace, braincell, brainstate: depend on `brainpy>=2.9.0`.
   - braintools: the `downstream.yml` matrix.
   - brainx: pins, tests and docs.
   - The website: slug `brainspike`, plus 301 redirects from `/brainpy-state/`.
   - BrainX-skill: skill rename.
4. Release in this order:
   1. brainpy 2.9.0.
   2. brainspike 0.2.0.
   3. Downstream pins and brainx.
   4. The website redirects.

   The GitHub repo rename `chaobrain/brainpy.state` → `chaobrain/brainspike` and the
   PyPI registration are done by hand.

## Verification

- `python -c "import brainspike, sys; assert not any(m == 'brainpy' or m.startswith('brainpy.') for m in sys.modules)"`
  confirms there is no runtime dependency on brainpy.
- Run `pytest brainspike/ -m "not requires_nest"` with brainpy 2.9.0 installed, so the
  parity tests have it.
- `grep -rn "brainpy_state\|brainpy\.state" brainspike/` should hit only the kernel
  parity tests and the docstrings that cite their reference models.
- The docs build with no links to the moved pages, and
  `grep -rn "import brainspike as"` finds nothing.
- BrainPy: `pytest brainpy/` passes, `brainpy.state.LIF(10)` works without
  `brainpy_state` installed, and the BrainPy docs build.
