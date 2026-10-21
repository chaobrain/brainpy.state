# Spec: Restructure the `brainpy_state` package

Status: **Approved (design)** · Date: 2026-06-16 · Scope: package layout + doc moves
Location: gitignored (`../../dev`), per CLAUDE.md rule 7.

## 1. Goal

Dissolve the flat `brainpy_state/_nest/` directory (~120 models, ~248 `.py` files)
into category packages, promote/rename two trees, and relocate internal dev docs.
The change is **structural only** — the public API (`brainpy_state.X` /
`brainpy.state.X`) and every public name must remain byte-for-byte identical.

## 2. Target layout

```
brainpy_state/
├── _base.py                 # unchanged (Dynamics/Neuron/Synapse — the brainpy-level base)
├── _brainpy/                # unchanged
├── _nest_base/              # NEW  ← _nest/_base.py, _utils.py, _plastic_base.py
├── _nest_neuron/            # NEW  ← neuron models
├── _nest_synapse/           # NEW  ← static synapses, connections, short-term plasticity
├── _nest_plasticity/        # NEW  ← STDP + voltage-based learning rules
├── _nest_device/            # NEW  ← generators, recorders, detectors
├── _nest_network/           # RENAME ← _network/
├── _nest_validation/        # PROMOTE+RENAME ← _nest/_validation/
├── _nest_spatial/           # unchanged
└── __init__.py              # rewritten import block; __all__ unchanged
```

`_nest/` is deleted. No backward-compat shim (it is a private module; nothing
public imports it).

## 3. File → package mapping (verified, 0 unclassified)

Membership lifted from the comment groups in `_nest/__init__.py` and the published
`docs/api/nest-*` taxonomy. A categorization pass over all 248 files left **zero**
unclassified. Each model file moves **with its `_test.py` sibling(s)**.

### `_nest_base/` (shared infra)
`_base.py`, `_utils.py`, `_plastic_base.py`, `_utils_test.py`, `jit_compat_test.py`, new `__init__.py`.

### `_nest_neuron/` (67 model stems)
All `iaf_*`, `aeif_*`, `gif_*`, `glif_*`, `amat2_psc_exp`, `mat2_psc_exp`,
all `hh_*`, `ht_neuron`, `izhikevich`, all `pp_*`,
`erfc_neuron`, `ginzburg_neuron`, `mcculloch_pitts_neuron`,
`gauss_rate`, `lin_rate`, `sigmoid_rate`, `sigmoid_rate_gg_1998`, `tanh_rate`,
`threshold_lin_rate`, `rate_neuron_ipn`, `rate_neuron_opn`, `rate_transformer_node`,
`siegert_neuron`, `astrocyte_lr_1994`, `ignore_and_fire`, `parrot_neuron`, `cm_default`.
(Includes test-only siblings `aeif_cond_beta_multisynapse_wbyrec_test.py`,
`gif_cond_exp_multisynapse_wbyrec_test.py`, `cm_default_state_test.py`.)

### `_nest_synapse/` (13 stems)
`static_synapse`, `static_synapse_hom_w`, `bernoulli_synapse`, `cont_delay_synapse`,
`gap_junction`, `diffusion_connection`, `sic_connection`,
`rate_connection_delayed`, `rate_connection_instantaneous`,
`tsodyks_synapse`, `tsodyks_synapse_hom`, `tsodyks2_synapse`, `quantal_stp_synapse`.

### `_nest_plasticity/` (15 stems)
`stdp_synapse`, `stdp_synapse_hom`, `stdp_triplet_synapse`, `stdp_dopamine_synapse`,
`stdp_facetshw_synapse_hom`, `stdp_nn_pre_centered_synapse`, `stdp_nn_restr_synapse`,
`stdp_nn_symm_synapse`, `stdp_pl_synapse_hom`,
`clopath_synapse`, `urbanczik_synapse`, `vogels_sprekeler_synapse`, `ht_synapse`,
`jonke_synapse`, plus `stdp_docs_crosslink_test.py`.

### `_nest_device/` (27 stems)
`ac_generator`, `dc_generator`, `noise_generator`, `step_current_generator`,
`step_rate_generator`, `host_drive`, `spike_generator`, `spike_train_injector`,
`spike_dilutor`, `inhomogeneous_poisson_generator`, `poisson_generator`,
`poisson_generator_ps`, `sinusoidal_poisson_generator`, `gamma_sup_generator`,
`mip_generator`, `ppd_sup_generator`, `pulsepacket_generator`,
`sinusoidal_gamma_generator`, `correlation_detector`, `correlomatrix_detector`,
`correlospinmatrix_detector`, `multimeter`, `voltmeter`, `spike_recorder`,
`spin_detector`, `volume_transmitter`, `weight_recorder`.

### `_nest_validation/`  ← entire `_nest/_validation/` (incl. `conftest.py`, `nest_compare.py`,
`tolerance_conventions.py`, `_clopath_drive.py`, `_pong_drive.py`, `_stdp_drive.py`,
`_stdp_dopamine_drive.py`, `_urbanczik_drive.py`, `__init__.py`, ~140 `*_test.py`).

### `_nest_network/`  ← entire `_network/` (all source + tests, moved as a unit).

## 3.1. Marker base-class wiring (behavioral addition)

In addition to the move, the currently-unwired synapse/plasticity **spec classes**
are given their NEST marker base, making the synapse family uniform with the
connection specs (`gap_junction` already subclasses `NESTSynapse(Dynamics)`).

Mechanism (mirror `gap_junction`): add the marker as base class and insert
`super().__init__(in_size=1)` as the **first** statement of each `__init__`
(none call `super().__init__` today; specs keep their existing class-level
`__module__ = 'brainpy.state'`). `in_size=1` is the placeholder the connection
specs already use.

- **→ `NESTSynapse` (8):** `static_synapse`, `static_synapse_hom_w`,
  `bernoulli_synapse`, `cont_delay_synapse`, `tsodyks_synapse`,
  `tsodyks_synapse_hom`, `tsodyks2_synapse`, `quantal_stp_synapse`.
- **→ `NESTPlasticity` (13):** `stdp_synapse`, `stdp_triplet_synapse`,
  `stdp_dopamine_synapse`, `stdp_facetshw_synapse_hom`,
  `stdp_nn_pre_centered_synapse`, `stdp_nn_restr_synapse`, `stdp_nn_symm_synapse`,
  `stdp_pl_synapse_hom`, `clopath_synapse`, `urbanczik_synapse`,
  `vogels_sprekeler_synapse`, `ht_synapse`, `jonke_synapse`.
- **Transitive (no base edit):** `stdp_synapse_hom(stdp_synapse)` inherits
  `NESTPlasticity` from its parent — only verify its `super().__init__` chain reaches
  `Dynamics`.
- **Already wired:** the 5 connection specs (`gap_junction`, `diffusion_connection`,
  `sic_connection`, `rate_connection_delayed`, `rate_connection_instantaneous`).

`_base.py` docstrings are updated so `NESTSynapse` enumerates static + connections +
STP, and `NESTPlasticity` is described as STDP + voltage-based learning rules only.
The `update(state, ctx)` rule kernels and all NEST parameter semantics are unchanged —
this is base-class + `__init__`-prologue only.

Risk: making a spec a `Dynamics` means `EventPlasticProj` stores it as a sub-module;
specs hold only frozen plain attrs (no `brainstate.State`), so nothing extra is
collected — but the plastic-substrate tests (`_event_plastic_test.py`, the simulator
plastic/dopamine/voltage-coupled tests) are the regression guard. No spec defines
`__hash__`/`__eq__` or is used as a dict key (verified), so no hashing/identity break.

## 4. Import-rewrite rules

A single model→package lookup table (from §3) drives the rewrite. Forms observed:

| # | Import form (count) | Rewrite |
|---|---|---|
| 1 | intra-`_nest` relative `from ._base` / `._utils` / `._plastic_base` (≈172) | absolute `from brainpy_state._nest_base._base` / `._utils` / `._plastic_base` |
| 2 | intra-`_nest` relative `from .<model> import` | **same category** → keep relative; **cross-category** → absolute `from brainpy_state._nest_<cat>.<model>` |
| 3 | absolute `from brainpy_state._nest.<model>` (≈68) | `from brainpy_state._nest_<cat>.<model>` via table |
| 4 | absolute `from brainpy_state._nest._utils/_base/_plastic_base` | `from brainpy_state._nest_base.<mod>` |
| 5 | absolute `from brainpy_state._nest._validation…` (272) | `from brainpy_state._nest_validation…` |
| 6 | absolute `from brainpy_state._network…` (156) | `from brainpy_state._nest_network…` |
| 7 | relative imports **inside** `_network/` and `_nest/_validation/` (`from ._x`) | unchanged — each tree moves as a unit |
| 8 | `../../brainpy_state/__init__.py`: `from ._nest import (…)` | split into `from ._nest_base/_nest_neuron/_nest_synapse/_nest_plasticity/_nest_device import (…)`; `from . import _network as network` → `from . import _nest_network as network`; `__all__` unchanged |

No relative `.._nest` / `..._validation` cross-package forms exist (verified), so
cross-package rewrites are all absolute-path edits. Sphinx cross-ref strings in
docstrings (`:class:\`~brainpy_state._network._event_proj.EventProjection\``, etc.)
are rewritten by the same prefix rules (#5, #6) so doc xrefs keep resolving.

## 5. Circular-import preservation

`_nest_synapse`/`_nest_plasticity` ↔ `_nest_network` is a real cycle today
(`_nest` ↔ `_network`) that works because `_nest_network/__init__.py` imports
`_event_plastic` (defining `KernelContext`/`EventPlasticProj`) **before**
`_simulator`. Preserve that order and the submodule-level import form
(`from brainpy_state._nest_network._event_plastic import KernelContext`). The
existing import / `jit_compat` / `_namespace` tests are the regression guard.

## 6. Doc & example moves (`git mv`)

| From | To |
|---|---|
| `examples/nest/` | `../../examples/nest_like` |
| `CONTEXT.md` | `develop/NEST_PARITY_LEDGER.md` |
| `DESIGN_delays_and_missing_features.md` | `develop/DESIGN_delays_and_missing_features.md` |
| `docs/nest-status/internal/` | `develop/nest-status/internal/` |

Then update textual references to these paths: docstrings/comments in moved files,
`../../changelog.md`, the ledger's own self-references, and the ASCII file annotations in
`docs/api/nest-base.rst` (`_nest/_base.py` → `_nest_base/_base.py`). The published
`docs/nest-status/index.rst` does **not** reference `internal/`, so the Sphinx build
is unaffected. No test does runtime path assertions on these locations (verified).

## 7. Edge cases

1. **Test-only stems** (`*_wbyrec_test.py`, `cm_default_state_test.py`) have no source
   module — move with their model's category (neuron).
2. **`_plastic_base.py` is shared** by STP (now in `_nest_synapse`) and STDP/voltage
   (`_nest_plasticity`) → it must live in `_nest_base` (it does).
3. **`_nest_network` depends on** `_nest_base`, `_nest_device`, `_nest_neuron`,
   `_nest_synapse`, `_nest_validation` — confirm none of those import `_nest_network`
   at *package-init* top level. Only synapse/plasticity *model* modules import it
   (at module top, via `from brainpy_state._nest_network._event_plastic import …`),
   and the cycle resolves through the §5 init ordering — exactly as today.
4. **`../../pyproject.toml`** uses `packages = ["brainpy_state*"]` (wildcard) — new packages
   are auto-discovered; no packaging edit needed (verify post-move).
5. **`conftest.py`** inside `_validation` moves with it; confirm pytest still collects
   under the new path (no hardcoded `_nest/_validation` in `../../pyproject.toml` testpaths).
6. **`docs/api/nest-*.rst`** use `.. currentmodule:: brainpy.state` + public names, so
   autodoc is unaffected by the internal move except the one ASCII annotation (item §6).

## 8. Verification gates (run before & after)

1. `python -c "import brainpy_state"` — succeeds, no ImportError / circular import.
2. `python -c "import brainpy_state as b; assert set(b.__all__) == EXPECTED"` — public
   surface unchanged (snapshot `__all__` before the refactor, diff after).
3. `pytest brainpy_state/ -m "not requires_nest" --co -q` — **collection succeeds**
   (no import errors) before and after; count of collected tests unchanged.
4. Targeted: `pytest brainpy_state/_nest_base/jit_compat_test.py
   brainpy_state/_nest_spatial/_namespace_test.py -m "not requires_nest"`.
5. `git mv` used for every move — `git log --follow` still traces each file.
6. `git grep -n "brainpy_state\._nest\b" -- '*.py'` and `git grep -n
   "brainpy_state\._network\b"` return **zero** matches after rewrite.
7. **Marker wiring:** `isinstance(tsodyks_synapse(...), NESTSynapse)` is `True` and
   `isinstance(stdp_synapse(...), NESTPlasticity)` is `True`; the plastic-substrate
   tests (`_nest_network/_event_plastic_test.py`, `_simulator_plastic_test.py`,
   `_simulator_dopamine_test.py`, `_voltage_coupled_plastic_test.py`) pass.

## 9. Out of scope

- No change to any model's `update`/rule kernel or NEST parameter semantics. The only
  model-code change is adding the marker base + `super().__init__(in_size=1)` prologue
  to the 21 spec classes in §3.1.
- No change to `_brainpy/`, `_dist`, `_nest_spatial/` contents. (`_base.py` gets only
  the NESTSynapse/NESTPlasticity docstring update from §3.1.)
- No change to public names or `../../examples/nest_like` file contents (only the dir name
  and inbound textual references).
- The `examples` import style (`from brainpy_state import …`) is left as-is.
```
