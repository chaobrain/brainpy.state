# Remove the temporary `exp_euler` dispatch (monkey-patch)

## Context

`brainpy_state/_exp_euler.py` was a **temporary workaround**: it shipped a
unit-corrected copy of `brainstate.nn.exp_euler_step` plus
`install_exp_euler_patch()`, which `../../brainpy_state/__init__.py` invoked at import
time to overwrite the buggy upstream symbol — **but only when a runtime probe
detected the regression** (the "dispatch"). The bug: upstream rebuilt the
linearised Jacobian unit as `unit(drift)/unit(jacobian)`, and since
`vector_grad` strips the Jacobian unit, the result was mislabelled, breaking the
strict `brainunit>=0.4.0` `exprel`.

Upstream `brainstate.nn.exp_euler_step` is now **fixed**.

### Precondition verified (2026-06-16)

```
brainstate 0.5.0 + brainunit 0.4.0
brainstate.nn.exp_euler_step(lambda v: -v/(10*ms), ones*mV) -> unit == mV  ✓
```

So `install_exp_euler_patch()` is already a no-op (its probe returns "not
broken"). The dispatch can be deleted without behavioural change.

### Usage audit

All model code (`_brainpy/*.py`, `_base.py`) and all tests call
`brainstate.nn.exp_euler_step` **directly**. Nothing imports the local
`exp_euler_step` except `__init__.py` (the patch wiring) and
`_exp_euler_test.py` (which tests the workaround itself).

## Plan

1. **`../../brainpy_state/__init__.py`** — delete the patch block: the
   "Temporary upstream compatibility patch" comment, the
   `from ._exp_euler import install_exp_euler_patch`, the
   `install_exp_euler_patch()` call, and `del install_exp_euler_patch`.
2. **Delete `brainpy_state/_exp_euler.py`** — the workaround module.
3. **Delete `brainpy_state/_exp_euler_test.py`** — it tests only the deleted
   workaround (corrected impl + patch). The upstream fix is brainstate's to test.
4. **`../../pyproject.toml`** — bump `brainstate>=0.2.0` → `brainstate>=0.5.0` (the
   verified-fixed floor). With the fallback patch gone, a fresh install must not
   be allowed to pull a brainstate that still carries the bug. (User-approved.)
5. **`CONTEXT.md:1907`** — the dated dev-log note references
   `install_exp_euler_patch()` (a coverage-segfault workaround). Annotate that
   the patch has since been removed so a future reader grepping the symbol isn't
   confused. (User-approved.)

## Verification

- `import brainpy_state` succeeds with no patch wiring.
- `grep -rn "exp_euler_patch\|_exp_euler"` returns nothing in package source.
- A unit-bearing model still integrates: instantiate an `LIF` (or run an
  `exp_euler_step` on a `mV` ODE) and confirm the result carries `mV`.
- Run `_base_test.py` + a representative `_brainpy` model test to confirm
  nothing depended on the patch.

## Edge cases considered

- Editable install resolution: ensure the worktree's `brainpy_state` is the one
  imported when verifying (run from the worktree root).
- Idempotency / repeated-import behaviour of the patch is irrelevant once
  deleted; no other module references `_PATCH_APPLIED`.
- No docs/rst/ipynb reference the patch (swept).

## Git workflow (after verification)

Commit → push → PR → squash-merge → delete remote branch → merge origin/main
into local main → remove worktree.
