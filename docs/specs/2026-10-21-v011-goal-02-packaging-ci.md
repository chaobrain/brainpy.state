# v0.1.1 goal 02 — packaging, CI, deprecated APIs, test hygiene

_Date: 2026-10-21_

Part of [roadmap](2026-10-21-roadmap.md) Phase 0. Branch `v011/02-packaging-ci`.

## Dependencies (`pyproject.toml:39-46`, `requirements.txt`)
- Raise floors to what is actually tested: `brainstate>=0.5.1`, `braintools>=0.3.0`
  (brainpy 2.8.2 already requires both), add a `brainunit` floor (code uses
  `split_mantissa_unit`, `u.lax.sort`), `numpy>=1.26`. Pick the jax floor from a
  minimum-versions CI job, not by guess.
- Fix stale comments: mypy comment says "supports 3.11+"; pytest-marker comment points
  at `_nest/_validation/nest_compare.py` (also `nest_compare.py:18`,
  `_nest_validation/README.md:82-91`).

## CI (`.github/workflows/`)
- Python matrix: test 3.12 (floor) and newest, not only 3.13. Add one minimum-deps job.
- Add `pytest-cov` and `pytest-timeout` to `requirements-dev.txt`; upload coverage.
- Build docs on PRs (today only on release, `deploy-docs.yml`).
- `Publish.yml`: choose trusted publishing **or** `PYPI_API_TOKEN`, not both.
- `labeler.yml` matches non-existent paths (`docs_version2_version2/**`,
  `docs_classic/**`, `brainpy_state/_src/**`, `tests/**`); `dependabot.yml` has an
  invalid reviewer `brainpy_state/maintainers`.
- `.gitignore:17,239` refer to `examples/nest/*.png` (now `nest_like`) and
  `/docs_state/changelog.md`.

## Tests never run by the NEST workflow
`_nest_validation/iaf_psc_alpha_parity_test.py:60` and
`brunel_alpha_evolution_strategies_test.py:113` use `unittest.skipUnless(_HAS_NEST)`
instead of `@requires_nest`, so `-m requires_nest` deselects them.

## Test import hygiene (CLAUDE.md #9)
306 test files import through `brainpy.state` instead of `brainpy_state`. Under pytest it
currently resolves to the local source (verified), but it re-enters `brainpy`'s import
machinery and can pick up an installed copy elsewhere. Rewrite mechanically to
`from brainpy_state import ...`; add a guard test that greps for the pattern.
Dev setup note: a stale non-editable `brainpy_state` 0.1.0 in site-packages shadows the
repo outside pytest — CONTRIBUTING should say `pip install -e .`.

## Deprecated / fragile APIs
- `_brainpy/delay.py:93-94` passes deprecated `interp_method='linear_interp'` (dead —
  `update()` interpolates itself).
- `_nest_base/utils.py:34,49` imports `DynamicJaxprTracer` from
  `jax.interpreters.partial_eval`; use `jax.core.Tracer` (`ShapedArray` is not a tracer).
- `_mixin.py:50,68,123,147` docstring examples subclass deprecated
  `brainstate.mixin.AlignPost/BindCondData`.
- Cross-repo (file upstream): `brainstate.nn.__getattr__` still redirects
  `LeakySpikeReadout` to `brainpy.state.LeakySpikeReadout`.

## Verification
CI green on the new matrix; coverage report produced; `pytest -m requires_nest
--collect-only` lists the two fixed files.
