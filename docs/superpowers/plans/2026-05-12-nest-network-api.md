# NEST Network API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `brainpy.state.Network` (declarative subclass, canonical) and `brainpy.state.Builder` (imperative subclass) over a single `brainstate.nn.Module` tree, with eight rule-based `Projection` subclasses, a `Recorder` helper, and a Brunel flagship example — per `docs/superpowers/specs/2026-05-12-nest-network-api-design.md`.

**Architecture:** A new `brainpy_state/_network/` package. `Network` adds `update()` (one timestep, projection-first traversal of the module tree) and `simulate(duration, monitor=...)` (JIT-wrapped `brainstate.transform.for_loop`) on top of `brainstate.nn.Module`. Each rule-based `*Proj` is a thin `Projection` subclass that stores `pre`/`post`, builds a `brainstate.nn` connectivity module (`Linear`, `FixedNumConn`, `FixedProb`), composes it through the existing `AlignPostProj` machinery, and overrides `update()` to feed `pre.spike.value` forward. A small `brainpy_state/_dist.py` module supplies `Normal`/`LogNormal`/`Uniform` distribution objects sampled once at projection `__init__`.

**Tech Stack:** Python 3.11+, JAX, brainstate ≥ 0.3, brainevent, saiunit, pytest + unittest.

---

## File map

Files to create:

| Path | Responsibility |
|---|---|
| `brainpy_state/_dist.py` | `Distribution` ABC + `Normal`, `LogNormal`, `Uniform`. Tiny module. |
| `brainpy_state/_dist_test.py` | Tests for the three distribution classes. |
| `brainpy_state/_network/__init__.py` | Re-export public classes. |
| `brainpy_state/_network/_base.py` | `Network` class. |
| `brainpy_state/_network/_builder.py` | `Builder` class. |
| `brainpy_state/_network/_connectivity.py` | Internal helpers that sample indices/masks per rule and resolve scalar/array/Distribution → arrays. |
| `brainpy_state/_network/_projections.py` | `_RuleProj` base + the eight `*Proj` subclasses. |
| `brainpy_state/_network/_recorders.py` | `Recorder` helper for wiring `NESTDevice` recorders. |
| `brainpy_state/_network/_base_test.py` | Tests for `Network`. |
| `brainpy_state/_network/_builder_test.py` | Tests for `Builder`. |
| `brainpy_state/_network/_connectivity_test.py` | Tests for connectivity samplers. |
| `brainpy_state/_network/_projections_test.py` | Per-rule unit tests for `*Proj` classes. |
| `brainpy_state/_network/_recorders_test.py` | Tests for `Recorder`. |
| `brainpy_state/_network/_brunel_test.py` | Brunel integration test. |
| `examples/brunel.py` | Flagship Brunel example using `Builder`. |

Files to modify:

| Path | What changes |
|---|---|
| `brainpy_state/__init__.py` | Add imports + `__all__` entries for `Network`, `Builder`, `Recorder`, eight `*Proj` classes, and the `dist` submodule. |
| `CHANGELOG.md` | Add unreleased section with the new API. |

---

## Test conventions

- All test classes inherit from `unittest.TestCase` and are run by `pytest`.
- Every `setUp` includes `brainstate.environ.set(dt=0.1 * u.ms)` unless the test overrides `dt`.
- Random seeds passed explicitly in tests so all assertions are deterministic.
- Tests live next to source as `*_test.py` per the project convention in `.claude/CLAUDE.md`.

Run all new tests as you go:
```
pytest brainpy_state/_network/ brainpy_state/_dist_test.py -v
```

---

## Task 1: Network skeleton

**Files:**
- Create: `brainpy_state/_network/__init__.py`
- Create: `brainpy_state/_network/_base.py`
- Create: `brainpy_state/_network/_base_test.py`

- [ ] **Step 1.1: Write the failing test**

`brainpy_state/_network/_base_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest
import brainstate
import saiunit as u

from brainpy_state._network._base import Network
from brainpy_state import LIF


class TestNetworkSkeleton(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_network_is_a_brainstate_module(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(10)

        net = Net()
        self.assertIsInstance(net, brainstate.nn.Module)
        self.assertIs(net.pop, net.nodes()['pop'])

    def test_module_attribute_is_brainpy_state(self):
        self.assertEqual(Network.__module__, 'brainpy.state')
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest brainpy_state/_network/_base_test.py -v`
Expected: `ModuleNotFoundError: No module named 'brainpy_state._network'` (or `ImportError`).

- [ ] **Step 1.3: Create the package**

`brainpy_state/_network/__init__.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
from ._base import Network

__all__ = ['Network']
```

`brainpy_state/_network/_base.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import brainstate

__all__ = ['Network']


class Network(brainstate.nn.Module):
    """brainpy.state network base class.

    Subclass and define populations, projections, and devices as
    attributes. See ``brainpy.state.Builder`` for an imperative
    variant of the same underlying object.
    """
    __module__ = 'brainpy.state'
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `pytest brainpy_state/_network/_base_test.py -v`
Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```
git add brainpy_state/_network/__init__.py brainpy_state/_network/_base.py brainpy_state/_network/_base_test.py
git commit -m "feat(network): add Network base class skeleton"
```

---

## Task 2: Network.update() walks projections before dynamics

**Files:**
- Modify: `brainpy_state/_network/_base.py`
- Modify: `brainpy_state/_network/_base_test.py`

The traversal order matches the existing convention documented at `brainpy_state/_brainpy/projection.py:46-51`: projections first, then dynamics.

- [ ] **Step 2.1: Write failing tests**

Append to `brainpy_state/_network/_base_test.py`:
```python
from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import Projection


class _TraceNode(Dynamics):
    def __init__(self, calls, tag):
        super().__init__(in_size=1)
        self._calls = calls
        self._tag = tag

    def init_state(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        self._calls.append(self._tag)


class _TraceProj(Projection):
    def __init__(self, calls, tag):
        super().__init__()
        self._calls = calls
        self._tag = tag

    def update(self, *args, **kwargs):
        self._calls.append(self._tag)


class TestNetworkUpdateOrder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_projections_run_before_dynamics(self):
        calls = []

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.neuron = _TraceNode(calls, 'neuron')
                self.proj = _TraceProj(calls, 'proj')

        net = Net()
        net.update()
        self.assertEqual(calls, ['proj', 'neuron'])

    def test_introspection_properties(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.neuron = _TraceNode([], 'neuron')
                self.proj = _TraceProj([], 'proj')

        net = Net()
        self.assertIn('neuron', net.populations)
        self.assertIn('proj', net.projections)
        self.assertEqual(net.devices, {})
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest brainpy_state/_network/_base_test.py -v`
Expected: `TestNetworkUpdateOrder` fails — `Network` has no `update` / `populations` / `projections` / `devices`.

- [ ] **Step 2.3: Implement update() and introspection**

Overwrite `brainpy_state/_network/_base.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
from typing import Dict

import brainstate

from brainpy_state._base import Neuron
from brainpy_state._brainpy.projection import Projection
from brainpy_state._nest._base import NESTDevice

__all__ = ['Network']


class Network(brainstate.nn.Module):
    """brainpy.state network base class.

    Subclass and define populations, projections, and devices as
    attributes. ``update()`` walks the immediate module-tree children
    in projection-first then dynamics order.
    """
    __module__ = 'brainpy.state'

    def update(self, t=None) -> None:
        # Depth-1 traversal — matches the existing brainpy.state convention
        # documented in brainpy_state/_brainpy/projection.py:46-51. Networks
        # that need nested projection chains (Projection containing
        # Projection) currently must override update() explicitly; this is
        # tracked as an open question in the design spec §12.
        children = self.nodes(allowed_hierarchy=(1, 1))
        projections = [m for m in children.values() if isinstance(m, Projection)]
        others = [m for m in children.values() if not isinstance(m, Projection)]
        for m in projections:
            m()
        for m in others:
            m()

    @property
    def populations(self) -> Dict[str, Neuron]:
        return {n: m for n, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Neuron)}

    @property
    def projections(self) -> Dict[str, Projection]:
        return {n: m for n, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Projection)}

    @property
    def devices(self) -> Dict[str, NESTDevice]:
        return {n: m for n, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, NESTDevice)}
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest brainpy_state/_network/_base_test.py -v`
Expected: 4 passed.

- [ ] **Step 2.5: Commit**

```
git add brainpy_state/_network/_base.py brainpy_state/_network/_base_test.py
git commit -m "feat(network): Network.update walks projections before dynamics"
```

---

## Task 3: Distribution objects

**Files:**
- Create: `brainpy_state/_dist.py`
- Create: `brainpy_state/_dist_test.py`

Distributions are sampled once at projection `__init__` (per spec §7). They are pure dataclasses with a `sample(shape, key) -> Array` method. No lazy / runtime semantics.

- [ ] **Step 3.1: Write the failing tests**

`brainpy_state/_dist_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._dist import Distribution, Normal, LogNormal, Uniform


class TestDistributions(unittest.TestCase):
    def test_normal_shape_and_seed_determinism(self):
        d = Normal(mean=0.0, std=1.0)
        key = jax.random.key(0)
        a = d.sample((100,), key)
        b = d.sample((100,), key)
        self.assertEqual(a.shape, (100,))
        self.assertTrue(jnp.allclose(a, b))

    def test_normal_carries_units(self):
        d = Normal(mean=0.1 * u.nS, std=0.01 * u.nS)
        key = jax.random.key(1)
        x = d.sample((10,), key)
        self.assertTrue(u.Quantity(x).unit.has_same_dim(u.nS))

    def test_uniform_bounds(self):
        d = Uniform(low=-1.0, high=2.0)
        key = jax.random.key(2)
        x = d.sample((1000,), key)
        self.assertGreaterEqual(float(jnp.min(x)), -1.0 - 1e-6)
        self.assertLessEqual(float(jnp.max(x)), 2.0 + 1e-6)

    def test_lognormal_positive(self):
        d = LogNormal(mean=0.0, std=1.0)
        key = jax.random.key(3)
        x = d.sample((100,), key)
        self.assertTrue(bool(jnp.all(x > 0)))

    def test_is_distribution(self):
        for cls in [Normal, LogNormal, Uniform]:
            self.assertTrue(issubclass(cls, Distribution))
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest brainpy_state/_dist_test.py -v`
Expected: `ModuleNotFoundError: No module named 'brainpy_state._dist'`.

- [ ] **Step 3.3: Implement**

`brainpy_state/_dist.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distribution objects sampled once at projection init.

This is the brainpy.state-native parameter-randomization API. It
deliberately differs from NEST's lazy ``Parameter`` (which evaluates
at ``Connect`` time): our distributions are sampled eagerly during
projection construction, so the JIT trace sees concrete arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import saiunit as u

__all__ = ['Distribution', 'Normal', 'LogNormal', 'Uniform']


class Distribution:
    """Abstract base. Subclasses implement ``sample(shape, key)``."""
    __module__ = 'brainpy.state.dist'

    def sample(self, shape, key):  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Normal(Distribution):
    mean: float | u.Quantity
    std: float | u.Quantity

    __module__ = 'brainpy.state.dist'

    def sample(self, shape, key):
        mean_val, mean_unit = u.split_mantissa_unit(self.mean)
        std_val, std_unit = u.split_mantissa_unit(self.std)
        if mean_unit != std_unit:
            raise ValueError(
                f'mean and std must share units, got {mean_unit} and {std_unit}'
            )
        samples = mean_val + std_val * jax.random.normal(key, shape)
        return u.maybe_decimal(u.Quantity(samples, unit=mean_unit))


@dataclass
class LogNormal(Distribution):
    mean: float
    std: float

    __module__ = 'brainpy.state.dist'

    def sample(self, shape, key):
        return jnp.exp(self.mean + self.std * jax.random.normal(key, shape))


@dataclass
class Uniform(Distribution):
    low: float | u.Quantity
    high: float | u.Quantity

    __module__ = 'brainpy.state.dist'

    def sample(self, shape, key):
        low_val, low_unit = u.split_mantissa_unit(self.low)
        high_val, high_unit = u.split_mantissa_unit(self.high)
        if low_unit != high_unit:
            raise ValueError(
                f'low and high must share units, got {low_unit} and {high_unit}'
            )
        u01 = jax.random.uniform(key, shape)
        samples = low_val + (high_val - low_val) * u01
        return u.maybe_decimal(u.Quantity(samples, unit=low_unit))
```

If `u.split_mantissa_unit` / `u.maybe_decimal` aren't available in the installed `saiunit` version, replace with `u.get_mantissa` / `u.get_unit` and `u.Quantity(...)` constructors; verify against an existing repo file that splits units (e.g. `brainpy_state/_nest/aeif_cond_alpha.py`).

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `pytest brainpy_state/_dist_test.py -v`
Expected: 5 passed.

- [ ] **Step 3.5: Commit**

```
git add brainpy_state/_dist.py brainpy_state/_dist_test.py
git commit -m "feat(dist): add Normal/LogNormal/Uniform distributions"
```

---

## Task 4: Connectivity samplers and parameter resolver

**Files:**
- Create: `brainpy_state/_network/_connectivity.py`
- Create: `brainpy_state/_network/_connectivity_test.py`

Each rule produces a `ConnSpec` containing `(pre_idx, post_idx, n_edges)` arrays: index lists for the source and target sides of each edge. The Proj classes use this to build a dense weight matrix initially (shape `(n_pre, n_post)`) — sparse swap-in is a follow-up optimization not in this plan.

Also includes `resolve_param(value, shape, key, unit_hint)` that turns scalar / array / `Distribution` into a concrete array of the requested shape.

- [ ] **Step 4.1: Write failing tests**

`brainpy_state/_network/_connectivity_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u

from brainpy_state._dist import Normal
from brainpy_state._network._connectivity import (
    sample_one_to_one,
    sample_all_to_all,
    sample_pairwise_bernoulli,
    sample_fixed_indegree,
    sample_fixed_outdegree,
    sample_fixed_total_number,
    sample_pairwise_poisson,
    resolve_param,
)


class TestConnectivitySamplers(unittest.TestCase):
    def test_one_to_one_requires_equal_sizes(self):
        with self.assertRaises(ValueError):
            sample_one_to_one(5, 4)

    def test_one_to_one_edges(self):
        spec = sample_one_to_one(4, 4)
        np.testing.assert_array_equal(spec.pre_idx, [0, 1, 2, 3])
        np.testing.assert_array_equal(spec.post_idx, [0, 1, 2, 3])

    def test_all_to_all_with_autapses(self):
        spec = sample_all_to_all(3, 3, pre_is_post=True, allow_autapses=True)
        self.assertEqual(spec.n_edges, 9)

    def test_all_to_all_without_autapses(self):
        spec = sample_all_to_all(3, 3, pre_is_post=True, allow_autapses=False)
        self.assertEqual(spec.n_edges, 6)
        pairs = set(zip(spec.pre_idx.tolist(), spec.post_idx.tolist()))
        for i in range(3):
            self.assertNotIn((i, i), pairs)

    def test_pairwise_bernoulli_density(self):
        key = jax.random.key(0)
        spec = sample_pairwise_bernoulli(
            100, 100, p=0.1, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=True,
        )
        density = spec.n_edges / (100 * 100)
        self.assertAlmostEqual(density, 0.1, delta=0.02)

    def test_fixed_indegree_each_post_has_K(self):
        key = jax.random.key(1)
        spec = sample_fixed_indegree(
            n_pre=50, n_post=20, K=10, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=False,
        )
        for j in range(20):
            self.assertEqual(int(jnp.sum(spec.post_idx == j)), 10)

    def test_fixed_outdegree_each_pre_has_K(self):
        key = jax.random.key(2)
        spec = sample_fixed_outdegree(
            n_pre=20, n_post=50, K=10, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=False,
        )
        for i in range(20):
            self.assertEqual(int(jnp.sum(spec.pre_idx == i)), 10)

    def test_fixed_total_number(self):
        key = jax.random.key(3)
        spec = sample_fixed_total_number(
            n_pre=50, n_post=50, N=137, key=key,
            pre_is_post=False, allow_autapses=True, allow_multapses=True,
        )
        self.assertEqual(spec.n_edges, 137)

    def test_pairwise_poisson_mean(self):
        key = jax.random.key(4)
        spec = sample_pairwise_poisson(
            n_pre=100, n_post=100, mean=0.05, key=key,
            pre_is_post=False, allow_autapses=True,
        )
        expected = 100 * 100 * 0.05
        self.assertAlmostEqual(spec.n_edges, expected, delta=0.1 * expected)


class TestResolveParam(unittest.TestCase):
    def test_scalar_broadcast(self):
        key = jax.random.key(0)
        out = resolve_param(0.5, (10,), key)
        self.assertEqual(out.shape, (10,))
        self.assertTrue(bool(jnp.all(out == 0.5)))

    def test_array_passthrough(self):
        key = jax.random.key(0)
        arr = jnp.arange(10.0)
        out = resolve_param(arr, (10,), key)
        np.testing.assert_array_equal(out, arr)

    def test_distribution_sampled(self):
        key = jax.random.key(0)
        out = resolve_param(Normal(mean=0.0, std=1.0), (1000,), key)
        self.assertEqual(out.shape, (1000,))
        self.assertAlmostEqual(float(jnp.mean(out)), 0.0, delta=0.1)

    def test_array_shape_mismatch_raises(self):
        key = jax.random.key(0)
        with self.assertRaises(ValueError):
            resolve_param(jnp.zeros(7), (10,), key)
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `pytest brainpy_state/_network/_connectivity_test.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 4.3: Implement**

`brainpy_state/_network/_connectivity.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Internal connectivity samplers and parameter resolver.

Each sampler returns a ``ConnSpec`` of (pre_idx, post_idx) int arrays
plus n_edges. Projection classes turn these into a dense
``(n_pre, n_post)`` weight matrix.

These helpers are private to ``brainpy_state._network``. Public APIs
are the ``*Proj`` classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._dist import Distribution


@dataclass
class ConnSpec:
    pre_idx: jnp.ndarray
    post_idx: jnp.ndarray
    n_edges: int


def sample_one_to_one(n_pre: int, n_post: int) -> ConnSpec:
    if n_pre != n_post:
        raise ValueError(
            f'one_to_one requires equal sizes, got n_pre={n_pre}, n_post={n_post}'
        )
    idx = jnp.arange(n_pre)
    return ConnSpec(idx, idx, int(n_pre))


def sample_all_to_all(
    n_pre: int,
    n_post: int,
    *,
    pre_is_post: bool,
    allow_autapses: bool,
) -> ConnSpec:
    pre = jnp.repeat(jnp.arange(n_pre), n_post)
    post = jnp.tile(jnp.arange(n_post), n_pre)
    if pre_is_post and not allow_autapses:
        mask = pre != post
        pre = pre[mask]
        post = post[mask]
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_pairwise_bernoulli(
    n_pre: int,
    n_post: int,
    *,
    p: float,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    # Multapses not meaningful for Bernoulli (single trial per pair) — flag
    # exists for API symmetry. allow_multapses is ignored here.
    del allow_multapses
    mask = jax.random.uniform(key, (n_pre, n_post)) < p
    if pre_is_post and not allow_autapses:
        mask = mask & (1 - jnp.eye(n_pre, n_post, dtype=jnp.int32)).astype(bool)
    pre, post = jnp.where(mask)
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_fixed_indegree(
    n_pre: int,
    n_post: int,
    *,
    K: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if K < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    pre_lists = []
    post_lists = []
    for j in range(n_post):
        sub = jax.random.fold_in(key, j)
        candidates = jnp.arange(n_pre)
        if pre_is_post and not allow_autapses:
            candidates = candidates[candidates != j]
        if allow_multapses:
            chosen = jax.random.choice(sub, candidates, (K,), replace=True)
        else:
            if K > candidates.shape[0]:
                raise ValueError(
                    f'cannot pick {K} unique pre for post {j} from '
                    f'{candidates.shape[0]} candidates with allow_multapses=False'
                )
            chosen = jax.random.choice(sub, candidates, (K,), replace=False)
        pre_lists.append(chosen)
        post_lists.append(jnp.full((K,), j))
    return ConnSpec(jnp.concatenate(pre_lists), jnp.concatenate(post_lists), int(K * n_post))


def sample_fixed_outdegree(
    n_pre: int,
    n_post: int,
    *,
    K: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if K < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    pre_lists = []
    post_lists = []
    for i in range(n_pre):
        sub = jax.random.fold_in(key, i)
        candidates = jnp.arange(n_post)
        if pre_is_post and not allow_autapses:
            candidates = candidates[candidates != i]
        if allow_multapses:
            chosen = jax.random.choice(sub, candidates, (K,), replace=True)
        else:
            if K > candidates.shape[0]:
                raise ValueError(
                    f'cannot pick {K} unique post for pre {i} from '
                    f'{candidates.shape[0]} candidates with allow_multapses=False'
                )
            chosen = jax.random.choice(sub, candidates, (K,), replace=False)
        pre_lists.append(jnp.full((K,), i))
        post_lists.append(chosen)
    return ConnSpec(jnp.concatenate(pre_lists), jnp.concatenate(post_lists), int(K * n_pre))


def sample_fixed_total_number(
    n_pre: int,
    n_post: int,
    *,
    N: int,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
    allow_multapses: bool,
) -> ConnSpec:
    if N < 0:
        raise ValueError(f'N must be >= 0, got {N}')
    k1, k2 = jax.random.split(key)
    pre = jax.random.randint(k1, (N,), 0, n_pre)
    post = jax.random.randint(k2, (N,), 0, n_post)
    if pre_is_post and not allow_autapses:
        # Resample autapses one extra round; if any remain, raise.
        mask = pre == post
        if bool(jnp.any(mask)):
            k3 = jax.random.fold_in(key, 1)
            replacement = jax.random.randint(k3, (int(jnp.sum(mask)),), 0, n_post)
            post = post.at[mask].set(replacement)
            if bool(jnp.any(pre == post)):
                raise ValueError(
                    'failed to remove autapses in fixed_total_number with one resample pass'
                )
    if not allow_multapses:
        # Deduplicate pairs; document that this can produce fewer than N edges
        # — NEST documents the same behaviour.
        pairs = jnp.stack([pre, post], axis=1)
        _, unique_idx = jnp.unique(pairs, axis=0, return_index=True)
        pre = pre[unique_idx]
        post = post[unique_idx]
    return ConnSpec(pre, post, int(pre.shape[0]))


def sample_pairwise_poisson(
    n_pre: int,
    n_post: int,
    *,
    mean: float,
    key,
    pre_is_post: bool,
    allow_autapses: bool,
) -> ConnSpec:
    counts = jax.random.poisson(key, mean, (n_pre, n_post))
    if pre_is_post and not allow_autapses:
        counts = counts.at[jnp.arange(n_pre), jnp.arange(min(n_pre, n_post))].set(0)
    pre = jnp.repeat(jnp.arange(n_pre)[:, None], n_post, axis=1).reshape(-1)
    post = jnp.tile(jnp.arange(n_post), n_pre)
    counts_flat = counts.reshape(-1)
    repeats = counts_flat.astype(jnp.int32)
    pre = jnp.repeat(pre, repeats, total_repeat_length=int(jnp.sum(repeats)))
    post = jnp.repeat(post, repeats, total_repeat_length=int(jnp.sum(repeats)))
    return ConnSpec(pre, post, int(pre.shape[0]))


def resolve_param(value, shape, key):
    """Turn scalar | array | Distribution into a concrete array of ``shape``."""
    if isinstance(value, Distribution):
        return value.sample(shape, key)
    if hasattr(value, 'shape') and value.shape != () and value.shape != (1,):
        if tuple(value.shape) != tuple(shape):
            raise ValueError(
                f'parameter array shape {value.shape} does not match target {shape}'
            )
        return value
    return jnp.broadcast_to(jnp.asarray(value), shape) if not isinstance(value, u.Quantity) \
        else u.Quantity(jnp.broadcast_to(value.mantissa, shape), unit=value.unit)
```

Note: `sample_fixed_indegree` and `sample_fixed_outdegree` use a Python `for` loop over post/pre. This is acceptable for `__init__`-time edge construction (runs once, not in the hot path) and avoids vmap-pytree complexity. If population sizes are very large (>10⁵) the engineer may revisit.

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `pytest brainpy_state/_network/_connectivity_test.py -v`
Expected: 12 passed.

- [ ] **Step 4.5: Commit**

```
git add brainpy_state/_network/_connectivity.py brainpy_state/_network/_connectivity_test.py
git commit -m "feat(network): add connectivity samplers and resolve_param"
```

---

## Task 5: `_RuleProj` base class

**Files:**
- Create: `brainpy_state/_network/_projections.py`
- Create: `brainpy_state/_network/_projections_test.py`

`_RuleProj` is the shared base for all eight rule classes. It stores `pre`/`post`, builds a dense weight matrix from a `ConnSpec`, wraps it in a `brainstate.nn.Linear` comm module, composes it through `AlignPostProj`, and overrides `update()` to feed `pre.spike.value` (or `pre()`) into the inner projection.

Delays are deferred to a follow-up — `_RuleProj` accepts `delay=` and stores it on `self.delay` but does not yet wire `brainstate.nn.Delay`. A `NotImplementedError` is raised for non-`None` delay; later tasks can add the wrapper once a real example demands it. (Brunel example uses scalar delay — Task 19 enables it then.)

Actually, let's enable scalar delay from the start so Brunel works. Per-edge delay deferred.

- [ ] **Step 5.1: Write failing tests**

`brainpy_state/_network/_projections_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state import LIF, Expon, COBA
from brainpy_state._network._projections import _RuleProj, OneToOneProj


class TestRuleProjBase(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_stores_pre_and_post(self):
        pre = LIF(5)
        post = LIF(5)
        proj = OneToOneProj(
            pre, post,
            weight=0.1 * u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, pre)
        self.assertIs(proj.post, post)

    def test_per_edge_weight_shape(self):
        pre = LIF(5)
        post = LIF(5)
        proj = OneToOneProj(
            pre, post,
            weight=0.1 * u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        w = proj._weight_matrix.value  # (n_pre, n_post)
        self.assertEqual(w.shape, (5, 5))
        # one-to-one: only diagonal has weight
        diag = jnp.diag(u.get_mantissa(w))
        off = u.get_mantissa(w) - jnp.diag(diag)
        self.assertTrue(bool(jnp.all(jnp.abs(off) < 1e-9)))

    def test_seed_determinism(self):
        # Determinism is delegated to rule classes that use randomness.
        # OneToOne has no randomness, so two builds with same args match.
        pre1, post1 = LIF(5), LIF(5)
        pre2, post2 = LIF(5), LIF(5)
        p1 = OneToOneProj(pre1, post1, weight=0.1*u.nS,
                          syn=Expon.desc(5, tau=5*u.ms),
                          out=COBA.desc(E=0*u.mV))
        p2 = OneToOneProj(pre2, post2, weight=0.1*u.nS,
                          syn=Expon.desc(5, tau=5*u.ms),
                          out=COBA.desc(E=0*u.mV))
        self.assertTrue(jnp.allclose(
            u.get_mantissa(p1._weight_matrix.value),
            u.get_mantissa(p2._weight_matrix.value)))
```

- [ ] **Step 5.2: Run to verify they fail**

Run: `pytest brainpy_state/_network/_projections_test.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement `_RuleProj`**

`brainpy_state/_network/_projections.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Rule-based projection classes for the Network API."""
from __future__ import annotations

from typing import Optional, Union

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import AlignPostProj
from brainpy_state._network._connectivity import (
    ConnSpec,
    sample_one_to_one,
    sample_all_to_all,
    sample_pairwise_bernoulli,
    sample_fixed_indegree,
    sample_fixed_outdegree,
    sample_fixed_total_number,
    sample_pairwise_poisson,
    resolve_param,
)

__all__ = [
    'OneToOneProj', 'AllToAllProj',
    'PairwiseBernoulliProj', 'SymmetricPairwiseBernoulliProj',
    'FixedIndegreeProj', 'FixedOutdegreeProj',
    'FixedTotalNumberProj', 'PairwisePoissonProj',
]


def _pre_output(pre: Dynamics):
    """Pull the per-step output from a pre-synaptic module.

    Convention: spiking populations expose a ``spike`` State; rate
    populations expose ``r`` or a callable. We default to ``.spike``
    when present, else call the module.
    """
    if hasattr(pre, 'spike'):
        return pre.spike.value
    return pre()


class _RuleProj(brainstate.nn.Module):
    """Shared base for rule-based projections.

    Subclasses override ``_build_conn_spec(self, key) -> ConnSpec``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        pre: Dynamics,
        post: Dynamics,
        *,
        weight,
        delay=None,
        syn,
        out,
        allow_autapses: bool = True,
        allow_multapses: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(pre, Dynamics):
            raise TypeError(f'pre must be a Dynamics instance, got {type(pre).__name__}')
        if not isinstance(post, Dynamics):
            raise TypeError(f'post must be a Dynamics instance, got {type(post).__name__}')

        self.pre = pre
        self.post = post
        self.allow_autapses = allow_autapses
        self.allow_multapses = allow_multapses
        self._pre_is_post = pre is post

        # 1. Sample connectivity
        key = jax.random.key(0 if seed is None else int(seed))
        k_conn, k_w, k_d = jax.random.split(key, 3)
        spec = self._build_conn_spec(k_conn)

        # 2. Per-edge weight, then scatter into a dense (n_pre, n_post) matrix
        n_pre = self._size(pre)
        n_post = self._size(post)
        w_per_edge = resolve_param(weight, (spec.n_edges,), k_w)
        w_mantissa, w_unit = u.split_mantissa_unit(w_per_edge) if isinstance(w_per_edge, u.Quantity) \
            else (jnp.asarray(w_per_edge), u.UNITLESS)
        W = jnp.zeros((n_pre, n_post), dtype=w_mantissa.dtype)
        W = W.at[spec.pre_idx, spec.post_idx].add(w_mantissa)
        W_with_unit = u.Quantity(W, unit=w_unit) if w_unit is not u.UNITLESS else W
        self._weight_matrix = brainstate.ParamState(W_with_unit)

        # 3. Delay (scalar only for v1)
        self.delay = delay
        if delay is not None:
            if isinstance(delay, (list, tuple, jnp.ndarray)) or (
                isinstance(delay, u.Quantity) and delay.mantissa.ndim > 0
            ):
                raise NotImplementedError(
                    'per-edge delay is not implemented in v1; pass a scalar delay'
                )

        # 4. Build comm and inner AlignPostProj
        comm = _DenseMatMul(self._weight_matrix, delay=delay, pre=pre)
        self._inner = AlignPostProj(comm=comm, syn=syn, out=out, post=post)

    def _build_conn_spec(self, key) -> ConnSpec:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _size(module: Dynamics) -> int:
        sz = module.in_size if hasattr(module, 'in_size') else module.varshape
        if isinstance(sz, tuple):
            n = 1
            for s in sz:
                n *= int(s)
            return n
        return int(sz)

    def update(self, *args, **kwargs):
        x = _pre_output(self.pre)
        self._inner(x)


class _DenseMatMul(brainstate.nn.Module):
    """Tiny comm module: input @ W, optionally through a Delay."""
    __module__ = 'brainpy.state'

    def __init__(self, weight_state: brainstate.ParamState, delay, pre):
        super().__init__()
        self._W = weight_state
        if delay is None:
            self._delay = None
        else:
            self._delay = brainstate.nn.Delay(jnp.zeros(_RuleProj._size(pre)), delay)

    def update(self, *args, **kwargs):  # called as a Module too
        return self(*args, **kwargs)

    def __call__(self, x):
        if self._delay is not None:
            self._delay.update(x)
            x = self._delay.retrieve_at_step(-1)
        W = self._W.value
        return x @ W if not isinstance(W, u.Quantity) else u.Quantity(
            x @ W.mantissa, unit=W.unit)
```

Notes for the engineer:
- `brainstate.nn.Delay`'s exact API may differ from `retrieve_at_step(-1)` — confirm against `brainstate` source (`python -c "import brainstate; help(brainstate.nn.Delay)"`). If the API differs, adjust `_DenseMatMul.__call__` to use the correct retrieval call. The behavior contract is: delayed by `delay` simulation time, with the current step's input pushed in.
- `_size` falls back to `in_size` then `varshape`; if a model uses a different attribute, prefer `in_size` first since it's the brainstate convention.

- [ ] **Step 5.4: Add `OneToOneProj` (needed for the test)**

Append to `brainpy_state/_network/_projections.py`:
```python
class OneToOneProj(_RuleProj):
    r"""One-to-one connection: edge ``(i, i)`` for ``i = 0..N-1``.

    Requires ``len(pre) == len(post)``.
    """
    __module__ = 'brainpy.state'

    def _build_conn_spec(self, key):
        del key
        return sample_one_to_one(self._size(self.pre), self._size(self.post))
```

- [ ] **Step 5.5: Run tests to verify they pass**

Run: `pytest brainpy_state/_network/_projections_test.py -v`
Expected: 3 passed.

- [ ] **Step 5.6: Commit**

```
git add brainpy_state/_network/_projections.py brainpy_state/_network/_projections_test.py
git commit -m "feat(network): add _RuleProj base and OneToOneProj"
```

---

## Task 6: `AllToAllProj`

**Files:**
- Modify: `brainpy_state/_network/_projections.py`
- Modify: `brainpy_state/_network/_projections_test.py`

- [ ] **Step 6.1: Write failing test**

Append to `brainpy_state/_network/_projections_test.py`:
```python
from brainpy_state._network._projections import AllToAllProj


class TestAllToAllProj(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_dense_weight_with_autapses(self):
        pre = LIF(4); post = LIF(4)
        proj = AllToAllProj(
            pre, post, weight=0.5*u.nS,
            syn=Expon.desc(4, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        self.assertTrue(jnp.allclose(W, jnp.full((4, 4), 0.5)))

    def test_no_autapses_when_pre_is_post(self):
        pop = LIF(4)
        proj = AllToAllProj(
            pop, pop, weight=0.5*u.nS,
            syn=Expon.desc(4, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            allow_autapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        self.assertTrue(jnp.allclose(jnp.diag(W), 0.0))
        off = W - jnp.diag(jnp.diag(W))
        self.assertTrue(jnp.allclose(off, 0.5 * (1 - jnp.eye(4))))
```

- [ ] **Step 6.2: Run to verify failure**

Run: `pytest brainpy_state/_network/_projections_test.py::TestAllToAllProj -v`
Expected: `ImportError: cannot import name 'AllToAllProj'`.

- [ ] **Step 6.3: Implement**

Append to `brainpy_state/_network/_projections.py`:
```python
class AllToAllProj(_RuleProj):
    """All-to-all connectivity. Honors ``allow_autapses`` when pre is post."""
    __module__ = 'brainpy.state'

    def _build_conn_spec(self, key):
        del key
        return sample_all_to_all(
            self._size(self.pre), self._size(self.post),
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
        )
```

- [ ] **Step 6.4: Run to verify passing**

Run: `pytest brainpy_state/_network/_projections_test.py::TestAllToAllProj -v`
Expected: 2 passed.

- [ ] **Step 6.5: Commit**

```
git add brainpy_state/_network/_projections.py brainpy_state/_network/_projections_test.py
git commit -m "feat(network): add AllToAllProj"
```

---

## Task 7: `PairwiseBernoulliProj` and `SymmetricPairwiseBernoulliProj`

**Files:**
- Modify: `brainpy_state/_network/_projections.py`
- Modify: `brainpy_state/_network/_projections_test.py`

- [ ] **Step 7.1: Write failing tests**

Append to `brainpy_state/_network/_projections_test.py`:
```python
from brainpy_state._network._projections import (
    PairwiseBernoulliProj, SymmetricPairwiseBernoulliProj,
)


class TestBernoulliProjs(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_pairwise_bernoulli_density_within_tolerance(self):
        pre = LIF(80); post = LIF(80)
        proj = PairwiseBernoulliProj(
            pre, post, p=0.1, weight=1.0*u.nS,
            syn=Expon.desc(80, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=42,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        density = float(jnp.mean(W > 0))
        self.assertAlmostEqual(density, 0.1, delta=0.025)

    def test_pairwise_bernoulli_seed_determinism(self):
        pre = LIF(40); post = LIF(40)
        kw = dict(p=0.2, weight=1.0*u.nS,
                  syn=Expon.desc(40, tau=5*u.ms),
                  out=COBA.desc(E=0*u.mV),
                  seed=7)
        a = u.get_mantissa(PairwiseBernoulliProj(pre, post, **kw)._weight_matrix.value)
        b = u.get_mantissa(PairwiseBernoulliProj(pre, post, **kw)._weight_matrix.value)
        self.assertTrue(jnp.allclose(a, b))

    def test_symmetric_pairwise_bernoulli_is_symmetric(self):
        pop = LIF(40)
        proj = SymmetricPairwiseBernoulliProj(
            pop, pop, p=0.2, weight=1.0*u.nS,
            syn=Expon.desc(40, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=11,
        )
        W = u.get_mantissa(proj._weight_matrix.value) > 0
        self.assertTrue(jnp.array_equal(W, W.T))
```

- [ ] **Step 7.2: Run to verify failure**

Run: `pytest brainpy_state/_network/_projections_test.py::TestBernoulliProjs -v`
Expected: `ImportError`.

- [ ] **Step 7.3: Implement**

Append to `brainpy_state/_network/_projections.py`:
```python
class PairwiseBernoulliProj(_RuleProj):
    """Each (pre, post) pair has independent Bernoulli probability ``p``."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, p: float, **kwargs):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'p must be in [0, 1], got {p}')
        self._p = p
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_pairwise_bernoulli(
            self._size(self.pre), self._size(self.post),
            p=self._p, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class SymmetricPairwiseBernoulliProj(_RuleProj):
    """Symmetric Bernoulli: if edge (i,j) exists then (j,i) exists too.

    Requires pre is post.
    """
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, p: float, **kwargs):
        if pre is not post:
            raise ValueError(
                'symmetric_pairwise_bernoulli requires pre is post'
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f'p must be in [0, 1], got {p}')
        self._p = p
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        # Draw upper triangle, mirror to lower.
        n = self._size(self.pre)
        upper = jax.random.uniform(key, (n, n)) < self._p
        upper = jnp.triu(upper, k=0 if self.allow_autapses else 1)
        mask = upper | upper.T
        if not self.allow_autapses:
            mask = mask & (~jnp.eye(n, dtype=bool))
        pre, post = jnp.where(mask)
        return ConnSpec(pre, post, int(pre.shape[0]))
```

- [ ] **Step 7.4: Run to verify passing**

Run: `pytest brainpy_state/_network/_projections_test.py::TestBernoulliProjs -v`
Expected: 3 passed.

- [ ] **Step 7.5: Commit**

```
git add brainpy_state/_network/_projections.py brainpy_state/_network/_projections_test.py
git commit -m "feat(network): add PairwiseBernoulli and Symmetric variants"
```

---

## Task 8: `FixedIndegreeProj` and `FixedOutdegreeProj`

- [ ] **Step 8.1: Write failing tests**

Append to `brainpy_state/_network/_projections_test.py`:
```python
from brainpy_state._network._projections import FixedIndegreeProj, FixedOutdegreeProj


class TestFixedDegreeProjs(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_fixed_indegree_each_post_has_K(self):
        pre = LIF(50); post = LIF(20)
        proj = FixedIndegreeProj(
            pre, post, K=10, weight=1.0*u.nS,
            syn=Expon.desc(20, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=3, allow_multapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        for j in range(20):
            self.assertEqual(int(jnp.sum(W[:, j] > 0)), 10)

    def test_fixed_outdegree_each_pre_has_K(self):
        pre = LIF(20); post = LIF(50)
        proj = FixedOutdegreeProj(
            pre, post, K=8, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=4, allow_multapses=False,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        for i in range(20):
            self.assertEqual(int(jnp.sum(W[i, :] > 0)), 8)
```

- [ ] **Step 8.2: Run to verify failure** — `ImportError`.

- [ ] **Step 8.3: Implement**

Append to `brainpy_state/_network/_projections.py`:
```python
class FixedIndegreeProj(_RuleProj):
    """Each post-synaptic neuron receives exactly ``K`` incoming edges."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, K: int, **kwargs):
        if K < 0:
            raise ValueError(f'K must be >= 0, got {K}')
        self._K = int(K)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_indegree(
            self._size(self.pre), self._size(self.post),
            K=self._K, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class FixedOutdegreeProj(_RuleProj):
    """Each pre-synaptic neuron has exactly ``K`` outgoing edges."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, K: int, **kwargs):
        if K < 0:
            raise ValueError(f'K must be >= 0, got {K}')
        self._K = int(K)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_outdegree(
            self._size(self.pre), self._size(self.post),
            K=self._K, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )
```

- [ ] **Step 8.4: Run to verify passing.** Expected: 2 passed.

- [ ] **Step 8.5: Commit**

```
git add brainpy_state/_network/_projections.py brainpy_state/_network/_projections_test.py
git commit -m "feat(network): add FixedIndegree and FixedOutdegree projections"
```

---

## Task 9: `FixedTotalNumberProj` and `PairwisePoissonProj`

- [ ] **Step 9.1: Write failing tests**

Append to `brainpy_state/_network/_projections_test.py`:
```python
from brainpy_state._network._projections import (
    FixedTotalNumberProj, PairwisePoissonProj,
)


class TestFixedTotalAndPoisson(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_fixed_total_number(self):
        pre = LIF(50); post = LIF(50)
        proj = FixedTotalNumberProj(
            pre, post, N=137, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=8,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        # With allow_multapses=True (default), the W.at[].add() accumulates
        # — count non-zero entries.
        self.assertGreaterEqual(int(jnp.sum(W > 0)), 130)

    def test_pairwise_poisson_mean(self):
        pre = LIF(50); post = LIF(50)
        proj = PairwisePoissonProj(
            pre, post, mean=0.1, weight=1.0*u.nS,
            syn=Expon.desc(50, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
            seed=9,
        )
        W = u.get_mantissa(proj._weight_matrix.value)
        # Expected count of edges ≈ 50*50*0.1 = 250.
        # W accumulates per-pair counts via add(); sum / mean per pair ≈ 0.1.
        self.assertAlmostEqual(float(jnp.mean(W)), 0.1, delta=0.025)
```

- [ ] **Step 9.2: Run to verify failure** — `ImportError`.

- [ ] **Step 9.3: Implement**

Append to `brainpy_state/_network/_projections.py`:
```python
class FixedTotalNumberProj(_RuleProj):
    """Exactly ``N`` edges drawn uniformly over the (pre, post) grid."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, N: int, **kwargs):
        if N < 0:
            raise ValueError(f'N must be >= 0, got {N}')
        self._N = int(N)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_fixed_total_number(
            self._size(self.pre), self._size(self.post),
            N=self._N, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
            allow_multapses=self.allow_multapses,
        )


class PairwisePoissonProj(_RuleProj):
    """Each (pre, post) pair has a Poisson-distributed number of edges with mean ``mean``."""
    __module__ = 'brainpy.state'

    def __init__(self, pre, post, *, mean: float, **kwargs):
        if mean < 0:
            raise ValueError(f'mean must be >= 0, got {mean}')
        self._mean = float(mean)
        super().__init__(pre, post, **kwargs)

    def _build_conn_spec(self, key):
        return sample_pairwise_poisson(
            self._size(self.pre), self._size(self.post),
            mean=self._mean, key=key,
            pre_is_post=self._pre_is_post,
            allow_autapses=self.allow_autapses,
        )
```

- [ ] **Step 9.4: Run to verify passing.** Expected: 2 passed.

- [ ] **Step 9.5: Commit**

```
git add brainpy_state/_network/_projections.py brainpy_state/_network/_projections_test.py
git commit -m "feat(network): add FixedTotalNumber and PairwisePoisson projections"
```

---

## Task 10: `Network.simulate()` (basic, no monitor)

**Files:**
- Modify: `brainpy_state/_network/_base.py`
- Modify: `brainpy_state/_network/_base_test.py`

- [ ] **Step 10.1: Write failing test**

Append to `brainpy_state/_network/_base_test.py`:
```python
import jax.numpy as jnp


class TestNetworkSimulate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_simulate_runs_for_loop(self):
        from brainpy_state import LIF

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(5)

        net = Net()
        brainstate.nn.init_all_states(net)
        # Should run without error; returns an empty dict when monitor is None.
        out = net.simulate(1.0 * u.ms)
        self.assertEqual(out, {})

    def test_simulate_update_parity(self):
        from brainpy_state import LIF

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net_a = Net()
        net_b = Net()
        brainstate.nn.init_all_states(net_a)
        brainstate.nn.init_all_states(net_b)

        net_a.simulate(0.5 * u.ms)

        times = u.math.arange(0 * u.ms, 0.5 * u.ms, 0.1 * u.ms)
        for t in times:
            with brainstate.environ.context(t=t):
                net_b.update(t)

        # both should leave net.pop.V at the same value (no input)
        self.assertTrue(jnp.allclose(
            net_a.pop.V.value.mantissa, net_b.pop.V.value.mantissa))
```

- [ ] **Step 10.2: Run to verify failure** — `AttributeError: 'Net' object has no attribute 'simulate'`.

- [ ] **Step 10.3: Implement**

Modify `brainpy_state/_network/_base.py` — add `simulate` method:
```python
    def simulate(self, duration, *, dt=None, monitor=None) -> dict:
        """Run the network for ``duration``.

        Wraps ``brainstate.transform.for_loop`` over ``self.update``.

        Parameters
        ----------
        duration : saiunit.Quantity
            Wall-clock time to simulate.
        dt : saiunit.Quantity, optional
            Timestep override. Defaults to ``brainstate.environ.get('dt')``.
        monitor : list[str] | dict[str, Callable] | None
            Recording specification (see ``simulate``'s docstring on the
            next iteration of this module).
        """
        import brainstate.transform as transform
        if dt is None:
            dt = brainstate.environ.get('dt')
        if dt is None:
            raise ValueError(
                'dt must be set via brainstate.environ.set(dt=...) or '
                'passed explicitly as simulate(..., dt=...)'
            )
        times = u.math.arange(0.0 * dt.unit, duration, dt)
        indices = u.math.arange(times.size)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                return None

        transform.for_loop(step, times, indices)
        return {}
```

Make sure `import saiunit as u` is at the top of `_base.py` (add it if not).

- [ ] **Step 10.4: Run to verify passing.** Expected: 2 passed.

- [ ] **Step 10.5: Commit**

```
git add brainpy_state/_network/_base.py brainpy_state/_network/_base_test.py
git commit -m "feat(network): add Network.simulate (basic, no monitor)"
```

---

## Task 11: `Network.simulate(monitor=...)`

**Files:**
- Modify: `brainpy_state/_network/_base.py`
- Modify: `brainpy_state/_network/_base_test.py`

- [ ] **Step 11.1: Write failing tests**

Append to `brainpy_state/_network/_base_test.py`:
```python
class TestNetworkMonitor(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_monitor_attribute_path(self):
        from brainpy_state import LIF

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net = Net()
        brainstate.nn.init_all_states(net)
        out = net.simulate(0.5 * u.ms, monitor=['pop.V'])
        self.assertIn('pop.V', out)
        # 5 timesteps, 3 neurons
        self.assertEqual(out['pop.V'].shape, (5, 3))

    def test_monitor_callable(self):
        from brainpy_state import LIF

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net = Net()
        brainstate.nn.init_all_states(net)
        out = net.simulate(
            0.5 * u.ms,
            monitor={'mean_V': lambda n: jnp.mean(u.get_mantissa(n.pop.V.value))},
        )
        self.assertIn('mean_V', out)
        self.assertEqual(out['mean_V'].shape, (5,))
```

- [ ] **Step 11.2: Run to verify failure.**

- [ ] **Step 11.3: Implement**

Modify `simulate` in `brainpy_state/_network/_base.py`:
```python
    def simulate(self, duration, *, dt=None, monitor=None) -> dict:
        import brainstate.transform as transform
        if dt is None:
            dt = brainstate.environ.get('dt')
        if dt is None:
            raise ValueError(
                'dt must be set via brainstate.environ.set(dt=...) or '
                'passed explicitly as simulate(..., dt=...)'
            )
        times = u.math.arange(0.0 * dt.unit, duration, dt)
        indices = u.math.arange(times.size)

        # Normalize monitor to a dict of callables (one per recorded key).
        callables = {}
        if monitor is None:
            pass
        elif isinstance(monitor, (list, tuple)):
            for path in monitor:
                callables[path] = self._make_path_callable(path)
        elif isinstance(monitor, dict):
            for name, fn in monitor.items():
                if not callable(fn):
                    raise TypeError(
                        f'monitor dict value for {name!r} must be callable, '
                        f'got {type(fn).__name__}'
                    )
                callables[name] = fn
        else:
            raise TypeError(
                f'monitor must be list, dict, or None, got {type(monitor).__name__}'
            )

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                if callables:
                    return {k: fn(self) for k, fn in callables.items()}
                return None

        if callables:
            stacked = transform.for_loop(step, times, indices)
            return dict(stacked)
        transform.for_loop(step, times, indices)
        return {}

    def _make_path_callable(self, path: str):
        parts = path.split('.')

        def fn(net):
            obj = net
            for p in parts:
                obj = getattr(obj, p)
            if hasattr(obj, 'value'):
                obj = obj.value
            return obj
        return fn
```

- [ ] **Step 11.4: Run to verify passing.** Expected: 2 passed.

- [ ] **Step 11.5: Commit**

```
git add brainpy_state/_network/_base.py brainpy_state/_network/_base_test.py
git commit -m "feat(network): add monitor= kwarg to simulate()"
```

---

## Task 12: `Builder` class

**Files:**
- Create: `brainpy_state/_network/_builder.py`
- Create: `brainpy_state/_network/_builder_test.py`

- [ ] **Step 12.1: Write failing tests**

`brainpy_state/_network/_builder_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state import LIF, Expon, COBA
from brainpy_state._network._builder import Builder
from brainpy_state._network._base import Network
from brainpy_state._network._projections import OneToOneProj, FixedIndegreeProj


class TestBuilder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_builder_is_a_network(self):
        b = Builder()
        self.assertIsInstance(b, Network)

    def test_add_sets_attribute_and_returns_module(self):
        b = Builder()
        pop = LIF(5)
        ret = b.add('exc', pop)
        self.assertIs(ret, pop)
        self.assertIs(b.exc, pop)

    def test_connect_by_reference(self):
        b = Builder()
        pre = b.add('pre', LIF(5))
        post = b.add('post', LIF(5))
        proj = b.connect(
            pre, post, rule=OneToOneProj,
            weight=0.1*u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, pre)
        self.assertIs(proj.post, post)
        # Auto-named attribute
        self.assertIn(proj.name, b.nodes())

    def test_connect_by_string_name(self):
        b = Builder()
        b.add('pre', LIF(5))
        b.add('post', LIF(5))
        proj = b.connect(
            'pre', 'post', rule=OneToOneProj,
            weight=0.1*u.nS,
            syn=Expon.desc(5, tau=5*u.ms),
            out=COBA.desc(E=0*u.mV),
        )
        self.assertIs(proj.pre, b.pre)
        self.assertIs(proj.post, b.post)

    def test_duplicate_add_raises(self):
        b = Builder()
        b.add('exc', LIF(5))
        with self.assertRaises(ValueError):
            b.add('exc', LIF(5))

    def test_simulate_works(self):
        b = Builder()
        b.add('pop', LIF(5))
        brainstate.nn.init_all_states(b)
        out = b.simulate(0.5 * u.ms)
        self.assertEqual(out, {})
```

- [ ] **Step 12.2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 12.3: Implement**

`brainpy_state/_network/_builder.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Builder — imperative subclass of Network for scripts and notebooks."""
from __future__ import annotations

import itertools
from typing import Union

import brainstate

from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import Projection
from brainpy_state._network._base import Network

__all__ = ['Builder']


class Builder(Network):
    """Imperative variant of :class:`Network`.

    Use ``add(name, module)`` to register a population/device and
    ``connect(pre, post, rule=..., **params)`` to register a projection.
    Otherwise identical to :class:`Network`.
    """
    __module__ = 'brainpy.state'

    def __init__(self):
        super().__init__()
        self._proj_counter = itertools.count()

    def add(self, name: str, module: brainstate.nn.Module) -> brainstate.nn.Module:
        """Register ``module`` as ``self.<name>``; return the module."""
        if hasattr(self, name):
            raise ValueError(f'attribute {name!r} already exists on this Builder')
        setattr(self, name, module)
        return module

    def connect(
        self,
        pre: Union[str, Dynamics],
        post: Union[str, Dynamics],
        *,
        rule: type,
        **kwargs,
    ) -> Projection:
        """Instantiate ``rule(pre, post, **kwargs)`` and register it."""
        pre_mod = getattr(self, pre) if isinstance(pre, str) else pre
        post_mod = getattr(self, post) if isinstance(post, str) else post
        proj = rule(pre_mod, post_mod, **kwargs)
        attr = f'_proj_{next(self._proj_counter)}'
        setattr(self, attr, proj)
        return proj
```

- [ ] **Step 12.4: Run to verify passing.** Expected: 6 passed.

- [ ] **Step 12.5: Commit**

```
git add brainpy_state/_network/_builder.py brainpy_state/_network/_builder_test.py
git commit -m "feat(network): add Builder imperative API"
```

---

## Task 13: `Recorder` helper for NESTDevice recorders

**Files:**
- Create: `brainpy_state/_network/_recorders.py`
- Create: `brainpy_state/_network/_recorders_test.py`

`Recorder` is a thin Module that reads `getattr(source, attr).value` each step and forwards it to a passive `NESTDevice`. The device's existing `update()` signature is preserved.

- [ ] **Step 13.1: Write failing tests**

`brainpy_state/_network/_recorders_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state import LIF
from brainpy_state._nest.spike_recorder import spike_recorder
from brainpy_state._network._base import Network
from brainpy_state._network._recorders import Recorder


class TestRecorder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_recorder_forwards_attr_to_device(self):
        captured = {'spikes': None}

        class FakeDevice:
            __module__ = 'brainpy_state._nest._base'

            def update(self, spikes=None, **kw):
                captured['spikes'] = spikes

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(5)
                self.rec = Recorder(source=self.pop, attr='spike',
                                    device=FakeDevice())

        net = Net()
        brainstate.nn.init_all_states(net)
        # set pop.spike artificially to simulate a step's output
        net.pop.spike.value = jnp.ones(5)
        net.update()
        self.assertIsNotNone(captured['spikes'])
        self.assertEqual(captured['spikes'].shape, (5,))

    def test_recorder_with_real_spike_recorder_runs(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)
                self.rec = Recorder(
                    source=self.pop, attr='spike',
                    device=spike_recorder(in_size=3),
                )

        net = Net()
        brainstate.nn.init_all_states(net)
        # one step — no spikes (LIF at rest)
        net.update()
        # spike_recorder.events accessible after at least one update.
        self.assertIn('senders', net.rec.device.events)
```

- [ ] **Step 13.2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 13.3: Implement**

`brainpy_state/_network/_recorders.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Recorder — wires a passive NESTDevice to a source population attribute."""
from __future__ import annotations

import brainstate

from brainpy_state._base import Dynamics

__all__ = ['Recorder']


class Recorder(brainstate.nn.Module):
    """Forward ``source.<attr>.value`` to ``device.update(...)`` each step.

    Parameters
    ----------
    source : Dynamics
        Module to read from.
    attr : str
        Attribute name on ``source`` (e.g. ``'spike'``, ``'V'``).
    device : NESTDevice
        Recording device with a compatible ``update`` signature
        (e.g. ``spike_recorder``, ``multimeter``).
    """
    __module__ = 'brainpy.state'

    def __init__(self, source: Dynamics, attr: str, device):
        super().__init__()
        if not hasattr(source, attr):
            raise AttributeError(
                f'source {type(source).__name__} has no attribute {attr!r}'
            )
        self.source = source
        self.attr = attr
        self.device = device

    def update(self, *args, **kwargs):
        val = getattr(self.source, self.attr)
        if hasattr(val, 'value'):
            val = val.value
        self.device.update(val)
```

Note: `Recorder` is a plain `Module`, not a `Projection`, so it runs in the dynamics-phase of `Network.update()` — after projections have fired and the source population has updated its `spike` State. That's the correct timing for capturing the step's spikes.

- [ ] **Step 13.4: Run to verify passing.** Expected: 2 passed.

- [ ] **Step 13.5: Commit**

```
git add brainpy_state/_network/_recorders.py brainpy_state/_network/_recorders_test.py
git commit -m "feat(network): add Recorder helper for NESTDevice recorders"
```

---

## Task 14: Top-level exports

**Files:**
- Modify: `brainpy_state/_network/__init__.py`
- Modify: `brainpy_state/__init__.py`

- [ ] **Step 14.1: Update `brainpy_state/_network/__init__.py`**

Overwrite:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
from ._base import Network
from ._builder import Builder
from ._recorders import Recorder
from ._projections import (
    OneToOneProj,
    AllToAllProj,
    PairwiseBernoulliProj,
    SymmetricPairwiseBernoulliProj,
    FixedIndegreeProj,
    FixedOutdegreeProj,
    FixedTotalNumberProj,
    PairwisePoissonProj,
)

__all__ = [
    'Network',
    'Builder',
    'Recorder',
    'OneToOneProj',
    'AllToAllProj',
    'PairwiseBernoulliProj',
    'SymmetricPairwiseBernoulliProj',
    'FixedIndegreeProj',
    'FixedOutdegreeProj',
    'FixedTotalNumberProj',
    'PairwisePoissonProj',
]
```

- [ ] **Step 14.2: Update `brainpy_state/__init__.py`**

Find the location just before any `__all__` list at the bottom of the file (read the file first; the existing exports section ends around line 80+ based on existing structure). Add these imports:

```python
# =============================================================================
# Network API
# =============================================================================

from ._network import (
    Network,
    Builder,
    Recorder,
    OneToOneProj,
    AllToAllProj,
    PairwiseBernoulliProj,
    SymmetricPairwiseBernoulliProj,
    FixedIndegreeProj,
    FixedOutdegreeProj,
    FixedTotalNumberProj,
    PairwisePoissonProj,
)

from . import _dist as dist  # noqa: F401  -- exposed as brainpy.state.dist
```

If the file has an `__all__`, append every new symbol from the import block above to it, and add `'dist'` to expose the submodule.

- [ ] **Step 14.3: Smoke-test the imports**

Run:
```
python -c "import brainpy_state as bps; print(bps.Network, bps.Builder, bps.Recorder, bps.FixedIndegreeProj, bps.dist.Normal)"
```
Expected: all five class repr lines print without error.

- [ ] **Step 14.4: Run the full new-module test suite**

Run: `pytest brainpy_state/_network/ brainpy_state/_dist_test.py -v`
Expected: all tests pass.

- [ ] **Step 14.5: Commit**

```
git add brainpy_state/_network/__init__.py brainpy_state/__init__.py
git commit -m "feat(network): export Network API from brainpy.state"
```

---

## Task 15: Brunel flagship example + integration test

**Files:**
- Create: `examples/brunel.py`
- Create: `brainpy_state/_network/_brunel_test.py`

Per `nest-status/internal/index.md` P0 #17 — the Brunel example doubles as the IAF-psc-family flagship and the first end-to-end network-level acceptance test. We keep the network small (1000 neurons total) so the test stays under a few seconds.

- [ ] **Step 15.1: Write the integration test**

`brainpy_state/_network/_brunel_test.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Integration test: Brunel network using the Network API."""
import unittest

import brainstate
import jax.numpy as jnp
import saiunit as u

from brainpy_state import (
    Builder, LIF, Expon, COBA,
    FixedIndegreeProj,
)


class TestBrunelIntegration(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_brunel_small_runs_end_to_end(self):
        N_E, N_I = 800, 200
        eps = 0.1  # connection probability target -> K = eps * N
        K_E = int(eps * N_E)
        K_I = int(eps * N_I)

        b = Builder()
        exc = b.add('exc', LIF(N_E, tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV))
        inh = b.add('inh', LIF(N_I, tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV))

        for src, tgt, w, K in [
            (exc, exc, 0.1*u.nS, K_E),
            (exc, inh, 0.1*u.nS, K_E),
            (inh, exc, -0.5*u.nS, K_I),
            (inh, inh, -0.5*u.nS, K_I),
        ]:
            b.connect(src, tgt, rule=FixedIndegreeProj,
                      K=K, weight=w,
                      syn=Expon.desc(tgt.in_size, tau=5*u.ms),
                      out=COBA.desc(E=0*u.mV),
                      seed=42, allow_multapses=False)

        brainstate.nn.init_all_states(b)
        out = b.simulate(50 * u.ms, monitor=['exc.spike', 'inh.spike'])

        self.assertEqual(out['exc.spike'].shape, (500, N_E))
        self.assertEqual(out['inh.spike'].shape, (500, N_I))
        # We don't assert biological behaviour here — that's the validation
        # harness's job. We only assert the network ran without NaNs.
        self.assertFalse(bool(jnp.any(jnp.isnan(out['exc.spike']))))
        self.assertFalse(bool(jnp.any(jnp.isnan(out['inh.spike']))))
```

- [ ] **Step 15.2: Run to verify failure** — the test depends on the new API. If anything is wired wrong, this catches it. Expected on first run: the simulation runs but takes >5 s, OR fails on a specific API mismatch. Either way, fix until it passes.

- [ ] **Step 15.3: Write the standalone example**

`examples/brunel.py`:
```python
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel random network — flagship example for the brainpy.state Network API.

Two populations (excitatory + inhibitory), random fixed-indegree connectivity,
conductance-based synapses. Adapted to the brainpy.state ``Builder`` style.

Run:
    python examples/brunel.py
"""
import brainstate
import matplotlib.pyplot as plt
import saiunit as u

from brainpy_state import (
    Builder, LIF, Expon, COBA, FixedIndegreeProj,
)


def main():
    brainstate.environ.set(dt=0.1 * u.ms)

    N_E, N_I = 800, 200
    eps = 0.1
    K_E = int(eps * N_E)
    K_I = int(eps * N_I)

    b = Builder()
    exc = b.add('exc', LIF(N_E, tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV))
    inh = b.add('inh', LIF(N_I, tau=20*u.ms, V_th=-50*u.mV, V_reset=-60*u.mV))

    for src, tgt, w, K in [
        (exc, exc, 0.1*u.nS, K_E),
        (exc, inh, 0.1*u.nS, K_E),
        (inh, exc, -0.5*u.nS, K_I),
        (inh, inh, -0.5*u.nS, K_I),
    ]:
        b.connect(src, tgt, rule=FixedIndegreeProj,
                  K=K, weight=w,
                  syn=Expon.desc(tgt.in_size, tau=5*u.ms),
                  out=COBA.desc(E=0*u.mV),
                  seed=42, allow_multapses=False)

    brainstate.nn.init_all_states(b)
    out = b.simulate(500 * u.ms, monitor=['exc.spike', 'inh.spike'])

    # Raster plot of the excitatory population
    spikes = out['exc.spike']  # (T, N_E)
    times, neurons = (spikes > 0).nonzero()
    plt.figure(figsize=(8, 4))
    plt.scatter(times * 0.1, neurons, s=0.5, color='k')
    plt.xlabel('time (ms)')
    plt.ylabel('exc neuron index')
    plt.title('Brunel-style network — excitatory raster')
    plt.tight_layout()
    plt.savefig('examples/brunel_raster.png', dpi=100)
    print(f'wrote examples/brunel_raster.png; '
          f'{int(spikes.sum())} excitatory spikes over 500 ms')


if __name__ == '__main__':
    main()
```

- [ ] **Step 15.4: Run the integration test**

Run: `pytest brainpy_state/_network/_brunel_test.py -v`
Expected: 1 passed.

- [ ] **Step 15.5: Run the example as a smoke test (optional, slow)**

Run: `python examples/brunel.py`
Expected: writes `examples/brunel_raster.png` and prints the spike count.

- [ ] **Step 15.6: Commit**

```
git add examples/brunel.py brainpy_state/_network/_brunel_test.py
git commit -m "feat(network): add Brunel flagship example and integration test"
```

---

## Task 16: Changelog and final sweep

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 16.1: Read existing changelog format**

Run: `head -50 /mnt/d/codes/projects/brainpy-state/CHANGELOG.md`

- [ ] **Step 16.2: Add a new entry**

Prepend (after any top header) to `CHANGELOG.md`:
```
## Unreleased

### Added — Network API for NEST-style models

- `brainpy.state.Network` — `brainstate.nn.Module` subclass with
  projection-first `update()` traversal and JIT-wrapped
  `simulate(duration, monitor=...)`.
- `brainpy.state.Builder` — imperative subclass exposing `add()` and
  `connect()`; produces the same underlying module tree as a subclassed
  `Network`.
- Rule-based projections: `OneToOneProj`, `AllToAllProj`,
  `PairwiseBernoulliProj`, `SymmetricPairwiseBernoulliProj`,
  `FixedIndegreeProj`, `FixedOutdegreeProj`, `FixedTotalNumberProj`,
  `PairwisePoissonProj`. Uniform constructor `(pre, post, *, weight,
  delay=None, syn, out, allow_autapses, allow_multapses, seed,
  **rule_kwargs)`.
- `brainpy.state.Recorder` — helper that wires a passive
  `NESTDevice` recorder (e.g. `spike_recorder`) to a source population
  attribute.
- `brainpy.state.dist.{Normal, LogNormal, Uniform}` — distribution
  objects sampled once at projection `__init__`.
- Brunel flagship example at `examples/brunel.py`.

See `docs/superpowers/specs/2026-05-12-nest-network-api-design.md`.
```

- [ ] **Step 16.3: Run the full test suite**

Run: `pytest brainpy_state/_network/ brainpy_state/_dist_test.py -v`
Expected: all green.

- [ ] **Step 16.4: Commit**

```
git add CHANGELOG.md
git commit -m "docs: changelog entry for Network API"
```

---

## Task 17: User-facing documentation page

**Files:**
- Create: `docs/brainpy-guide/network-api.md`
- Modify: `docs/nest-status/internal/network-api-gap.md` (add a short status note)

Per spec §11, the Network API ships with a guide page that introduces both
styles side-by-side.

- [ ] **Step 17.1: Write the guide**

`docs/brainpy-guide/network-api.md`:
```markdown
# Network API

`brainpy.state.Network` is the foundation for assembling NEST-style
neurons, synapses, and devices into a runnable network. Two entry
points produce the same underlying `brainstate.nn.Module` tree:

- **`brainpy.state.Network`** — subclass it; define populations and
  projections as attributes. Canonical style.
- **`brainpy.state.Builder`** — subclass of `Network` that adds
  `add(name, module)` and `connect(pre, post, *, rule, **kwargs)`
  imperative methods. Convenient for scripts and notebooks.

## Quick start — both styles

```python
import brainstate
import saiunit as u
import brainpy_state as bps

brainstate.environ.set(dt=0.1 * u.ms)

# --- Subclass style -----------------------------------------------------
class TwoPopNet(bps.Network):
    def __init__(self):
        super().__init__()
        self.exc = bps.LIF(800)
        self.inh = bps.LIF(200)
        self.e_to_i = bps.FixedIndegreeProj(
            self.exc, self.inh, K=80,
            weight=0.1 * u.nS,
            syn=bps.Expon.desc(200, tau=5*u.ms),
            out=bps.COBA.desc(E=0*u.mV),
        )

net = TwoPopNet()
brainstate.nn.init_all_states(net)
out = net.simulate(100 * u.ms, monitor=['exc.spike'])

# --- Builder style ------------------------------------------------------
b = bps.Builder()
b.add('exc', bps.LIF(800))
b.add('inh', bps.LIF(200))
b.connect(b.exc, b.inh, rule=bps.FixedIndegreeProj,
          K=80, weight=0.1 * u.nS,
          syn=bps.Expon.desc(200, tau=5*u.ms),
          out=bps.COBA.desc(E=0*u.mV))
brainstate.nn.init_all_states(b)
out = b.simulate(100 * u.ms, monitor=['exc.spike'])
```

Both produce identical module trees and identical simulated output for
the same seed.

## Connection rules

| Class | NEST equivalent |
|---|---|
| `OneToOneProj` | `one_to_one` |
| `AllToAllProj` | `all_to_all` |
| `PairwiseBernoulliProj(p=...)` | `pairwise_bernoulli` |
| `SymmetricPairwiseBernoulliProj(p=...)` | `symmetric_pairwise_bernoulli` |
| `FixedIndegreeProj(K=...)` | `fixed_indegree` |
| `FixedOutdegreeProj(K=...)` | `fixed_outdegree` |
| `FixedTotalNumberProj(N=...)` | `fixed_total_number` |
| `PairwisePoissonProj(mean=...)` | `pairwise_poisson` |

All accept the same uniform keyword set: `weight`, `delay=None`, `syn`,
`out`, `allow_autapses=True`, `allow_multapses=True`, `seed=None`.

## Weights and delays

`weight` and `delay` accept scalars, arrays, or
`brainpy.state.dist.{Normal, LogNormal, Uniform}` distribution objects
that are sampled **once** at projection `__init__`. This deliberately
differs from NEST's lazy `Parameter` — concrete values are deterministic
given a `seed` and play cleanly with JIT.

```python
proj = bps.FixedIndegreeProj(
    pre, post, K=80,
    weight=bps.dist.Normal(mean=0.1 * u.nS, std=0.01 * u.nS),
    delay=1.5 * u.ms,
    syn=bps.Expon.desc(len(post), tau=5*u.ms),
    out=bps.COBA.desc(E=0*u.mV),
    seed=42,
)
```

## Recording

Two ways to capture state during `simulate()`:

1. **`monitor=` kwarg** — lightweight, returns stacked arrays:

   ```python
   out = net.simulate(100 * u.ms,
                      monitor=['exc.spike', 'inh.V'])
   spikes = out['exc.spike']   # (T, N_E)
   ```

2. **`Recorder` + `NESTDevice`** — full NEST-faithful recorder semantics:

   ```python
   from brainpy_state._nest.spike_recorder import spike_recorder

   class Net(bps.Network):
       def __init__(self):
           super().__init__()
           self.exc = bps.LIF(800)
           self.rec = bps.Recorder(source=self.exc, attr='spike',
                                   device=spike_recorder(in_size=800))
   ```

   After `simulate()`, access `net.rec.device.events`.

## Stepping by hand

`update()` is canonical — `simulate()` is sugar. Power users can drive
the loop themselves:

```python
times = u.math.arange(0 * u.ms, 100 * u.ms, 0.1 * u.ms)
indices = u.math.arange(times.size)

def step(t, i):
    with brainstate.environ.context(t=t, i=i):
        return net.update(t)

brainstate.transform.for_loop(step, times, indices)
```
```

- [ ] **Step 17.2: Update the gap doc with a status note**

Open `docs/nest-status/internal/network-api-gap.md`. At the end of §7
(the "Prioritized roadmap" section), append:

```
**Status note (post-Network-API):** the brainpy.state-native foundation
(``Network``, ``Builder``, eight ``*Proj`` rule classes, ``Recorder``,
``brainpy.state.dist``) shipped on 2026-05-12. The
``brainpy_state.nest_compat`` PyNEST-style facade tracked above can now
be built as a thin shim over this layer. See
``docs/brainpy-guide/network-api.md`` for the user guide and
``docs/superpowers/specs/2026-05-12-nest-network-api-design.md`` for the
design spec.
```

- [ ] **Step 17.3: Verify the docs build (if Sphinx is configured)**

Run: `ls docs/conf.py && python -c "import sphinx" && cd docs && make html 2>&1 | tail -20`
Expected: build succeeds, or — if Sphinx isn't installed in this env — skip and verify markdown renders by reading both files back. The user guide is markdown so it should be picked up by any MyST-enabled Sphinx build.

- [ ] **Step 17.4: Commit**

```
git add docs/brainpy-guide/network-api.md docs/nest-status/internal/network-api-gap.md
git commit -m "docs(network): add Network API user guide and gap-doc status note"
```

---

## Deferred follow-ups (NOT in this plan)

These are explicitly out of scope for this plan and should be tracked as new tickets after merge:

- Per-edge `delay=` support (v1 supports scalar only).
- Sparse-comm optimization (today every `*Proj` builds a dense `(n_pre, n_post)` weight matrix; replace with `brainstate.nn.SparseLinear` or `brainevent.nn.EventFixedProb` once the API stabilizes).
- Generator-device first-class wiring: spike-emitting generators feeding `*Proj` `pre`, current-emitting generators wired through `add_current_input`.
- `TripartiteConnect` (P1 in `network-api-gap.md`).
- `CollocatedSynapses` (P1).
- Spatial / topology layer (P2).
- `nest_compat` PyNEST shim that sits on top of this foundation.
- Recorder dot-path support for nested attributes (current API is single-level).
