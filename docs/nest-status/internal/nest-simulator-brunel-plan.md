# NEST-style `Simulator` API + `brunel_alpha` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicit NEST-flavored `Simulator` network API in
`brainpy_state/_network`, port `brunel_alpha_nest.py` into `examples/nest/`
driving the real `iaf_psc_alpha`/`poisson_generator`/`spike_recorder` models,
validate it against live NEST 3.9 within 5 % firing-rate, and fix every `_nest`
model bug the port surfaces (failing-test-first).

**Architecture:** A `Simulator` (brainstate Module) builds a module graph and
runs one `brainstate.transform.for_loop`. Populations expose spikes through a
Simulator-managed `ShortTermState` (no per-model `.spike` churn). Recurrent
connections are **delta-event projections**: pre spike → `InputDelay` (reused,
full-delay convention) → weighted-pA `_DenseMatMul` → `post.add_delta_input`,
matching how `iaf_psc_alpha` ingests events (`sum_delta_inputs`, sign-split
ex/in). A `poisson_generator` source is realised as one independent train per
target (NEST fan-out semantics). Recording is collected as a stacked JAX array
from the loop (the `spike_recorder` device mutates Python lists and cannot live
inside the jitted loop).

**Tech Stack:** Python ≥3.11, JAX, brainstate, brainevent, saiunit (`u`),
braintools, NumPy, scipy (Lambert-W for PSP normalisation), pytest/unittest,
live `nest` 3.9 (optional, for parity tests only).

**Design spec:** `docs/nest-status/internal/nest-simulator-brunel-design.md`.

**Scope note (vs spec §6):** the validation harness runs at small `order`
(`NE≈800`); the example defaults to a **dense-feasible `order=400`** (≈2000
neurons). Reaching NEST's `order=2500` needs the dense `_DenseMatMul` swapped for
`brainstate.nn.EventFixedNumConn` — captured as **Task 12 (follow-up)**, not
Phase 1.

**Conventions (project CLAUDE.md):** every class/function sets
`__module__ = 'brainpy.state'`; NumPy-style docstrings on public APIs; `saiunit`
(aliased `u`) for all units; tests are `unittest.TestCase` colocated as
`*_test.py`; commit messages **omit** any `Co-Authored-By` trailer.

---

## File Structure

| Path | Responsibility |
|---|---|
| `brainpy_state/_network/_nodeview.py` | `NodeView` — NEST NodeCollection algebra (`+`, slicing) over `(population, local-indices)` segments. |
| `brainpy_state/_network/_rules.py` | Connection rules as values: `all_to_all`, `one_to_one`, `fixed_indegree(K)` — thin wrappers over existing samplers. |
| `brainpy_state/_network/_event_proj.py` | `EventProjection` — single-pop delta+delay event projection (dense matmul or one-to-one element-wise). |
| `brainpy_state/_network/_simulator.py` | `Simulator`, `SimulationResult`, `_SpikeHolder`, generator-spec handling, the step loop. |
| `brainpy_state/_network/__init__.py` | Re-export the new public names (additive; keep existing exports). |
| `brainpy_state/__init__.py` | Expose `brainpy.state.network` + `Simulator`/rules at top level (additive). |
| `examples/nest/brunel_alpha.py` | The flagship port. |
| `examples/nest/__init__.py` | (empty; marks the package dir). |
| `brainpy_state/_nest/_validation/__init__.py` | (empty; new validation package). |
| `brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py` | Single-neuron `iaf_psc_alpha` vs live NEST (deterministic) — delay calibration + bug surfacing. |
| `brainpy_state/_nest/_validation/device_parity_test.py` | `poisson_generator` rate + `spike_recorder` stamping vs NEST. |
| `brainpy_state/_nest/_validation/brunel_alpha_test.py` | Network-level firing-rate parity vs live NEST (skip-if-unavailable). |
| `brainpy_state/_network/_nodeview_test.py`, `_rules_test.py`, `_event_proj_test.py`, `_simulator_test.py` | Colocated unit tests for the new network layer. |
| `changelog.md` | Version entry on completion. |

Each `_nest/*` model fix lands in that model's existing colocated `*_test.py`
with a failing test first (Tasks 7–8).

---

## Task 1: Scaffold the `_network` additions (import contract)

**Files:**
- Create: `brainpy_state/_network/_nodeview.py`, `_rules.py`, `_event_proj.py`, `_simulator.py`
- Modify: `brainpy_state/_network/__init__.py`
- Test: `brainpy_state/_network/_simulator_test.py`

- [ ] **Step 1: Write the failing import test**

```python
# brainpy_state/_network/_simulator_test.py
import unittest


class TestNetworkPublicAPI(unittest.TestCase):
    def test_public_names_importable(self):
        from brainpy_state.network import (
            Simulator, SimulationResult, NodeView,
            all_to_all, one_to_one, fixed_indegree,
        )
        self.assertTrue(callable(fixed_indegree))
        self.assertIsNotNone(Simulator)
        self.assertIsNotNone(SimulationResult)
        self.assertIsNotNone(NodeView)
        self.assertIsNotNone(all_to_all)
        self.assertIsNotNone(one_to_one)
```

- [ ] **Step 2: Run it; verify it fails**

Run: `python -m pytest brainpy_state/_network/_simulator_test.py -q`
Expected: FAIL — `ImportError` (modules/names not defined yet). Note:
`brainpy_state.network` resolves via the top-level alias added in Task 6; until
then import `from brainpy_state._network import ...`. Use the `_network` path in
this test for now and switch to `brainpy_state.network` in Task 6.

Adjust Step 1's import to `from brainpy_state._network import (...)` for Tasks 1–5.

- [ ] **Step 3: Create the four module files as stubs with the public names**

```python
# brainpy_state/_network/_nodeview.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NodeView — NEST NodeCollection-style view over (population, local indices)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import jax.numpy as jnp
from brainpy_state._base import Dynamics

__all__ = ['NodeView']


def _flat_size(module: Dynamics) -> int:
    sz = module.varshape
    if isinstance(sz, tuple):
        n = 1
        for s in sz:
            n *= int(s)
        return n
    return int(sz)


@dataclass(frozen=True)
class _Segment:
    population: object
    indices: jnp.ndarray  # 1-D int array: local indices into the population


class NodeView:
    """A view over one or more slices of populations/devices (NEST-style)."""
    __module__ = 'brainpy.state'

    def __init__(self, segments: List[_Segment]):
        self._segments = list(segments)

    @classmethod
    def of(cls, population) -> 'NodeView':
        return cls([_Segment(population, jnp.arange(_flat_size(population)))])

    @property
    def segments(self) -> List[_Segment]:
        return self._segments

    @property
    def size(self) -> int:
        return int(sum(int(s.indices.shape[0]) for s in self._segments))

    def __add__(self, other: 'NodeView') -> 'NodeView':
        if not isinstance(other, NodeView):
            return NotImplemented
        return NodeView(self._segments + other._segments)

    def __getitem__(self, item) -> 'NodeView':
        if len(self._segments) != 1:
            raise NotImplementedError('slicing supported on single-segment views only')
        seg = self._segments[0]
        idx = seg.indices[item]
        idx = idx[None] if idx.ndim == 0 else idx
        return NodeView([_Segment(seg.population, idx)])

    def __len__(self) -> int:
        return self.size
```

```python
# brainpy_state/_network/_rules.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connection rules as values, wrapping the internal samplers."""
from __future__ import annotations
from dataclasses import dataclass
from brainpy_state._network._connectivity import (
    ConnSpec, sample_all_to_all, sample_one_to_one, sample_fixed_indegree,
)

__all__ = ['ConnRule', 'all_to_all', 'one_to_one', 'fixed_indegree']


class ConnRule:
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses) -> ConnSpec:
        raise NotImplementedError


class _AllToAll(ConnRule):
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_all_to_all(n_pre, n_post, pre_is_post=pre_is_post,
                                 allow_autapses=allow_autapses)


class _OneToOne(ConnRule):
    __module__ = 'brainpy.state'

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_one_to_one(n_pre, n_post)


@dataclass(frozen=True)
class _FixedIndegree(ConnRule):
    K: int

    def sample(self, n_pre, n_post, *, key, pre_is_post, allow_autapses, allow_multapses):
        return sample_fixed_indegree(n_pre, n_post, K=self.K, key=key,
                                     pre_is_post=pre_is_post,
                                     allow_autapses=allow_autapses,
                                     allow_multapses=allow_multapses)


all_to_all = _AllToAll()
one_to_one = _OneToOne()


def fixed_indegree(K: int) -> _FixedIndegree:
    """Each post-synaptic neuron receives exactly ``K`` incoming edges."""
    if int(K) < 0:
        raise ValueError(f'K must be >= 0, got {K}')
    return _FixedIndegree(int(K))
```

```python
# brainpy_state/_network/_event_proj.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""EventProjection — single-population delta+delay event projection."""
from __future__ import annotations
from typing import Callable, Optional
import brainstate
import jax
import jax.numpy as jnp
import saiunit as u
from brainstate.util import get_unique_name
from brainpy_state._brainpy._delay import InputDelay
from brainpy_state._network._connectivity import resolve_param
from brainpy_state._network._rules import ConnRule, _OneToOne

__all__ = ['EventProjection']


class EventProjection(brainstate.nn.Module):
    """Route delayed, weighted (pA) pre-spike events into ``post.add_delta_input``.

    Reads the pre population's captured spike via ``pre_spike()`` (a callable
    returning the full pre-population spike/counts vector), applies ``InputDelay``
    (full-delay convention, matching ``AlignPostProj``), restricts to this
    projection's pre segment, maps to the post segment (dense weighted matmul, or
    element-wise for ``one_to_one``), scatters into a full post-population pA
    vector, and registers it as a delta input.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        *,
        pre_spike: Callable[[], jnp.ndarray],
        n_pre_pop: int,
        pre_local_idx: jnp.ndarray,
        post,
        post_local_idx: jnp.ndarray,
        rule: ConnRule,
        weight,
        delay=None,
        pre_is_post: bool = False,
        allow_autapses: bool = True,
        allow_multapses: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))
        self.pre_spike = pre_spike
        self.post = post
        self.pre_local_idx = jnp.asarray(pre_local_idx)
        self.post_local_idx = jnp.asarray(post_local_idx)
        self._n_pre_pop = int(n_pre_pop)
        self._n_post_pop = _post_pop_size(post)
        self._one_to_one = isinstance(rule, _OneToOne)

        n_pre = int(self.pre_local_idx.shape[0])
        n_post = int(self.post_local_idx.shape[0])
        key = jax.random.key(0 if seed is None else int(seed))
        k_conn, k_w = jax.random.split(key, 2)

        if self._one_to_one:
            # Element-wise: weight is a scalar (pA) applied per matched element.
            self._weight = weight
        else:
            spec = rule.sample(n_pre, n_post, key=k_conn, pre_is_post=pre_is_post,
                               allow_autapses=allow_autapses, allow_multapses=allow_multapses)
            if spec.n_edges == 0:
                W = jnp.zeros((n_pre, n_post))
                self._W = brainstate.ParamState(W)
            else:
                w_edge = resolve_param(weight, (spec.n_edges,), k_w)
                if isinstance(w_edge, u.Quantity):
                    w_mant, w_unit = u.split_mantissa_unit(w_edge)
                else:
                    w_mant, w_unit = jnp.asarray(w_edge), u.UNITLESS
                W = jnp.zeros((n_pre, n_post), dtype=w_mant.dtype).at[spec.pre_idx, spec.post_idx].add(w_mant)
                self._W = brainstate.ParamState(u.Quantity(W, unit=w_unit) if w_unit is not u.UNITLESS else W)

        self.delay_seam = InputDelay((self._n_pre_pop,), delay) if delay is not None else None

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        if self.delay_seam is not None:
            self.delay_seam.init_state(*args, **kwargs)

    def update(self):
        x_full = self.pre_spike()                       # (n_pre_pop,)
        if self.delay_seam is not None:
            x_full = self.delay_seam.update(x_full)
        x_seg = x_full[self.pre_local_idx]              # (n_pre,)
        if self._one_to_one:
            y = x_seg * self._weight                    # (n_post,) pA
        else:
            W = self._W.value
            if isinstance(W, u.Quantity):
                y = u.Quantity(jnp.asarray(x_seg) @ W.mantissa, unit=W.unit)
            else:
                y = jnp.asarray(x_seg) @ W
        contrib = _scatter_post(y, self.post_local_idx, self._n_post_pop)
        self.post.add_delta_input(self.name, contrib)


def _post_pop_size(post) -> int:
    sz = post.varshape
    if isinstance(sz, tuple):
        n = 1
        for s in sz:
            n *= int(s)
        return n
    return int(sz)


def _scatter_post(y, post_local_idx, n_post_pop):
    """Place per-segment contributions into a full (n_post_pop,) vector."""
    if int(post_local_idx.shape[0]) == n_post_pop and bool(
        jnp.all(post_local_idx == jnp.arange(n_post_pop))
    ):
        return y  # full-population segment: no scatter needed (Brunel fast path)
    if isinstance(y, u.Quantity):
        base = jnp.zeros(n_post_pop, dtype=y.mantissa.dtype)
        return u.Quantity(base.at[post_local_idx].add(y.mantissa), unit=y.unit)
    base = jnp.zeros(n_post_pop, dtype=y.dtype)
    return base.at[post_local_idx].add(y)
```

```python
# brainpy_state/_network/_simulator.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Simulator — explicit NEST-flavored network builder + runner."""
from __future__ import annotations
from typing import Optional
import itertools
import brainstate
import jax.numpy as jnp
import saiunit as u
from brainpy_state._base import Neuron
from brainpy_state._nest._base import NESTDevice
from brainpy_state._nest.spike_recorder import spike_recorder as _spike_recorder
from brainpy_state._network._nodeview import NodeView, _flat_size
from brainpy_state._network._rules import all_to_all
from brainpy_state._network._event_proj import EventProjection

__all__ = ['Simulator', 'SimulationResult']


class _SpikeHolder(brainstate.nn.Module):
    __module__ = 'brainpy.state'

    def __init__(self, n: int):
        super().__init__()
        self._n = int(n)

    def init_state(self, *args, **kwargs):
        self.spk = brainstate.ShortTermState(
            jnp.zeros(self._n, dtype=brainstate.environ.dftype())
        )


class _GeneratorSpec:
    """Deferred generator: realised per-target at connect() (NEST fan-out)."""
    def __init__(self, model_cls, params):
        self.model_cls = model_cls
        self.params = params


class SimulationResult:
    __module__ = 'brainpy.state'

    def __init__(self, recordings: dict, duration, dt):
        self._rec = recordings          # tap-id -> (T, n_rec) array
        self._T = duration
        self._dt = dt

    def spikes(self, node):
        return self._rec[id(node)]

    def n_events(self, node) -> int:
        return int(jnp.sum(self._rec[id(node)] > 0))

    def rate(self, node) -> float:
        spk = self._rec[id(node)]
        n = spk.shape[1]
        T_s = float(u.maybe_decimal(self._T / u.second))
        return float(jnp.sum(spk > 0)) / n / T_s


class Simulator(brainstate.nn.Module):
    __module__ = 'brainpy.state'

    def __init__(self, *, dt):
        super().__init__()
        brainstate.environ.set(dt=dt)
        self._dt = dt
        self._neurons = {}       # id -> module
        self._generators = {}    # id -> module
        self._holders = {}       # id -> _SpikeHolder
        self._taps = {}          # recorder-id -> (source population, local idx)
        self._proj_counter = itertools.count()

    # -- node creation -----------------------------------------------------
    def create(self, model_cls, size=1, *, params=None, **kw):
        p = dict(params or {})
        p.update(kw)
        if _is_generator(model_cls):
            spec = _GeneratorSpec(model_cls, p)
            return NodeView([_GenSegment(spec)])
        mod = model_cls(size, **p)
        setattr(self, f'_node_{id(mod)}', mod)
        if isinstance(mod, _spike_recorder):
            return NodeView.of(mod)
        n = _flat_size(mod)
        holder = _SpikeHolder(n)
        setattr(self, f'_holder_{id(mod)}', holder)
        self._holders[id(mod)] = holder
        self._neurons[id(mod)] = mod
        return NodeView.of(mod)

    # -- connection --------------------------------------------------------
    def connect(self, pre: NodeView, post: NodeView, *, rule=all_to_all,
                weight=None, delay=None, allow_autapses=True, allow_multapses=True,
                seed: Optional[int] = None):
        # recorder tap?
        if len(post.segments) == 1 and isinstance(post.segments[0].population, _spike_recorder):
            for pre_seg in pre.segments:
                rec = post.segments[0].population
                self._taps[id(rec)] = (pre_seg.population, pre_seg.indices)
            return
        for pre_seg in pre.segments:
            for post_seg in post.segments:
                self._connect_pair(pre_seg, post_seg, rule, weight, delay,
                                   allow_autapses, allow_multapses, seed)

    def _connect_pair(self, pre_seg, post_seg, rule, weight, delay,
                      allow_autapses, allow_multapses, seed):
        post_pop = post_seg.population
        # realise a generator source sized to this target (independent trains)
        if isinstance(pre_seg, _GenSegment):
            spec = pre_seg.spec
            n = int(post_seg.indices.shape[0])
            gen = spec.model_cls(n, **spec.params)
            setattr(self, f'_node_{id(gen)}', gen)
            holder = _SpikeHolder(n)
            setattr(self, f'_holder_{id(gen)}', holder)
            self._holders[id(gen)] = holder
            self._generators[id(gen)] = gen
            from brainpy_state._network._rules import one_to_one
            proj = EventProjection(
                pre_spike=_holder_reader(holder), n_pre_pop=n,
                pre_local_idx=jnp.arange(n), post=post_pop,
                post_local_idx=post_seg.indices, rule=one_to_one, weight=weight,
                delay=delay, seed=seed)
        else:
            pre_pop = pre_seg.population
            holder = self._holders[id(pre_pop)]
            proj = EventProjection(
                pre_spike=_holder_reader(holder), n_pre_pop=_flat_size(pre_pop),
                pre_local_idx=pre_seg.indices, post=post_pop,
                post_local_idx=post_seg.indices, rule=rule, weight=weight,
                delay=delay, pre_is_post=(pre_pop is post_pop),
                allow_autapses=allow_autapses, allow_multapses=allow_multapses, seed=seed)
        setattr(self, f'_proj_{next(self._proj_counter)}', proj)

    # -- run ---------------------------------------------------------------
    def update(self, t=None):
        for name, proj in self.nodes(allowed_hierarchy=(1, 1)).items():
            if isinstance(proj, EventProjection):
                proj.update()
        for id_, gen in self._generators.items():
            counts = gen.update()
            self._holders[id_].spk.value = jnp.asarray(counts, dtype=brainstate.environ.dftype())
        for id_, neu in self._neurons.items():
            spk = neu.update()
            hard = jnp.where(jnp.asarray(u.get_mantissa(spk)) > 0, 1.0, 0.0)
            self._holders[id_].spk.value = jnp.asarray(hard, dtype=brainstate.environ.dftype())

    def simulate(self, duration, *, dt=None) -> SimulationResult:
        import brainstate.transform as transform
        if dt is None:
            dt = self._dt
        brainstate.nn.init_all_states(self)
        times = u.math.arange(0.0 * u.get_unit(dt), duration, dt)
        indices = u.math.arange(times.size)

        taps = dict(self._taps)

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                return {rid: self._holders[id(src)].spk.value[idx]
                        for rid, (src, idx) in taps.items()}

        stacked = transform.for_loop(step, times, indices)
        recordings = {rid: jnp.asarray(stacked[rid]) for rid in taps}
        return SimulationResult(recordings, duration, dt)


def _is_generator(model_cls) -> bool:
    name = getattr(model_cls, '__name__', '')
    return 'generator' in name or 'injector' in name or name in ('spike_generator',)


class _GenSegment:
    """A NodeView segment carrying a deferred generator spec (size unknown)."""
    def __init__(self, spec: _GeneratorSpec):
        self.spec = spec
        self.population = None
        self.indices = jnp.arange(0)


def _holder_reader(holder: _SpikeHolder):
    return lambda: holder.spk.value
```

- [ ] **Step 4: Wire `__init__.py` exports**

```python
# brainpy_state/_network/__init__.py  — ADD to the existing file
from ._nodeview import NodeView
from ._rules import all_to_all, one_to_one, fixed_indegree, ConnRule
from ._event_proj import EventProjection
from ._simulator import Simulator, SimulationResult
```
Append `'NodeView', 'all_to_all', 'one_to_one', 'fixed_indegree', 'ConnRule',
'EventProjection', 'Simulator', 'SimulationResult'` to `__all__`.

- [ ] **Step 5: Point the Task-1 test at `_network` and run it**

Edit the test import to `from brainpy_state._network import (...)` (drop
`SimulationResult`/`Simulator` only if a circular import appears — it should not).
Run: `python -m pytest brainpy_state/_network/_simulator_test.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add brainpy_state/_network/_nodeview.py brainpy_state/_network/_rules.py \
        brainpy_state/_network/_event_proj.py brainpy_state/_network/_simulator.py \
        brainpy_state/_network/__init__.py brainpy_state/_network/_simulator_test.py
git commit -m "feat(network): scaffold NEST-style Simulator API (NodeView, rules, event proj)"
```

---

## Task 2: `NodeView` algebra

**Files:**
- Test: `brainpy_state/_network/_nodeview_test.py`
- Modify (if a bug is found): `brainpy_state/_network/_nodeview.py`

- [ ] **Step 1: Write the test**

```python
# brainpy_state/_network/_nodeview_test.py
import unittest
import brainstate
import saiunit as u
from brainpy_state import iaf_psc_alpha
from brainpy_state._network import NodeView


class TestNodeView(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_size_concat_slice(self):
        a = iaf_psc_alpha(4)
        b = iaf_psc_alpha(2)
        va, vb = NodeView.of(a), NodeView.of(b)
        self.assertEqual(va.size, 4)
        self.assertEqual((va + vb).size, 6)
        self.assertEqual(len((va + vb).segments), 2)
        self.assertEqual(va[:2].size, 2)
        self.assertEqual(va[:2].segments[0].population is a, True)

    def test_concat_then_slice_rejected_multisegment(self):
        a = iaf_psc_alpha(4)
        b = iaf_psc_alpha(2)
        with self.assertRaises(NotImplementedError):
            _ = (NodeView.of(a) + NodeView.of(b))[:3]
```

- [ ] **Step 2: Run; expect PASS** (implementation landed in Task 1)

Run: `python -m pytest brainpy_state/_network/_nodeview_test.py -q`
Expected: PASS. If FAIL, fix `_nodeview.py` minimally, re-run.

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/_network/_nodeview_test.py brainpy_state/_network/_nodeview.py
git commit -m "test(network): NodeView concat/slice algebra"
```

---

## Task 3: Connection rules

**Files:**
- Test: `brainpy_state/_network/_rules_test.py`

- [ ] **Step 1: Write the test**

```python
# brainpy_state/_network/_rules_test.py
import unittest
import jax
from brainpy_state._network import all_to_all, one_to_one, fixed_indegree


class TestRules(unittest.TestCase):
    def test_fixed_indegree_edge_count(self):
        spec = fixed_indegree(3).sample(
            10, 5, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 15)        # K=3 per each of 5 post

    def test_all_to_all_count(self):
        spec = all_to_all.sample(
            4, 6, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 24)

    def test_one_to_one_diagonal(self):
        spec = one_to_one.sample(
            5, 5, key=jax.random.key(0), pre_is_post=False,
            allow_autapses=True, allow_multapses=True)
        self.assertEqual(spec.n_edges, 5)

    def test_fixed_indegree_negative_K_rejected(self):
        with self.assertRaises(ValueError):
            fixed_indegree(-1)
```

- [ ] **Step 2: Run; expect PASS**

Run: `python -m pytest brainpy_state/_network/_rules_test.py -q`
Expected: PASS. Fix `_rules.py` minimally if needed.

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/_network/_rules_test.py brainpy_state/_network/_rules.py
git commit -m "test(network): connection-rule edge-count semantics"
```

---

## Task 4: `EventProjection` delivers delayed weighted delta input

**Files:**
- Test: `brainpy_state/_network/_event_proj_test.py`
- Modify (if needed): `brainpy_state/_network/_event_proj.py`

This test drives the projection by hand: a fake pre-spike source and a real
`iaf_psc_alpha` post, asserting the delta input arrives, weighted (pA), after the
configured delay.

- [ ] **Step 1: Write the test**

```python
# brainpy_state/_network/_event_proj_test.py
import unittest
import brainstate
import jax.numpy as jnp
import saiunit as u
from brainpy_state import iaf_psc_alpha
from brainpy_state._network import one_to_one
from brainpy_state._network._event_proj import EventProjection


class _Box:
    def __init__(self, val):
        self.val = val


class TestEventProjection(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_one_to_one_weighted_delta_after_delay(self):
        post = iaf_psc_alpha(1, tau_syn_ex=1.0 * u.ms)
        box = _Box(jnp.zeros(1))
        proj = EventProjection(
            pre_spike=lambda: box.val, n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=post, post_local_idx=jnp.arange(1), rule=one_to_one,
            weight=100.0 * u.pA, delay=0.5 * u.ms)
        brainstate.nn.init_all_states(post)
        brainstate.nn.init_all_states(proj)

        # No spike yet -> the delta summed by the neuron is ~0 pA.
        with brainstate.environ.context(t=0.0 * u.ms, i=0):
            proj.update()
            summed0 = post.sum_delta_inputs(0.0 * u.pA)
        self.assertAlmostEqual(float(u.get_mantissa(summed0 / u.pA)), 0.0, places=6)

        # Emit one spike; after >= delay steps the neuron must see ~100 pA once.
        box.val = jnp.ones(1)
        seen = []
        for k in range(1, 12):
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                proj.update()
                seen.append(float(u.get_mantissa(post.sum_delta_inputs(0.0 * u.pA) / u.pA)))
            box.val = jnp.zeros(1)  # single spike only at step 0
        self.assertTrue(any(abs(v - 100.0) < 1e-3 for v in seen),
                        f'expected a ~100 pA delta once; saw {seen}')
```

- [ ] **Step 2: Run; verify behavior**

Run: `python -m pytest brainpy_state/_network/_event_proj_test.py -q`
Expected: PASS. If the weighted delta never appears, inspect units (`weight` must
be pA), the `InputDelay` buffer (`init_state` must run), and `add_delta_input`
keying. Fix `_event_proj.py` minimally; re-run. (Exact delivery step is
calibrated against NEST in Task 7 — here we only assert the event arrives once,
weighted.)

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/_network/_event_proj_test.py brainpy_state/_network/_event_proj.py
git commit -m "test(network): EventProjection delivers delayed weighted delta input"
```

---

## Task 5: `Simulator` runs a tiny end-to-end network

**Files:**
- Test: `brainpy_state/_network/_simulator_test.py` (extend)
- Modify (if needed): `brainpy_state/_network/_simulator.py`

- [ ] **Step 1: Add the integration test**

```python
# append to brainpy_state/_network/_simulator_test.py
import brainstate
import jax.numpy as jnp
import saiunit as u
from brainpy_state import iaf_psc_alpha, poisson_generator, spike_recorder
from brainpy_state._network import Simulator, fixed_indegree, all_to_all


class TestSimulatorEndToEnd(unittest.TestCase):
    def test_two_population_network_runs(self):
        sim = Simulator(dt=0.1 * u.ms)
        npar = dict(C_m=250. * u.pF, tau_m=20. * u.ms, tau_syn_ex=0.5 * u.ms,
                    tau_syn_in=0.5 * u.ms, t_ref=2. * u.ms, E_L=0. * u.mV,
                    V_reset=0. * u.mV, V_th=20. * u.mV,
                    V_initializer=__import__('braintools').init.Constant(0. * u.mV))
        ne = sim.create(iaf_psc_alpha, 40, params=npar)
        ni = sim.create(iaf_psc_alpha, 10, params=npar)
        noise = sim.create(poisson_generator, rate=8000. * u.Hz)
        esr = sim.create(spike_recorder)

        sim.connect(noise, ne, weight=20. * u.pA, delay=1.5 * u.ms, rule=all_to_all)
        sim.connect(noise, ni, weight=20. * u.pA, delay=1.5 * u.ms, rule=all_to_all)
        sim.connect(ne, ne + ni, weight=20. * u.pA, delay=1.5 * u.ms,
                    rule=fixed_indegree(4), allow_multapses=False, seed=1)
        sim.connect(ni, ne + ni, weight=-100. * u.pA, delay=1.5 * u.ms,
                    rule=fixed_indegree(1), allow_multapses=False, seed=2)
        sim.connect(ne[:20], esr)

        res = sim.simulate(50. * u.ms)
        spk = res.spikes(esr.segments[0].population)
        self.assertEqual(spk.shape, (500, 20))         # 50 ms / 0.1 ms, N_rec=20
        self.assertFalse(bool(jnp.any(jnp.isnan(spk))))
        self.assertGreaterEqual(res.n_events(esr.segments[0].population), 0)
        self.assertGreater(float(jnp.sum(spk > 0)), 0.0)  # poisson drive -> some spikes
```

- [ ] **Step 2: Run; debug to green**

Run: `python -m pytest brainpy_state/_network/_simulator_test.py -q`
Expected: PASS. Likely first-run issues to fix in `_simulator.py`:
`init_all_states` not reaching projection/holder states (ensure holders &
projections are set as attributes so they're in the module tree); generator
realised twice (ensure one generator per `noise→target` connect); `for_loop`
monitor dict keys must be hashable ints (`id(rec)`). Use
superpowers:systematic-debugging if a failure is non-obvious.

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/_network/_simulator_test.py brainpy_state/_network/_simulator.py
git commit -m "test(network): Simulator runs a two-population poisson-driven network"
```

---

## Task 6: Expose `brainpy.state.network` + top-level names

**Files:**
- Modify: `brainpy_state/__init__.py`
- Test: `brainpy_state/_network/_simulator_test.py` (flip the Task-1 import)

- [ ] **Step 1: Add the alias + top-level exports**

In `brainpy_state/__init__.py`, after the existing imports, add:

```python
from brainpy_state import _network as network
from brainpy_state._network import (
    Simulator, SimulationResult, NodeView,
    all_to_all, one_to_one, fixed_indegree,
)
```
Add those six names to `brainpy_state/__init__.py`'s `__all__`. (Confirm
`brainpy.state` re-exports `brainpy_state`; the models already import from
`brainpy.state` in docstrings, so the package alias exists.)

- [ ] **Step 2: Flip the public-API test to the public path**

Change Task-1's test import back to `from brainpy_state.network import (...)`
(now including `Simulator`, `SimulationResult`).

Run: `python -m pytest brainpy_state/_network/_simulator_test.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/__init__.py brainpy_state/_network/_simulator_test.py
git commit -m "feat(network): expose brainpy.state.network + Simulator/rules at top level"
```

---

## Task 7: Calibrate `iaf_psc_alpha` against live NEST (single neuron)

This is the **delay calibration + bug-surfacing** task. Deterministic: identical
parameters, identical inputs, compare `V_m` traces and spike steps. NEST is
required; skip cleanly when absent.

**Files:**
- Create: `brainpy_state/_nest/_validation/__init__.py` (empty)
- Create: `brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py`
- Modify (only if a discrepancy is found): `brainpy_state/_nest/iaf_psc_alpha.py`
  and/or `brainpy_state/_network/_event_proj.py`

- [ ] **Step 1: Write the parity test (constant current)**

```python
# brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py
import unittest
import numpy as np
import brainstate
import saiunit as u
import braintools

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=0.5, tau_syn_in=0.5,
            t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=20.)


@unittest.skipUnless(_HAS_NEST, "live NEST not importable")
class TestIafPscAlphaParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_vm_trace(self, I_e, T_ms):
        nest.ResetKernel()
        nest.resolution = 0.1
        n = nest.Create("iaf_psc_alpha", 1, params={**NPAR, "I_e": I_e})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(mm, n)
        nest.Simulate(T_ms)
        return np.asarray(mm.get("events")["V_m"])

    def _bp_vm_trace(self, I_e, T_ms):
        neu = brainstate.nn.init_all_states(__import__('brainpy_state').iaf_psc_alpha(
            1, C_m=250. * u.pF, tau_m=20. * u.ms, tau_syn_ex=0.5 * u.ms,
            tau_syn_in=0.5 * u.ms, t_ref=2. * u.ms, E_L=0. * u.mV,
            V_reset=0. * u.mV, V_th=20. * u.mV, I_e=I_e * u.pA,
            V_initializer=braintools.init.Constant(0. * u.mV)))
        n_steps = int(round(T_ms / 0.1))
        vs = []
        for k in range(n_steps):
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                neu.update()
                vs.append(float(u.get_mantissa(neu.V.value[0] / u.mV)))
        return np.asarray(vs)

    def test_subthreshold_vm_matches_nest(self):
        # 200 pA keeps V below 20 mV threshold (steady ~16 mV).
        nest_v = self._nest_vm_trace(200.0, 100.0)
        bp_v = self._bp_vm_trace(200.0, 100.0)
        m = min(len(nest_v), len(bp_v))
        # Allow a one-step alignment offset between recorders; compare overlap.
        err = np.min([np.max(np.abs(nest_v[:m] - bp_v[:m])),
                      np.max(np.abs(nest_v[1:m] - bp_v[:m-1]))])
        self.assertLess(err, 0.05, f"max |Vm| diff {err} mV exceeds 0.05 mV")

    def test_suprathreshold_spike_count_matches_nest(self):
        nest.ResetKernel(); nest.resolution = 0.1
        n = nest.Create("iaf_psc_alpha", 1, params={**NPAR, "I_e": 400.0})
        sr = nest.Create("spike_recorder"); nest.Connect(n, sr)
        nest.Simulate(1000.0)
        nest_count = sr.n_events
        bp_v = self._bp_vm_trace(400.0, 1000.0)
        # brainpy spike when V reaches threshold then resets: count reset events.
        bp_count = int(np.sum((bp_v[:-1] >= 19.999) & (bp_v[1:] <= 0.001)))
        self.assertLessEqual(abs(nest_count - bp_count), 2,
                             f"spike count NEST={nest_count} brainpy={bp_count}")
```

- [ ] **Step 2: Run the constant-current parity**

Run: `python -m pytest brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py -q`
Expected (if the model is correct): PASS or skip. If it FAILS, the model has a
real bug — **proceed to Step 3**. If it passes, skip Step 3.

- [ ] **Step 3: If failing — fix `iaf_psc_alpha`, failing-test-first**

Use superpowers:systematic-debugging. Add the minimal failing unit test to
`brainpy_state/_nest/iaf_psc_alpha_test.py` capturing the precise discrepancy
(e.g. a 5-step `V_m` sequence under 200 pA), confirm it fails, fix the model
(prime suspects: propagator `P30/P31/P32`, `expm1` sign, `y0` one-step buffering,
`t_ref` step rounding), confirm both the unit test and the parity test pass.

```bash
# after fix
python -m pytest brainpy_state/_nest/iaf_psc_alpha_test.py \
                 brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py -q
```

- [ ] **Step 4: Write the delayed-spike calibration test (pins the event-proj delay)**

```python
# append to iaf_psc_alpha_parity_test.py, same class
    def test_single_input_spike_psc_timing_matches_nest(self):
        # NEST: spike_generator -> static_synapse(w, d) -> iaf_psc_alpha (subthreshold w)
        nest.ResetKernel(); nest.resolution = 0.1
        n = nest.Create("iaf_psc_alpha", 1, params=NPAR)
        sg = nest.Create("spike_generator", params={"spike_times": [10.0]})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(sg, n, syn_spec={"weight": 50.0, "delay": 1.5})
        nest.Connect(mm, n)
        nest.Simulate(60.0)
        nest_v = np.asarray(mm.get("events")["V_m"])
        # brainpy: drive via the Simulator one-to-one event proj with same w,d.
        import jax.numpy as jnp
        from brainpy_state import iaf_psc_alpha
        from brainpy_state._network import one_to_one
        from brainpy_state._network._event_proj import EventProjection
        post = iaf_psc_alpha(1, **{k: v * (u.pF if k == 'C_m' else u.ms if 'tau' in k or k == 't_ref' else u.mV)
                                   for k, v in NPAR.items() if k != 'V_m'},
                             V_initializer=braintools.init.Constant(0. * u.mV))
        box = {'v': jnp.zeros(1)}
        proj = EventProjection(pre_spike=lambda: box['v'], n_pre_pop=1,
                               pre_local_idx=jnp.arange(1), post=post,
                               post_local_idx=jnp.arange(1), rule=one_to_one,
                               weight=50. * u.pA, delay=1.5 * u.ms)
        brainstate.nn.init_all_states(post); brainstate.nn.init_all_states(proj)
        vs = []
        for k in range(600):
            box['v'] = jnp.ones(1) if k == 100 else jnp.zeros(1)  # spike at t=10 ms
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                proj.update(); post.update()
                vs.append(float(u.get_mantissa(post.V.value[0] / u.mV)))
        bp_v = np.asarray(vs)
        # Compare the PSP peak time: must match within +/- 1 step (calibration).
        m = min(len(nest_v), len(bp_v))
        nest_peak = int(np.argmax(nest_v[:m])); bp_peak = int(np.argmax(bp_v[:m]))
        self.assertLessEqual(abs(nest_peak - bp_peak), 1,
                             f"PSP peak step NEST={nest_peak} brainpy={bp_peak}")
```

- [ ] **Step 5: Run; calibrate the delay convention if off by >1 step**

Run: `python -m pytest brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py -q`
Expected: PASS. If the PSP peak is off by a constant step count, the
emission→delivery delay is miscounted: adjust the `InputDelay` delay in
`EventProjection` (the documented `delay` vs `delay - dt` choice from spec §10) so
the peak aligns, then re-run Tasks 4–5 to confirm no regression.

- [ ] **Step 6: Commit**

```bash
git add brainpy_state/_nest/_validation/__init__.py \
        brainpy_state/_nest/_validation/iaf_psc_alpha_parity_test.py \
        brainpy_state/_nest/iaf_psc_alpha.py brainpy_state/_nest/iaf_psc_alpha_test.py \
        brainpy_state/_network/_event_proj.py
git commit -m "test(nest): iaf_psc_alpha single-neuron parity vs NEST + delay calibration"
```

---

## Task 8: `poisson_generator` rate + `spike_recorder` stamping parity

**Files:**
- Create: `brainpy_state/_nest/_validation/device_parity_test.py`
- Modify (only if a discrepancy is found): the respective model + its `*_test.py`

- [ ] **Step 1: Write the device parity tests**

```python
# brainpy_state/_nest/_validation/device_parity_test.py
import unittest
import numpy as np
import brainstate
import saiunit as u

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False


class TestPoissonRate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_empirical_rate_close_to_configured(self):
        from brainpy_state import poisson_generator
        gen = brainstate.nn.init_all_states(
            poisson_generator(2000, rate=1000. * u.Hz, rng_seed=0))
        total = 0
        steps = 10000  # 1000 ms
        for k in range(steps):
            with brainstate.environ.context(t=k * 0.1 * u.ms, i=k):
                total += int(np.sum(np.asarray(gen.update())))
        rate = total / 2000 / 1.0  # spikes / neuron / second
        self.assertLess(abs(rate - 1000.) / 1000., 0.05, f"rate {rate} Hz")


@unittest.skipUnless(_HAS_NEST, "live NEST not importable")
class TestPoissonRateVsNest(unittest.TestCase):
    def test_mean_count_matches_nest_within_tolerance(self):
        nest.ResetKernel(); nest.resolution = 0.1
        n = nest.Create("parrot_neuron", 2000)
        g = nest.Create("poisson_generator", params={"rate": 1000.0})
        sr = nest.Create("spike_recorder")
        nest.Connect(g, n, syn_spec={"delay": 0.1})
        nest.Connect(n, sr)
        nest.Simulate(1000.0)
        nest_rate = sr.n_events / 2000 / 1.0
        self.assertLess(abs(nest_rate - 1000.) / 1000., 0.05, f"nest rate {nest_rate}")


class TestSpikeRecorderStamp(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_counts_and_stamp_step(self):
        from brainpy_state import spike_recorder
        sr = brainstate.nn.init_all_states(spike_recorder())
        with brainstate.environ.context(t=1.0 * u.ms, i=10):
            sr.update(spikes=np.array([1., 0., 2.]), senders=np.array([3, 4, 5]))
        ev = sr.events
        self.assertEqual(sr.n_events, 3)                  # 1 + 0 + 2
        # stamp step = round(1.0/0.1)+1 = 11 -> time 1.1 ms
        self.assertTrue(np.allclose(np.unique(ev["times"]), [1.1]))
```

- [ ] **Step 2: Run**

Run: `python -m pytest brainpy_state/_nest/_validation/device_parity_test.py -q`
Expected: PASS / skip. If poisson rate or recorder stamping is off, fix the model
failing-test-first (add the focused failing case to
`brainpy_state/_nest/poisson_generator_test.py` or `spike_recorder_test.py`),
then fix, then re-run.

- [ ] **Step 3: Commit**

```bash
git add brainpy_state/_nest/_validation/device_parity_test.py
# plus any model + model-test files touched by a fix
git commit -m "test(nest): poisson_generator rate + spike_recorder stamp parity"
```

---

## Task 9: `examples/nest/brunel_alpha.py`

**Files:**
- Create: `examples/nest/__init__.py` (empty)
- Create: `examples/nest/brunel_alpha.py`

- [ ] **Step 1: Write the example (faithful port; default `order=400`)**

```python
# examples/nest/brunel_alpha.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) random balanced network with alpha synapses — NEST-style port.

Port of NEST's ``brunel_alpha_nest.py`` onto brainpy.state's explicit Simulator
API, driving the real ``iaf_psc_alpha`` / ``poisson_generator`` / ``spike_recorder``
models. Default ``order=400`` keeps the dense event projection memory-light; pass
a larger ``order`` once the sparse comm (EventFixedNumConn) lands.

Run:  python examples/nest/brunel_alpha.py
"""
import numpy as np
import scipy.special as sp
import saiunit as u
import braintools

from brainpy_state import iaf_psc_alpha, poisson_generator, spike_recorder
from brainpy_state.network import Simulator, fixed_indegree, all_to_all


def LambertWm1(x):
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
    return (np.exp(1.0) / (tauSyn * CMem * b)
            * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b
               - t_max * np.exp(-t_max / tauSyn)))


def build(order=400, simtime=1000.0):
    dt, delay = 0.1, 1.5
    g, eta, epsilon = 5.0, 2.0, 0.1
    NE, NI = 4 * order, 1 * order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    N_rec = 50
    tauSyn, tauMem, CMem, theta, tref = 0.5, 20.0, 250.0, 20.0, 2.0

    J = 0.1
    J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
    J_ex = J / J_unit
    J_in = -g * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    p_rate = 1000.0 * (eta * nu_th) * CE

    npar = dict(C_m=CMem * u.pF, tau_m=tauMem * u.ms, tau_syn_ex=tauSyn * u.ms,
                tau_syn_in=tauSyn * u.ms, t_ref=tref * u.ms, E_L=0. * u.mV,
                V_reset=0. * u.mV, V_th=theta * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV))

    sim = Simulator(dt=dt * u.ms)
    ne = sim.create(iaf_psc_alpha, NE, params=npar)
    ni = sim.create(iaf_psc_alpha, NI, params=npar)
    noise = sim.create(poisson_generator, rate=p_rate * u.Hz)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    sim.connect(noise, ne, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(noise, ni, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(ne, ne + ni, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CE), allow_multapses=True, seed=1)
    sim.connect(ni, ne + ni, weight=J_in * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CI), allow_multapses=True, seed=2)
    sim.connect(ne[:N_rec], esr)
    sim.connect(ni[:N_rec], isr)
    return sim, esr, isr, N_rec, simtime


def main():
    sim, esr, isr, N_rec, simtime = build()
    res = sim.simulate(simtime * u.ms)
    erate = res.rate(esr.segments[0].population)
    irate = res.rate(isr.segments[0].population)
    print("Brunel network (brainpy.state, alpha synapses)")
    print(f"  Excitatory rate : {erate:.2f} spks/s")
    print(f"  Inhibitory rate : {irate:.2f} spks/s")

    try:
        import matplotlib.pyplot as plt
        spk = np.asarray(res.spikes(esr.segments[0].population))   # (T, N_rec)
        ts, ids = np.nonzero(spk > 0)
        plt.figure(figsize=(8, 4))
        plt.scatter(ts * 0.1, ids, s=1.0, color="k")
        plt.xlabel("time (ms)"); plt.ylabel("exc neuron")
        plt.title("Brunel network — excitatory raster")
        plt.tight_layout()
        plt.savefig("examples/nest/brunel_alpha_raster.png", dpi=100)
        print("  wrote examples/nest/brunel_alpha_raster.png")
    except ImportError:
        print("  (matplotlib not installed; skipping raster)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run: `python examples/nest/brunel_alpha.py`
Expected: prints exc/inh rates (low, single-digit-to-teens Hz in the AI regime)
and writes a raster PNG. If it errors, fix the example or the Simulator; if rates
are wildly off (0 or hundreds of Hz), that is a model/wiring bug — defer the
numeric verdict to Task 10's NEST comparison.

- [ ] **Step 3: Commit**

```bash
git add examples/nest/__init__.py examples/nest/brunel_alpha.py
git commit -m "feat(examples): brunel_alpha NEST-style port on the Simulator API"
```

---

## Task 10: Network firing-rate parity vs live NEST

**Files:**
- Create: `brainpy_state/_nest/_validation/brunel_alpha_test.py`

- [ ] **Step 1: Write the network parity test (small order, skip-if-no-NEST)**

```python
# brainpy_state/_nest/_validation/brunel_alpha_test.py
import unittest
import numpy as np

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

ORDER = 200          # NE=800, NI=200 -> dense-feasible, fast
SIMTIME = 1000.0
TOL = 0.05           # 5% mean-rate parity


def _nest_rates(order, simtime):
    import scipy.special as sp
    nest.ResetKernel()
    nest.resolution = 0.1
    g, eta, epsilon, delay = 5.0, 2.0, 0.1, 1.5
    NE, NI = 4 * order, order
    CE, CI = int(epsilon * NE), int(epsilon * NI)
    N_rec = 50
    tauSyn, tauMem, CMem, theta = 0.5, 20.0, 250.0, 20.0
    npar = {"C_m": CMem, "tau_m": tauMem, "tau_syn_ex": tauSyn, "tau_syn_in": tauSyn,
            "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": theta}

    def psp(tauMem, CMem, tauSyn):
        a = tauMem / tauSyn; b = 1.0 / tauSyn - 1.0 / tauMem
        tmax = 1.0 / b * (-sp.lambertw(-np.exp(-1.0 / a) / a, k=-1).real - 1.0 / a)
        return (np.exp(1.0) / (tauSyn * CMem * b)
                * ((np.exp(-tmax / tauMem) - np.exp(-tmax / tauSyn)) / b
                   - tmax * np.exp(-tmax / tauSyn)))

    J_ex = 0.1 / psp(tauMem, CMem, tauSyn); J_in = -g * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    p_rate = 1000.0 * eta * nu_th * CE

    ne = nest.Create("iaf_psc_alpha", NE, params=npar)
    ni = nest.Create("iaf_psc_alpha", NI, params=npar)
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    esr = nest.Create("spike_recorder")
    nest.CopyModel("static_synapse", "exc", {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inh", {"weight": J_in, "delay": delay})
    nest.Connect(noise, ne, syn_spec="exc"); nest.Connect(noise, ni, syn_spec="exc")
    nest.Connect(ne[:N_rec], esr, syn_spec="exc")
    nest.Connect(ne, ne + ni, {"rule": "fixed_indegree", "indegree": CE}, "exc")
    nest.Connect(ni, ne + ni, {"rule": "fixed_indegree", "indegree": CI}, "inh")
    nest.Simulate(simtime)
    return esr.n_events / simtime * 1000.0 / N_rec


@unittest.skipUnless(_HAS_NEST, "live NEST not importable")
class TestBrunelAlphaParity(unittest.TestCase):
    def test_excitatory_rate_within_5pct_of_nest(self):
        from examples.nest.brunel_alpha import build
        sim, esr, _isr, _n, _t = build(order=ORDER, simtime=SIMTIME)
        res = sim.simulate(SIMTIME * __import__('saiunit').ms)
        bp_rate = res.rate(esr.segments[0].population)
        nest_rate = _nest_rates(ORDER, SIMTIME)
        self.assertGreater(nest_rate, 0.0)
        rel = abs(bp_rate - nest_rate) / nest_rate
        self.assertLess(rel, TOL,
                        f"exc rate brainpy={bp_rate:.2f} nest={nest_rate:.2f} "
                        f"rel={rel:.3f} > {TOL}")
```

- [ ] **Step 2: Run the parity test**

Run: `python -m pytest brainpy_state/_nest/_validation/brunel_alpha_test.py -q`
Expected: PASS (or skip if NEST missing). If the relative error exceeds 5 %, the
port has surfaced a real model/wiring bug. **Do not loosen `TOL`.** Diagnose with
superpowers:systematic-debugging: bisect by reusing Task 7/8 component tests
(single-neuron drive, poisson rate, recorder counts) to localise, fix
failing-test-first in the relevant `_nest` model or in `_network`, re-run.

- [ ] **Step 3: Make `examples` importable from the test**

Ensure `examples/__init__.py` and `examples/nest/__init__.py` exist so
`from examples.nest.brunel_alpha import build` resolves under pytest's rootdir.
Create `examples/__init__.py` (empty) if absent.

Run: `python -m pytest brainpy_state/_nest/_validation/brunel_alpha_test.py -q`
Expected: PASS / skip.

- [ ] **Step 4: Commit**

```bash
git add brainpy_state/_nest/_validation/brunel_alpha_test.py examples/__init__.py
# plus any _nest/_network fix files
git commit -m "test(nest): brunel_alpha network firing-rate parity vs live NEST"
```

---

## Task 11: Docs, changelog, full-suite green

**Files:**
- Modify: `changelog.md`, `examples/README.md`
- Create: `examples/nest/README.md`

- [ ] **Step 1: Add a changelog entry**

Prepend an entry to `changelog.md` (match the existing format) under a new
version bump, summarising: NEST-style `Simulator` API in `brainpy.state.network`;
`brunel_alpha` flagship port in `examples/nest/`; live-NEST validation harness;
and the specific `_nest` model fixes made (list each model touched in Tasks 7–8,
or "no model changes required" if all parity tests passed unmodified).

- [ ] **Step 2: Document the new example surface**

Add a "NEST-style ports" section to `examples/README.md` pointing at
`examples/nest/`, and a short `examples/nest/README.md` explaining the
`Simulator` API and the `order` scaling note.

- [ ] **Step 3: Run the full suite (no-NEST path must be green)**

Run: `python -m pytest brainpy_state/ -q`
Expected: PASS (the NEST-only parity tests skip when NEST is absent; the
component logic tests and the whole existing suite pass).

Then, with NEST available:
Run: `python -m pytest brainpy_state/_nest/_validation/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add changelog.md examples/README.md examples/nest/README.md
git commit -m "docs(network): changelog + examples docs for NEST Simulator API"
```

---

## Task 12 (follow-up, optional): sparse comm for full `order=2500`

Out of Phase-1 scope; unblocks the flagship at NEST's native size.

**Files:**
- Modify: `brainpy_state/_network/_event_proj.py`
- Test: `brainpy_state/_network/_event_proj_test.py`

- [ ] **Step 1:** Add a `comm="sparse"` path to `EventProjection` backed by
  `brainstate.nn.EventFixedNumConn(n_pre, n_post, K, conn_weight=weight,
  efferent_target='post')` (verify indegree vs outdegree orientation against a
  small `fixed_indegree` count test before wiring), keeping the dense path as the
  default for small networks.
- [ ] **Step 2:** Add a test asserting the sparse path produces the same per-post
  indegree distribution and a firing rate within 5 % of the dense path on
  `order=200`.
- [ ] **Step 3:** Switch `examples/nest/brunel_alpha.py` default to `order=2500`
  with the sparse comm; re-run Task 10 parity at `order=2500`.
- [ ] **Step 4:** Commit.

---

## Self-Review

**Spec coverage:**
- §3 API → Tasks 1–6 (Simulator/NodeView/rules/connect/simulate, top-level
  exposure). ✓
- §4.1 reuse (InputDelay, add_delta_input, samplers) → Task 4 EventProjection. ✓
- §4.3 spike capture (A) → Task 5 `_SpikeHolder` + `Simulator.update`. ✓
- §4.4 generator fan-out → Task 5 `_GeneratorSpec`, Task 8 rate parity. ✓
- §5 recorder JIT boundary → Task 5 stacked-array taps. ✓
- §6 example + validation → Tasks 9, 10. ✓ (order caveat: Task 12 covers full
  size; noted at the top.)
- §7 model-fix workflow → Tasks 7, 8 (failing-test-first). ✓
- §10 spike-timing risk → Task 7 Steps 4–5 calibration. ✓
- §11 acceptance → Tasks 9–11. ✓

**Placeholder scan:** no "TBD/handle edge cases/similar to". The only conditional
work is the *fixes* in Tasks 7–8/10, which are genuinely discovery-gated; each
provides complete, runnable parity-test code and a concrete diagnose→fix
procedure. ✓

**Type/name consistency:** `Simulator.create/connect/simulate`,
`SimulationResult.rate/n_events/spikes`, `NodeView.of/segments/size/+/[]`,
`EventProjection(pre_spike=, n_pre_pop=, pre_local_idx=, post=, post_local_idx=,
rule=, weight=, delay=)`, `_SpikeHolder.spk`, rules `all_to_all/one_to_one/
fixed_indegree(K)` are used identically across tasks. ✓
