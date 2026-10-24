# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Communication kernels used by ``EventProjection``: input @ W in dense, sparse, and per-receptor form."""
from __future__ import annotations


import brainevent
import brainstate
import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainspike._nest_network.connectivity import csr_structure


class _DenseMatMul(brainstate.nn.Module):
    """Tiny comm module: input @ W.

    Delay support is deferred to a follow-up — pass ``delay=None`` only.
    """
    __module__ = 'brainspike'

    def __init__(self, weight_state: brainstate.ParamState):
        super().__init__()
        self._W = weight_state

    def update(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, x):
        W = self._W.value
        if isinstance(W, u.Quantity):
            return u.Quantity(jnp.asarray(x) @ W.mantissa, unit=W.unit)
        return jnp.asarray(x) @ W


class _SparseEventMatMul(brainstate.nn.Module):
    """Comm module: ``BinaryArray(input) @ CSR`` — event matmul over sparse edges.

    Builds a ``brainevent`` CSR (rows = pre) from sampled ``(pre_idx, post_idx)``
    edges and per-edge weights, for memory-light fan-out at large network sizes.
    Edges and weights match the dense path exactly (same sampler), so the result
    is bit-identical to ``input @ W`` on event-valued input.
    """
    __module__ = 'brainspike'

    def __init__(self, pre_idx, post_idx, w_mantissa, w_unit, *, n_pre: int, n_post: int):
        super().__init__()
        order, self._indices, self._indptr = csr_structure(pre_idx, post_idx, n_pre)
        self._shape = (int(n_pre), int(n_post))
        self._unit = w_unit
        self._data = brainstate.ParamState(jnp.asarray(np.asarray(w_mantissa)[order]))

    def update(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, x):
        csr = brainevent.CSR((self._data.value, self._indices, self._indptr), shape=self._shape)
        y = jnp.asarray(brainevent.BinaryArray(jnp.asarray(x)) @ csr)
        return u.Quantity(y, unit=self._unit) if self._unit is not u.UNITLESS else y


class _ReceptorScatter(brainstate.nn.Module):
    """Comm module: scatter per-edge spike events into ``(n_post, n_receptors)``.

    Each edge carries a target post index and a receptor port index; the per-step
    output places ``weight * pre_spike`` into ``out[post_idx, rec_idx]`` (summing
    multapses). This drives multi-receptor neurons such as
    ``iaf_psc_exp_multisynapse``, whose ports each integrate an exponential PSC
    with their own time constant ``tau_syn[r]``.
    """
    __module__ = 'brainspike'

    def __init__(self, pre_idx, post_idx, rec_idx, w_mantissa, w_unit,
                 *, n_post: int, n_receptors: int):
        super().__init__()
        self._pre_idx = jnp.asarray(np.asarray(pre_idx))
        self._post_idx = jnp.asarray(np.asarray(post_idx))
        self._rec_idx = jnp.asarray(np.asarray(rec_idx))
        self._n_post = int(n_post)
        self._n_receptors = int(n_receptors)
        self._unit = w_unit
        self._data = brainstate.ParamState(jnp.asarray(np.asarray(w_mantissa)))

    def update(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __call__(self, x):
        active = jnp.asarray(x)[self._pre_idx]               # (n_edges,) pre-spike per edge
        vals = active * self._data.value                     # (n_edges,)
        out = jnp.zeros((self._n_post, self._n_receptors), dtype=vals.dtype)
        out = out.at[self._post_idx, self._rec_idx].add(vals)
        return u.Quantity(out, unit=self._unit) if self._unit is not u.UNITLESS else out
