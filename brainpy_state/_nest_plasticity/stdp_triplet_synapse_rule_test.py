# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free spec/rule tests for the rebuilt ``stdp_triplet_synapse``.

Kernel-level checks of the Pfister-Gerstner triplet rule on the substrate's
multi-trace seam: the (fast, slow) trace tuples per side, potentiation reading
the fast pre trace ``r1`` weighted by the slow post trace ``o2``, depression
reading the fast post trace ``o1`` weighted by the slow pre trace ``r2``, the
triplet term's distinct contribution, the ``Wmax`` / ``0`` clamps, the
simultaneous-spike exclusion on every trace, and per-edge freezing.

The substrate feeds decay-then-add traces that **include** the current step's
spike, so the firing side's traces are always ``>= 1``; the kernel subtracts the
spike to recover the pre-increment value (``>= 0``). Test inputs respect that
invariant. Live-NEST equivalence is covered by the parity test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import stdp_triplet_synapse
from brainpy_state._nest_network._event_plastic import KernelContext


def _ctx(pre_spike, post_spike, r, o, E=1, t=10.0, dt=0.1):
    """Build a multi-trace KernelContext; ``r=(r1,r2)``, ``o=(o1,o2)`` per edge.

    ``r`` / ``o`` are the substrate-fed (post-increment) traces — the firing side
    includes its own +1.
    """
    g = lambda v: jnp.broadcast_to(jnp.asarray(v, float), (E,))
    pt = jnp.broadcast_to(jnp.asarray(r, float), (E, 2))
    ot = jnp.broadcast_to(jnp.asarray(o, float), (E, 2))
    return KernelContext(g(pre_spike), g(post_spike), pt[:, 0], ot[:, 0],
                         jnp.asarray(t), jnp.asarray(dt), jax.random.key(0), pt, ot)


def _host(w, pre_s, post_s, r, o, *, A2p, A3p, A2m, A3m, Wmax=100.0):
    """Reference mirroring the kernel: exclude current spike, potentiate then depress."""
    r1, r2 = r[0] - pre_s, r[1] - pre_s
    o1, o2 = o[0] - post_s, o[1] - post_s
    if post_s > 0:
        w = min(w + r1 * (A2p + A3p * o2), Wmax)
    if pre_s > 0:
        w = max(w - o1 * (A2m + A3m * r2), 0.0)
    return w


# -- spec contract ---------------------------------------------------------
def test_spec_attributes_and_defaults():
    s = stdp_triplet_synapse()
    assert s.is_homogeneous_weight is False
    assert s.stochastic is False
    assert s.edge_state_init() == {}
    assert u.get_unit(s.weight) == u.pA
    # (fast, slow) tuple per side -> two trace columns each
    assert len(s.pre_trace_tau) == 2 and len(s.post_trace_tau) == 2
    pre = [float(u.Quantity(t).to_decimal(u.ms)) for t in s.pre_trace_tau]
    post = [float(u.Quantity(t).to_decimal(u.ms)) for t in s.post_trace_tau]
    assert pre == [16.8, 101.0] and post == [20.0, 110.0]
    assert (s.Aplus, s.Aplus_triplet, s.Aminus, s.Aminus_triplet) == \
        (5e-10, 0.0062, 0.007, 0.00023)
    assert s.Wmax == 100.0


def test_validation():
    for kw in [dict(tau_plus=0 * u.ms), dict(tau_plus_triplet=-1 * u.ms),
               dict(tau_minus=0 * u.ms), dict(tau_minus_triplet=-1 * u.ms)]:
        with pytest.raises(ValueError):
            stdp_triplet_synapse(**kw)


# -- potentiation on post spike: r1 weighted by slow post trace o2 ----------
def test_post_spike_potentiates_with_triplet_term():
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.001, Aplus_triplet=0.01)
    # post spike fires -> o includes +1 (o1=1.3, o2=1.4 -> excl 0.3, 0.4); pre idle
    st, w_eff = s.update({'weight': jnp.array([5.0])},
                         _ctx(0.0, 1.0, r=(0.6, 0.2), o=(1.3, 1.4)))
    expect = _host(5.0, 0.0, 1.0, (0.6, 0.2), (1.3, 1.4),
                   A2p=0.001, A3p=0.01, A2m=0.007, A3m=0.00023)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert np.allclose(np.asarray(w_eff), [expect])
    assert expect > 5.0


def test_triplet_term_distinct_from_pair():
    # same r1, larger slow post trace o2 -> larger potentiation (the triplet term)
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.001, Aplus_triplet=0.01)
    lo, _ = s.update({'weight': jnp.array([5.0])}, _ctx(0.0, 1.0, (0.6, 0.0), (1.0, 1.1)))
    hi, _ = s.update({'weight': jnp.array([5.0])}, _ctx(0.0, 1.0, (0.6, 0.0), (1.0, 1.9)))
    assert float(np.asarray(hi['weight'])[0]) > float(np.asarray(lo['weight'])[0])


# -- depression on pre spike: o1 weighted by slow pre trace r2 --------------
def test_pre_spike_depresses_with_triplet_term():
    s = stdp_triplet_synapse(weight=5.0, Aminus=0.01, Aminus_triplet=0.02)
    # pre spike fires -> r includes +1 (r1=0.2, r2=1.3 -> excl ., 0.3); post idle (o1=0.5)
    st, _ = s.update({'weight': jnp.array([5.0])},
                     _ctx(1.0, 0.0, r=(0.2, 1.3), o=(0.5, 0.1)))
    expect = _host(5.0, 1.0, 0.0, (0.2, 1.3), (0.5, 0.1),
                   A2p=5e-10, A3p=0.0062, A2m=0.01, A3m=0.02)
    assert np.allclose(np.asarray(st['weight']), [expect])
    assert expect < 5.0


# -- clamps ----------------------------------------------------------------
def test_potentiation_clamps_at_wmax():
    s = stdp_triplet_synapse(weight=99.5, Aplus=1.0, Aplus_triplet=1.0, Wmax=100.0)
    st, _ = s.update({'weight': jnp.array([99.5])}, _ctx(0.0, 1.0, (5.0, 0.0), (1.0, 5.0)))
    assert np.allclose(np.asarray(st['weight']), [100.0])


def test_depression_floors_at_zero():
    s = stdp_triplet_synapse(weight=0.2, Aminus=1.0, Aminus_triplet=1.0)
    st, _ = s.update({'weight': jnp.array([0.2])}, _ctx(1.0, 0.0, (1.0, 5.0), (5.0, 0.0)))
    assert np.allclose(np.asarray(st['weight']), [0.0])


# -- simultaneous pre&post excludes the current step's own spike -----------
def test_simultaneous_excludes_current_step_spike():
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.01, Aplus_triplet=0.02,
                             Aminus=0.01, Aminus_triplet=0.02)
    # all traces == 1 (only this step's own spikes); exclusion -> r=o=0 -> no-op
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(1.0, 1.0, (1.0, 1.0), (1.0, 1.0)))
    assert np.allclose(np.asarray(st['weight']), [5.0])


def test_no_spike_no_change():
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.01, Aminus=0.01)
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(0.0, 0.0, (0.7, 0.7), (0.7, 0.7)))
    assert np.allclose(np.asarray(st['weight']), [5.0])


def test_frozen_non_firing_edges():
    s = stdp_triplet_synapse(weight=5.0, Aminus=0.01, Aminus_triplet=0.02)
    # edge 0 fires pre (r includes +1 -> r2 excl = 0.3); edge 1 idle -> frozen
    ctx = _ctx([1.0, 0.0], [0.0, 0.0], r=[[0.2, 1.3], [0.2, 1.3]],
               o=[[0.5, 0.1], [0.5, 0.1]], E=2)
    st, _ = s.update({'weight': jnp.array([5.0, 5.0])}, ctx)
    assert float(np.asarray(st['weight'])[0]) < 5.0       # edge 0 fired -> depressed
    assert np.allclose(np.asarray(st['weight'])[1], 5.0)  # edge 1 frozen


# -- coincident pre&post applies potentiation then depression (NEST order) --
def test_post_then_pre_order_on_coincident_step():
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.01, Aplus_triplet=0.0,
                             Aminus=0.01, Aminus_triplet=0.0)
    # both fire -> traces include +1; fast pre/post excl = 1
    st, _ = s.update({'weight': jnp.array([5.0])}, _ctx(1.0, 1.0, (2.0, 1.0), (2.0, 1.0)))
    # potentiation first: 5 + 1*0.01 = 5.01 ; then depression: 5.01 - 1*0.01 = 5.0
    assert np.allclose(np.asarray(st['weight']), [5.0])


def test_vmap_grad_smoke():
    s = stdp_triplet_synapse(weight=5.0, Aplus=0.01, Aplus_triplet=0.02)

    def run(w):
        st, _ = s.update({'weight': w}, _ctx(0.0, 1.0, (0.6, 0.2), (1.3, 1.4)))
        return jnp.sum(st['weight'])

    g = jax.grad(run)(jnp.array([5.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    out = jax.vmap(run)(jnp.array([[3.0], [5.0], [7.0]]))
    assert out.shape == (3,)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
