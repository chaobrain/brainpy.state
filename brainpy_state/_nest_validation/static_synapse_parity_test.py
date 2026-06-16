# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``static_synapse`` delivery through ``EventPlasticProj``.

A deterministic spike train drives a single ``static_synapse`` edge into an
``iaf_psc_exp`` post neuron; the post ``V_m`` trace must match NEST step-for-step
(category B with a one-step recorder-alignment search). This proves the rebuilt
substrate (CSR event-matmul + ``InputDelay`` axonal seam) delivers a fixed weight
identically to NEST's ``static_synapse``.
"""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate import transform

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainpy_state
from brainpy_state import static_synapse
from brainpy_state._nest_network import EventPlasticProj
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

# Post neuron: linear, never-spiking (V_th huge) iaf_psc_exp so the comparison
# is a pure subthreshold trace.  tau_syn_ex matches the synapse PSC constant.
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)


def _bp_post():
    return brainpy_state.iaf_psc_exp(
        1, C_m=_NPAR['C_m'] * u.pF, tau_m=_NPAR['tau_m'] * u.ms,
        tau_syn_ex=_NPAR['tau_syn_ex'] * u.ms, tau_syn_in=_NPAR['tau_syn_in'] * u.ms,
        t_ref=_NPAR['t_ref'] * u.ms, E_L=_NPAR['E_L'] * u.mV,
        V_reset=_NPAR['V_reset'] * u.mV, V_th=_NPAR['V_th'] * u.mV,
        V_initializer=braintools.init.Constant(_NPAR['V_m'] * u.mV))


def _bp_vm_trace(rule, spike_steps, n_steps, dt_ms=0.1):
    """Drive ``rule`` (a single 0->0 edge) and return the post V_m trace (mV).

    The whole sweep runs inside ``brainstate.transform.for_loop`` — the per-step
    presynaptic spike is a *scanned argument* (not a Python-baked constant), so
    the loop is JIT-compiled once and the proj/delay/post States advance
    correctly.  This is the efficient drive pattern (no per-step retrace).
    """
    post = _bp_post()
    box = {'v': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(post)
    brainstate.nn.init_all_states(proj)

    spikes = np.zeros((n_steps, 1))
    spikes[np.asarray(spike_steps, dtype=int), 0] = 1.0
    spikes = jnp.asarray(spikes)
    times = jnp.arange(n_steps) * dt_ms * u.ms
    indices = jnp.arange(n_steps)

    def step(t, i, x_in):
        box['v'] = x_in
        with brainstate.environ.context(t=t, i=i):
            proj.update()
            post.update()
            return u.get_mantissa(post.V.value[0] / u.mV)

    vs = transform.for_loop(step, times, indices, spikes)
    return np.asarray(vs)


@requires_nest
class TestStaticSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_vm_trace(self, weight, delay, spike_times, T_ms):
        nest.ResetKernel()
        nest.resolution = 0.1
        n = nest.Create("iaf_psc_exp", 1, params=_NPAR)
        sg = nest.Create("spike_generator", params={"spike_times": list(spike_times)})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(sg, n, syn_spec={"weight": weight, "delay": delay})
        nest.Connect(mm, n)
        nest.Simulate(T_ms)
        return np.asarray(mm.get("events")["V_m"])

    def test_single_spike_psc_matches_nest(self):
        # spike emitted at 10 ms, delivered at 11.5 ms (delay 1.5 ms), w = 50 pA.
        T_ms = 60.0
        weight, delay = 50.0, 1.5
        nest_v = self._nest_vm_trace(weight, delay, [10.0], T_ms)
        bp_v = _bp_vm_trace(static_synapse(weight=weight * u.pA, delay=delay * u.ms),
                            spike_steps=[100], n_steps=int(round(T_ms / 0.1)))
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED, metric="static V_m").assert_()

    def test_spike_train_psc_matches_nest(self):
        # regular 20 ms train; fixed weight => identical EPSP envelope on both sides.
        T_ms = 200.0
        weight, delay = 80.0, 1.0
        train = list(np.arange(20.0, 181.0, 20.0))
        nest_v = self._nest_vm_trace(weight, delay, train, T_ms)
        steps = [int(round(t / 0.1)) for t in train]
        bp_v = _bp_vm_trace(static_synapse(weight=weight * u.pA, delay=delay * u.ms),
                            spike_steps=steps, n_steps=int(round(T_ms / 0.1)))
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED, metric="static train V_m").assert_()


if __name__ == "__main__":
    unittest.main()
