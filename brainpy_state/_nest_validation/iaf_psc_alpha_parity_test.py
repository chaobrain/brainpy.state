# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Deterministic single-neuron parity: brainpy.state iaf_psc_alpha vs live NEST."""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainpy_state
from brainpy_state._nest_network import one_to_one, EventProjection

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

# NEST parameters (NEST default units: pF, ms, mV).
NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=0.5, tau_syn_in=0.5,
            t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=20.)


def _bp_neuron(I_e):
    return brainstate.nn.init_all_states(brainpy_state.iaf_psc_alpha(
        1, C_m=250. * u.pF, tau_m=20. * u.ms, tau_syn_ex=0.5 * u.ms,
        tau_syn_in=0.5 * u.ms, t_ref=2. * u.ms, E_L=0. * u.mV,
        V_reset=0. * u.mV, V_th=20. * u.mV, I_e=I_e * u.pA,
        V_initializer=braintools.init.Constant(0. * u.mV)))


def _bp_run(I_e, T_ms):
    """Step the neuron and return (V_m trace in mV, spike count).

    The whole sweep lowers into one compiled ``brainstate.transform.for_loop``
    (CLAUDE.md rule #10): the per-step ``t``/``i`` are mapped in as ``xs`` and the
    neuron ``State`` is carried automatically; the stacked ``(spk, V_m)`` outputs are
    reduced host-side afterwards (instead of a per-step Python dispatch + device sync).
    """
    neu = _bp_neuron(I_e)
    n_steps = int(round(T_ms / 0.1))
    times = jnp.arange(n_steps) * 0.1 * u.ms
    indices = jnp.arange(n_steps)

    def _step(t, i):
        with brainstate.environ.context(t=t, i=i):
            spk = neu.update()
            return spk, neu.V.value[0]

    spks, vs = brainstate.transform.for_loop(_step, times, indices)
    spikes = int(np.sum(np.asarray(u.get_mantissa(spks)).reshape(-1) >= 0.5))
    return np.asarray(u.get_mantissa(vs / u.mV)).reshape(-1), spikes


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

    def test_subthreshold_vm_matches_nest(self):
        # 200 pA keeps V below the 20 mV threshold (steady ~16 mV).
        nest_v = self._nest_vm_trace(200.0, 100.0)
        bp_v, _ = _bp_run(200.0, 100.0)
        m = min(len(nest_v), len(bp_v))
        # Allow a one-step recorder alignment offset; compare the overlap.
        err = float(np.min([
            np.max(np.abs(nest_v[:m] - bp_v[:m])),
            np.max(np.abs(nest_v[1:m] - bp_v[:m - 1])),
            np.max(np.abs(nest_v[:m - 1] - bp_v[1:m])),
        ]))
        self.assertLess(err, 0.05, f"max |Vm| diff {err} mV exceeds 0.05 mV")

    def test_suprathreshold_spike_count_matches_nest(self):
        nest.ResetKernel()
        nest.resolution = 0.1
        n = nest.Create("iaf_psc_alpha", 1, params={**NPAR, "I_e": 400.0})
        sr = nest.Create("spike_recorder")
        nest.Connect(n, sr)
        nest.Simulate(1000.0)
        nest_count = int(sr.n_events)
        _, bp_count = _bp_run(400.0, 1000.0)
        self.assertLessEqual(abs(nest_count - bp_count), 2,
                             f"spike count NEST={nest_count} brainpy={bp_count}")

    def test_single_input_spike_psc_timing_matches_nest(self):
        # NEST: spike_generator -> static_synapse(w=50 pA, d=1.5 ms) -> iaf.
        nest.ResetKernel()
        nest.resolution = 0.1
        n = nest.Create("iaf_psc_alpha", 1, params=NPAR)
        sg = nest.Create("spike_generator", params={"spike_times": [10.0]})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
        nest.Connect(sg, n, syn_spec={"weight": 50.0, "delay": 1.5})
        nest.Connect(mm, n)
        nest.Simulate(60.0)
        nest_v = np.asarray(mm.get("events")["V_m"])

        # brainpy: same w, d via the one-to-one event projection (manual drive).
        post = brainpy_state.iaf_psc_alpha(
            1, C_m=250. * u.pF, tau_m=20. * u.ms, tau_syn_ex=0.5 * u.ms,
            tau_syn_in=0.5 * u.ms, t_ref=2. * u.ms, E_L=0. * u.mV,
            V_reset=0. * u.mV, V_th=20. * u.mV,
            V_initializer=braintools.init.Constant(0. * u.mV))
        box = {'v': jnp.zeros(1)}
        proj = EventProjection(
            pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
            post=post, post_local_idx=jnp.arange(1), rule=one_to_one,
            weight=50. * u.pA, delay=1.5 * u.ms)
        brainstate.nn.init_all_states(post)
        brainstate.nn.init_all_states(proj)
        # One compiled for_loop (CLAUDE.md rule #10): the per-step input spike is mapped
        # in as an xs array (a single spike at step 100 = t=10 ms), and the holder the
        # projection reads is set from it inside the traced body -- the data dependency
        # is established at trace time, so the delay buffer carries correctly per step.
        spikes = jnp.zeros((600, 1)).at[100].set(1.0)
        steps = jnp.arange(600)

        def _step(i, spk):
            box['v'] = spk
            with brainstate.environ.context(t=i * 0.1 * u.ms, i=i):
                proj.update()
                post.update()
                return post.V.value[0]

        bp_v = np.asarray(u.get_mantissa(
            brainstate.transform.for_loop(_step, steps, spikes) / u.mV)).reshape(-1)

        m = min(len(nest_v), len(bp_v))
        nest_peak = int(np.argmax(nest_v[:m]))
        bp_peak = int(np.argmax(bp_v[:m]))
        self.assertLessEqual(abs(nest_peak - bp_peak), 1,
                             f"PSP peak step NEST={nest_peak} brainpy={bp_peak}")
        # The PSP must be a real, positive deflection (sanity on amplitude).
        self.assertGreater(float(np.max(bp_v)), 0.1)
