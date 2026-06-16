# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: short-term-plasticity PSC-amplitude trains.

The three deterministic STP models rebuilt on ``EventPlasticProj`` —
``tsodyks_synapse``, ``tsodyks_synapse_hom`` and ``tsodyks2_synapse`` — are
driven by a regular spike burst plus a recovery pair into an ``iaf_psc_exp``
post neuron, and the post ``V_m`` (the EPSP-amplitude train carrying the
depression / facilitation envelope) must match NEST step-for-step within
category B with a one-step recorder-alignment search.

NEST routing notes faithfully reproduced here:

* plastic synapses cannot be driven directly by a device, so the spike train
  goes ``spike_generator -> parrot_neuron -> synapse`` (the parrot relays each
  spike one ``delay`` later; the inter-spike intervals — which drive the STP
  state — are preserved). The brainpy side injects spikes at the parrot's
  fire steps so ``h = t_now - t_lastspike`` matches.
* ``tsodyks_synapse_hom`` stores ``tau_psc / tau_rec / tau_fac / U`` as *common*
  (homogeneous) synapse properties, set via ``SetDefaults``; only
  ``weight / delay / x / u`` are per-connection.

Parameters follow ``pynest/examples/evaluate_tsodyks2_synapse.py`` (depression:
``U=0.67, tau_rec=450, tau_fac=0``; facilitation: ``U=0.1, tau_rec=100,
tau_fac=1000``; ``weight=250 pA``; post ``tau_syn_ex = tau_psc = 3 ms``).
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
brainstate.environ.set(precision=64, platform='cpu')

import brainpy_state
from brainpy_state import tsodyks_synapse, tsodyks_synapse_hom, tsodyks2_synapse
from brainpy_state._nest_network import EventPlasticProj
from brainpy_state._nest_validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

# Linear, never-spiking post neuron; tau_syn_ex == tau_psc so the PSC shape is
# the synapse's.  V_th huge => pure subthreshold EPSP train.
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)

_DT = 0.1
_D1 = 1.0   # spike_generator -> parrot relay delay (ms)
_D2 = 1.0   # parrot -> synapse axonal delay (ms)

# Spike protocol: a 50 ms-ISI burst then a recovery pair.
_TRAIN = [50., 100., 150., 200., 250., 300., 350., 400., 650., 700.]
_T_SIM = 800.0

# weight (CommonPropertiesHomW) + tau_psc/tau_rec/tau_fac/U are common
# (homogeneous) properties of ``tsodyks_synapse_hom``; delay/x/u stay per-conn.
_HOM_COMMON = ('weight', 'tau_psc', 'tau_rec', 'tau_fac', 'U')

_DEP = dict(weight=250., U=0.67, u=0.67, x=1.0, tau_rec=450., tau_fac=0.0)
_FAC = dict(weight=250., U=0.1, u=0.1, x=1.0, tau_rec=100., tau_fac=1000.)


def _bp_post():
    return brainpy_state.iaf_psc_exp(
        1, C_m=_NPAR['C_m'] * u.pF, tau_m=_NPAR['tau_m'] * u.ms,
        tau_syn_ex=_NPAR['tau_syn_ex'] * u.ms, tau_syn_in=_NPAR['tau_syn_in'] * u.ms,
        t_ref=_NPAR['t_ref'] * u.ms, E_L=_NPAR['E_L'] * u.mV,
        V_reset=_NPAR['V_reset'] * u.mV, V_th=_NPAR['V_th'] * u.mV,
        V_initializer=braintools.init.Constant(_NPAR['V_m'] * u.mV))


def _bp_vm_trace(rule):
    """Drive a single 0->0 plastic edge with the protocol; return post V_m (mV)."""
    post = _bp_post()
    box = {'v': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(post)
    brainstate.nn.init_all_states(proj)

    n_steps = int(round(_T_SIM / _DT))
    spikes = np.zeros((n_steps, 1))
    for t in _TRAIN:                       # inject at parrot fire steps (t + D1)
        spikes[int(round((t + _D1) / _DT)), 0] = 1.0
    spikes = jnp.asarray(spikes)
    times = jnp.arange(n_steps) * _DT * u.ms
    indices = jnp.arange(n_steps)

    def step(t, i, x_in):
        box['v'] = x_in
        with brainstate.environ.context(t=t, i=i):
            proj.update()
            post.update()
            return u.get_mantissa(post.V.value[0] / u.mV)

    return np.asarray(transform.for_loop(step, times, indices, spikes))


@requires_nest
class TestStpParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_vm_trace(self, model, syn):
        nest.ResetKernel()
        nest.resolution = _DT
        neuron = nest.Create("iaf_psc_exp", 1, params=_NPAR)
        pn = nest.Create("parrot_neuron")
        sg = nest.Create("spike_generator", params={"spike_times": list(_TRAIN)})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": _DT})
        nest.Connect(sg, pn, syn_spec={"delay": _D1})
        conn = {"synapse_model": model, "delay": _D2}
        if model == "tsodyks_synapse_hom":
            nest.SetDefaults(model, {k: syn[k] for k in _HOM_COMMON if k in syn})
            conn.update({k: v for k, v in syn.items() if k not in _HOM_COMMON})
        else:
            conn.update(syn)
        nest.Connect(pn, neuron, syn_spec=conn)
        nest.Connect(mm, neuron)
        nest.Simulate(_T_SIM)
        return np.asarray(mm.get("events")["V_m"])

    def _run(self, model, rule, syn, label):
        nest_v = self._nest_vm_trace(model, syn)
        bp_v = _bp_vm_trace(rule)
        m = min(len(nest_v), len(bp_v))
        res = compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED, metric=label)
        res.assert_()

    # ----- tsodyks_synapse (expm1 propagator form) -----------------------
    def test_tsodyks_depression(self):
        self._run("tsodyks_synapse",
                  tsodyks_synapse(weight=250. * u.pA, delay=_D2 * u.ms,
                                  tau_psc=3. * u.ms, tau_rec=450. * u.ms,
                                  tau_fac=0. * u.ms, U=0.67, u=0.67, x=1.0),
                  {**_DEP, "tau_psc": 3.0}, "tsodyks dep V_m")

    def test_tsodyks_facilitation(self):
        self._run("tsodyks_synapse",
                  tsodyks_synapse(weight=250. * u.pA, delay=_D2 * u.ms,
                                  tau_psc=3. * u.ms, tau_rec=100. * u.ms,
                                  tau_fac=1000. * u.ms, U=0.1, u=0.1, x=1.0),
                  {**_FAC, "tau_psc": 3.0}, "tsodyks fac V_m")

    # ----- tsodyks_synapse_hom (plain-exp propagator; common params) -----
    def test_tsodyks_hom_depression(self):
        self._run("tsodyks_synapse_hom",
                  tsodyks_synapse_hom(weight=250. * u.pA, delay=_D2 * u.ms,
                                      tau_psc=3. * u.ms, tau_rec=450. * u.ms,
                                      tau_fac=0. * u.ms, U=0.67, u=0.67, x=1.0),
                  {**_DEP, "tau_psc": 3.0}, "tsodyks_hom dep V_m")

    def test_tsodyks_hom_facilitation(self):
        self._run("tsodyks_synapse_hom",
                  tsodyks_synapse_hom(weight=250. * u.pA, delay=_D2 * u.ms,
                                      tau_psc=3. * u.ms, tau_rec=100. * u.ms,
                                      tau_fac=1000. * u.ms, U=0.1, u=0.1, x=1.0),
                  {**_FAC, "tau_psc": 3.0}, "tsodyks_hom fac V_m")

    # ----- tsodyks2_synapse (2-variable; weight = x*u*w) ------------------
    def test_tsodyks2_depression(self):
        self._run("tsodyks2_synapse",
                  tsodyks2_synapse(weight=250. * u.pA, delay=_D2 * u.ms,
                                   U=0.67, u=0.67, x=1.0, tau_rec=450. * u.ms,
                                   tau_fac=0. * u.ms),
                  dict(_DEP), "tsodyks2 dep V_m")

    def test_tsodyks2_facilitation(self):
        self._run("tsodyks2_synapse",
                  tsodyks2_synapse(weight=250. * u.pA, delay=_D2 * u.ms,
                                   U=0.1, u=0.1, x=1.0, tau_rec=100. * u.ms,
                                   tau_fac=1000. * u.ms),
                  dict(_FAC), "tsodyks2 fac V_m")


if __name__ == "__main__":
    unittest.main()
