# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``quantal_stp_synapse`` (stochastic) — distributional.

``quantal_stp_synapse`` releases an integer number of quanta per spike
(binomial recovery of depleted sites, then binomial release of available
sites). NEST draws one Bernoulli per release site while the rebuilt kernel
draws a single ``jax.random.binomial`` — distributionally identical but with
*independent* PRNG streams, so the two are **never** compared per-sample.
Instead we run several seeds per side and compare the seed-mean post
depolarization (mean ``V_m``), the category-D protocol (relative tolerance 5 %).

Routing mirrors the deterministic STP test: ``spike_generator -> parrot_neuron
-> quantal_stp_synapse -> iaf_psc_exp`` with the brainpy side injecting at the
parrot fire steps. Per seed, NEST's kernel ``rng_seed`` and the projection's
``rng`` State are set independently.
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
from brainpy_state import quantal_stp_synapse
from brainpy_state._nest_network import EventPlasticProj
from brainpy_state._nest_validation.nest_compare import compare_distributional, requires_nest
from brainpy_state._nest_validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)

_DT = 0.1
_D1 = 1.0
_D2 = 1.0
# Many sites + a long fast train => total released quanta in the thousands, so
# the per-seed mean V_m is stable to well under the 5 % category-D bound.
_N_SITES = 30
_TRAIN = list(np.arange(50.0, 50.0 + 30 * 15.0, 15.0))   # 30 spikes, 15 ms ISI
_T_SIM = 700.0
_SEEDS = (1, 2, 3, 4, 5, 6)

_DEP = dict(weight=60., U=0.5, tau_rec=150., tau_fac=0.0)
_FAC = dict(weight=60., U=0.15, tau_rec=120., tau_fac=500.0)


def _bp_post():
    return brainpy_state.iaf_psc_exp(
        1, C_m=_NPAR['C_m'] * u.pF, tau_m=_NPAR['tau_m'] * u.ms,
        tau_syn_ex=_NPAR['tau_syn_ex'] * u.ms, tau_syn_in=_NPAR['tau_syn_in'] * u.ms,
        t_ref=_NPAR['t_ref'] * u.ms, E_L=_NPAR['E_L'] * u.mV,
        V_reset=_NPAR['V_reset'] * u.mV, V_th=_NPAR['V_th'] * u.mV,
        V_initializer=braintools.init.Constant(_NPAR['V_m'] * u.mV))


def _bp_mean_vm(rule_factory, seed):
    """Mean post V_m (mV) for one seed; fresh proj/post, ``rng`` keyed by seed."""
    post = _bp_post()
    box = {'v': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule_factory())
    brainstate.nn.init_all_states(post)
    brainstate.nn.init_all_states(proj)
    proj.rng.value = jax.random.key(seed)

    n_steps = int(round(_T_SIM / _DT))
    spikes = np.zeros((n_steps, 1))
    for t in _TRAIN:
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

    return float(np.mean(np.asarray(transform.for_loop(step, times, indices, spikes))))


@requires_nest
class TestQuantalStpParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_mean_vm(self, syn, seed):
        nest.ResetKernel()
        nest.resolution = _DT
        nest.rng_seed = int(seed)
        neuron = nest.Create("iaf_psc_exp", 1, params=_NPAR)
        pn = nest.Create("parrot_neuron")
        sg = nest.Create("spike_generator", params={"spike_times": list(_TRAIN)})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": _DT})
        nest.Connect(sg, pn, syn_spec={"delay": _D1})
        # NEST's set_status leaves a_ and u_ at their constructor defaults unless
        # given explicitly, so we pin both to the rebuilt kernel's defaults:
        #   * a = n  -> all sites start available (NEST default a_ = 1).  Without
        #     this the first spike, which dominates a depressing train, diverges.
        #   * u = U  -> NEST keeps u_ at the *old* 0.5 default (u_(U_) used the
        #     0.5 default U_) and does NOT re-derive it when only 'U' is set, so
        #     the facilitation regime (U=0.15) would otherwise start at u=0.5 and
        #     bias ~4 % vs the kernel's u0 = U.
        nest.Connect(pn, neuron, syn_spec={"synapse_model": "quantal_stp_synapse",
                                           "delay": _D2, "n": _N_SITES,
                                           "a": _N_SITES, "u": syn["U"], **syn})
        nest.Connect(mm, neuron)
        nest.Simulate(_T_SIM)
        return float(np.mean(np.asarray(mm.get("events")["V_m"])))

    def _run(self, syn, rule_factory, label):
        ref = [self._nest_mean_vm(syn, s) for s in _SEEDS]
        cand = [_bp_mean_vm(rule_factory, s) for s in _SEEDS]
        compare_distributional(ref, cand, tol=tc.CAT_D, metric=label).assert_()

    def test_quantal_depression_distribution(self):
        self._run(_DEP,
                  lambda: quantal_stp_synapse(
                      weight=60. * u.pA, delay=_D2 * u.ms, n=_N_SITES, U=0.5,
                      tau_rec=150. * u.ms, tau_fac=0. * u.ms),
                  "quantal dep <V_m>")

    def test_quantal_facilitation_distribution(self):
        self._run(_FAC,
                  lambda: quantal_stp_synapse(
                      weight=60. * u.pA, delay=_D2 * u.ms, n=_N_SITES, U=0.15,
                      tau_rec=120. * u.ms, tau_fac=500. * u.ms),
                  "quantal fac <V_m>")


if __name__ == "__main__":
    unittest.main()
