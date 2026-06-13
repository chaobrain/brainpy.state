# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``bernoulli_synapse`` (static + stochastic transmission).

``bernoulli_synapse`` delivers the full ``weight`` on a presynaptic spike with
probability ``p_transmit`` and nothing otherwise (memoryless, no weight state).
Routing mirrors the stochastic-STP test: ``spike_generator -> parrot_neuron ->
bernoulli_synapse -> iaf_psc_exp``, with the brainpy side injecting at the parrot
fire steps and the ``bernoulli`` axonal delay carried by the projection's
``InputDelay``.

Three regimes, three tolerances:

* ``p_transmit = 1`` — every spike transmits, so the synapse is *deterministic*
  and identical to ``static_synapse``; the post ``V_m`` trace matches NEST
  step-for-step (category B, one-step recorder alignment).
* ``p_transmit = 0`` — nothing is ever delivered; ``V_m`` stays flat at rest on
  both sides (category B).
* ``0 < p_transmit < 1`` — NEST draws one Bernoulli per spike from a per-thread
  RNG while the rebuilt kernel draws a length-``E`` ``jax.random.uniform`` from the
  per-step key; the streams are independent, so the two are **never** compared
  per-sample. Instead several seeds per side are run and the seed-mean post ``V_m``
  is compared (category D, 5 %). The drive uses ``E`` parallel connections
  (multapses) so each spike yields ``E`` independent per-edge gates — this both
  exercises the per-edge ``ctx.key`` independence (design decision A) and shrinks
  the seed variance well under the category-D bound.
"""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate import transform

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainpy_state
from brainpy_state import bernoulli_synapse
from brainpy_state._network import EventPlasticProj
from brainpy_state._nest._validation.nest_compare import (
    compare_distributional, compare_trace, requires_nest)
from brainpy_state._nest._validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

# Post neuron: linear, never-spiking (V_th huge) iaf_psc_exp -> pure subthreshold.
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)

_DT = 0.1
_D1 = 1.0                       # spike_generator -> parrot delay
_D2 = 1.0                       # parrot -> post bernoulli delay (distributional)
# Distributional drive: 20 multapses x 25 spikes => ~500 Bernoulli arrivals/seed,
# so the per-seed mean V_m is stable to well under the 5 % category-D bound.
_P = 0.5
_E = 20
_W = 20.0
_TRAIN = list(np.arange(50.0, 50.0 + 25 * 25.0, 25.0))   # 25 spikes, 25 ms ISI
_T_SIM = 700.0
_SEEDS = (1, 2, 3, 4, 5, 6)


def _bp_post():
    return brainpy_state.iaf_psc_exp(
        1, C_m=_NPAR['C_m'] * u.pF, tau_m=_NPAR['tau_m'] * u.ms,
        tau_syn_ex=_NPAR['tau_syn_ex'] * u.ms, tau_syn_in=_NPAR['tau_syn_in'] * u.ms,
        t_ref=_NPAR['t_ref'] * u.ms, E_L=_NPAR['E_L'] * u.mV,
        V_reset=_NPAR['V_reset'] * u.mV, V_th=_NPAR['V_th'] * u.mV,
        V_initializer=braintools.init.Constant(_NPAR['V_m'] * u.mV))


def _bp_vm_trace(rule, spike_steps, n_steps, E=1, seed=None):
    """Post V_m trace (mV) driving ``E`` multapse edges; ``rng`` keyed by ``seed``.

    The sweep runs inside ``brainstate.transform.for_loop`` (the per-step spike is
    a scanned argument), so the loop is JIT-compiled once and the proj / delay /
    post States advance correctly. ``E`` parallel edges all run pre 0 -> post 0;
    the CSR event-matmul sums their per-edge effective weights.
    """
    post = _bp_post()
    box = {'v': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.zeros(E, dtype=int), post_idx=jnp.zeros(E, dtype=int), rule=rule)
    brainstate.nn.init_all_states(post)
    brainstate.nn.init_all_states(proj)
    if seed is not None and proj.rng is not None:
        proj.rng.value = jax.random.key(seed)

    spikes = np.zeros((n_steps, 1))
    spikes[np.asarray(spike_steps, dtype=int), 0] = 1.0
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


def _bp_steps(spike_times):
    """Brainpy injection steps: parrot re-emits each sg spike after ``_D1``."""
    return [int(round((t + _D1) / _DT)) for t in spike_times]


@requires_nest
class TestBernoulliSynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_vm_trace(self, weight, delay, p, spike_times, T_ms, E=1, seed=1):
        nest.ResetKernel()
        nest.resolution = _DT
        nest.rng_seed = int(seed)
        n = nest.Create("iaf_psc_exp", 1, params=_NPAR)
        pn = nest.Create("parrot_neuron")
        sg = nest.Create("spike_generator", params={"spike_times": list(spike_times)})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": _DT})
        nest.Connect(sg, pn, syn_spec={"delay": _D1})
        for _ in range(E):                       # E parallel (multapse) bernoulli edges
            nest.Connect(pn, n, syn_spec={"synapse_model": "bernoulli_synapse",
                                          "weight": weight, "delay": delay,
                                          "p_transmit": p})
        nest.Connect(mm, n)
        nest.Simulate(T_ms)
        return np.asarray(mm.get("events")["V_m"])

    # -- deterministic endpoints (exact trace) ----------------------------
    def test_p_one_matches_static(self):
        # p_transmit == 1 -> every spike transmits -> identical to static_synapse.
        T_ms, weight, delay = 60.0, 50.0, 1.5
        nest_v = self._nest_vm_trace(weight, delay, 1.0, [10.0], T_ms)
        bp_v = _bp_vm_trace(bernoulli_synapse(weight=weight * u.pA, delay=delay * u.ms,
                                              p_transmit=1.0),
                            _bp_steps([10.0]), int(round(T_ms / _DT)), E=1, seed=1)
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED,
                      metric="bernoulli p=1 V_m").assert_()

    def test_p_zero_is_flat(self):
        # p_transmit == 0 -> nothing delivered -> V_m flat at rest on both sides.
        T_ms, weight, delay = 60.0, 50.0, 1.5
        nest_v = self._nest_vm_trace(weight, delay, 0.0, [10.0, 20.0, 30.0], T_ms)
        bp_v = _bp_vm_trace(bernoulli_synapse(weight=weight * u.pA, delay=delay * u.ms,
                                              p_transmit=0.0),
                            _bp_steps([10.0, 20.0, 30.0]), int(round(T_ms / _DT)),
                            E=1, seed=1)
        m = min(len(nest_v), len(bp_v))
        self.assertLess(float(np.max(np.abs(bp_v))), 1e-9)     # genuinely flat
        compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED,
                      metric="bernoulli p=0 V_m").assert_()

    # -- partial transmission (distributional, multi-seed) ----------------
    def test_partial_transmission_distribution(self):
        n_steps = int(round(_T_SIM / _DT))
        steps = _bp_steps(_TRAIN)
        ref = [float(np.mean(self._nest_vm_trace(_W, _D2, _P, _TRAIN, _T_SIM, E=_E, seed=s)))
               for s in _SEEDS]
        cand = [float(np.mean(_bp_vm_trace(
                    bernoulli_synapse(weight=_W * u.pA, delay=_D2 * u.ms, p_transmit=_P),
                    steps, n_steps, E=_E, seed=s)))
                for s in _SEEDS]
        compare_distributional(ref, cand, tol=tc.CAT_D,
                               metric="bernoulli partial <V_m>").assert_()


if __name__ == "__main__":
    unittest.main()
