# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Seam test: presynaptically-integrated short-term plasticity (``iaf_tum_2000``).

``iaf_tum_2000`` integrates Tsodyks-Markram short-term plasticity *inside the
presynaptic neuron*: on each of its own spikes it releases an efficacy
``spike_offset = u * x`` (the depressing/facilitating fraction) rather than a
unit spike. A NEST ``static_synapse`` with ``receptor_type=1`` (TSODYKS) carries
that graded efficacy to the post, which integrates ``weight * efficacy`` as a
post-synaptic current.

A plain neuron->neuron connection in the Simulator delivers ``weight * spike``
(the binarised spike holder), which would throw the efficacy away. The seam
under test routes a ``receptor_type=1`` connection from an ``_emission_attr``
neuron through a dedicated *emission holder* so it delivers ``weight * efficacy``
instead -- per-connection, so the neuron's other (binary) connections are
unaffected.

These tests are NEST-free. The oracle is the manual three-phase reference
(run the pre, delay ``weight * spike_offset``, drive the post via its native
``_w_ex_jnp`` seam) already validated against ``iaf_psc_exp + tsodyks_synapse``
in ``iaf_tum_2000_test.py``; live-NEST parity is covered by
``_validation/iaf_tum_2000_stp_test.py``.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import brainstate
import brainunit as u

from brainpy_state import (Simulator, iaf_tum_2000, iaf_psc_exp, dc_generator,
                           voltmeter)

# Shared neuron parameters (the validated reference config in iaf_tum_2000_test).
_COMMON = dict(
    E_L=-70.0 * u.mV, C_m=250.0 * u.pF, tau_m=10.0 * u.ms, t_ref=2.0 * u.ms,
    V_th=-55.0 * u.mV, V_reset=-70.0 * u.mV, tau_syn_ex=2.0 * u.ms,
    tau_syn_in=2.0 * u.ms, I_e=0.0 * u.pA, rho=0.01 / u.second, delta=0.0 * u.mV,
)
_STP = dict(tau_psc=2.0 * u.ms, tau_rec=400.0 * u.ms, tau_fac=1000.0 * u.ms, U=0.5)
_DT = 0.1                  # ms
_DELAY_STEPS = 10          # delay = 1.0 ms
_WEIGHT = 250.0            # pA
_DC_AMP = 500.0            # pA
_T = 60.0                  # ms


def _reference_post_vm():
    """Manual three-phase reference: post V_m (mV) for a TSODYKS pre->post pair.

    Mirrors ``iaf_tum_2000_test.test_matches_iaf_psc_exp_plus_tsodyks_synapse_
    equivalence``: run the DC-driven pre, build the delayed ``weight * spike_offset``
    train, drive the post through its native ``_w_ex_jnp`` weight seam. Pure delay
    of ``_DELAY_STEPS`` (no holder lag) -- the Simulator's one-step holder lag is
    absorbed by the alignment in the comparison.
    """
    dt = _DT * u.ms
    n_steps = int(round(_T / _DT))
    with brainstate.environ.context(dt=dt):
        pre = iaf_tum_2000(1, **_COMMON, **_STP, x=0.0, y=0.0, u=0.0)
        post = iaf_tum_2000(1, **_COMMON, **_STP)
        pre.init_state()
        post.init_state()
        dc = jnp.array([_DC_AMP if (5.0 <= k * _DT < 45.0) else 0.0
                        for k in range(n_steps)], dtype=jnp.float64) * u.pA

        def _pre_step(k):
            with brainstate.environ.context(t=k * dt):
                spk = pre.update(x=dc[k])
            return spk, pre.spike_offset.value

        spk, off = brainstate.transform.for_loop(_pre_step, jnp.arange(n_steps))
        tum_spks = jnp.asarray(spk)[:, 0]
        tum_offsets = jnp.asarray(off)[:, 0]
        padded = jnp.concatenate([jnp.zeros(_DELAY_STEPS), tum_offsets])
        padded_spk = jnp.concatenate([jnp.zeros(_DELAY_STEPS), tum_spks])
        w_ex = jnp.where(padded_spk[:n_steps] > 0.0,
                         _WEIGHT * padded[:n_steps], 0.0)

        def _post_step(k):
            w_ex_arr = jnp.broadcast_to(w_ex[k], post.varshape)
            w_in_arr = jnp.zeros(post.varshape, dtype=jnp.float64)
            with brainstate.environ.context(t=k * dt):
                post.update(_w_ex_jnp=w_ex_arr, _w_in_jnp=w_in_arr)
            return post.V.value / u.mV

        v = brainstate.transform.for_loop(_post_step, jnp.arange(n_steps))
    return np.asarray(v)[:, 0]


def _sim_post_vm(receptor_type=1, post_cls=iaf_tum_2000):
    """Simulator post V_m (mV) for a DC-driven TSODYKS pre -> post connection."""
    sim = Simulator(dt=_DT * u.ms)
    pre = sim.create(iaf_tum_2000, 1, params={**_COMMON, **_STP, 'x': 0.0, 'y': 0.0, 'u': 0.0})
    post = sim.create(post_cls, 1, params={**_COMMON, **_STP} if post_cls is iaf_tum_2000
                      else {**_COMMON})
    dc = sim.create(dc_generator, amplitude=_DC_AMP * u.pA, start=5.0 * u.ms, stop=45.0 * u.ms)
    vm = sim.create(voltmeter)
    sim.connect(dc, pre)
    sim.connect(pre, post, receptor_type=receptor_type, weight=_WEIGHT * u.pA,
                delay=_DELAY_STEPS * _DT * u.ms)
    sim.connect(vm, post)
    res = sim.simulate(_T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)


def _best_aligned_residual(ref, cand, max_shift=2):
    """Min RMS over integer shifts in [0, max_shift] (cand delivered >= ref latency).

    Returns (best_shift, rms). Absorbs the Simulator's fixed one-step holder lag
    relative to the pure-delay reference (cluster-11 align_steps convention).
    """
    best = (0, np.inf)
    for s in range(max_shift + 1):
        a = ref[:len(ref) - s] if s else ref
        b = cand[s:] if s else cand
        n = min(len(a), len(b))
        rms = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
        if rms < best[1]:
            best = (s, rms)
    return best


class TestStpEmissionSeam(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        brainstate.environ.set(precision=64, platform='cpu')

    def test_stp_emission_matches_three_phase_reference(self):
        # The Simulator's receptor_type=1 delivery reproduces the validated manual
        # reference (post V_m) to machine precision, up to the one-step holder lag.
        ref = _reference_post_vm()
        sim = _sim_post_vm(receptor_type=1)
        self.assertEqual(len(ref), len(sim))
        shift, rms = _best_aligned_residual(ref, sim)
        self.assertLessEqual(shift, 1, f'unexpected latency offset {shift} steps')
        self.assertLess(rms, 1e-6, f'post V_m diverged from reference (rms={rms})')

    def test_delivers_efficacy_not_binary_spike(self):
        # receptor_type=1 delivers weight * (u*x) with u*x < 1, so the post is
        # depolarised LESS than a binary (weight * 1) delivery -- but still > rest.
        v_eff = _sim_post_vm(receptor_type=1)        # graded efficacy
        v_bin = _sim_post_vm(receptor_type=None)     # plain binary spike
        rest = _COMMON['E_L'] / u.mV
        # both deliver excitatory input (depolarise above rest)
        self.assertGreater(v_eff.max(), float(rest) + 1e-3)
        self.assertGreater(v_bin.max(), float(rest) + 1e-3)
        # efficacy (u*x in (0,1)) scales the EPSP down vs the unit spike
        self.assertLess(v_eff.max() - float(rest), v_bin.max() - float(rest))

    def test_tsodyks_receptor_requires_matching_post(self):
        # A TSODYKS (receptor_type=1) connection carries presynaptic efficacy and
        # requires an iaf_tum_2000 post; a mismatched post is a clear error, not a
        # silent fallthrough into the receptor-port path (iaf_psc_exp has no ports).
        with self.assertRaisesRegex(ValueError, 'iaf_tum_2000'):
            _sim_post_vm(receptor_type=1, post_cls=iaf_psc_exp)


if __name__ == '__main__':
    unittest.main()
