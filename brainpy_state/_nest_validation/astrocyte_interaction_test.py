# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Demo-port parity for ``examples/nest_like/astrocyte_interaction.py`` (goal 17b).

NEST's ``astrocyte_interaction`` is a tripartite loop: a presynaptic
``aeif_cond_alpha_astro`` projects to a postsynaptic neuron (direct EPSP) and to an
``astrocyte_lr_1994`` (raising IP3); the astrocyte's calcium crosses the SIC
threshold and feeds a slow inward current back into the postsynaptic neuron through
a ``sic_connection``.

* **Law class** (always runs, no NEST): both arms are live (spikes -> IP3 ->
  Ca>SIC_th -> I_SIC); the delta arm alone raises IP3 with no return SIC
  (``w_astro2post=0``); the SIC current measurably modulates the postsynaptic
  voltage (isolated with ``w_pre2post=0``); the loop lowers under the Simulator's
  ``for_loop`` with ``(T/dt,)`` traces.
* **Parity class** (``@requires_nest``): a *deterministic* constant-current drive
  on the presynaptic neuron (Poisson PRNG-diverges) makes the driver ``V_m``
  (``CAT_A``) and the astrocyte IP3/Ca + postsynaptic ``I_SIC`` (``ASTRO_TOL``)
  match live NEST 3.9.0 -- the 15d driven-loop parity, repackaged as the demo port.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:                                   # pragma: no cover - env dependent
    nest = None

from examples.nest_like.astrocyte_interaction import run
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance, CAT_A

DT = 0.1
SIC_DELAY_STEPS = 10
SIC_TH = 0.19669

ASTRO_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                           note='astrocyte_interaction IP3/Ca/I_SIC vs live NEST')

#: Deterministic Pillar-2 drive: a constant-current pre neuron spikes, raising IP3
#: enough (delta_IP3=2.0) for Ca to cross SIC_th and close the loop.
DET = dict(sim_time=400.0, drive='current', I_e=1000.0, delta_IP3=2.0, tau_syn_ex=2.0,
           w_pre2post=1.0, w_pre2astro=1.0, w_astro2post=1.0, conn_delay=DT,
           sic_delay_steps=SIC_DELAY_STEPS)


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


class TestAstrocyteInteractionLaw(unittest.TestCase):
    """Bidirectional-loop invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_loop_is_live_both_arms(self):
        """Driver spikes raise IP3, push Ca past SIC_th, and feed I_SIC downstream."""
        _t, v_pre, ip3, ca, isic = run(**DET)
        self.assertGreater(float(np.max(_ms(ip3))), 5.0, 'driver spikes accumulate IP3')
        self.assertGreater(float(np.max(_ms(ca))), SIC_TH, 'Ca crosses the SIC threshold')
        self.assertGreater(float(np.max(_ms(isic))), 0.0, 'SIC current is delivered')
        self.assertTrue(np.all(np.isfinite(_ms(v_pre))))

    def test_poisson_drive_spikes_pre_and_lights_the_loop(self):
        """The faithful Poisson drive (receptor 1 -> g_ex) spikes ``pre`` and lights
        the tripartite loop: IP3 climbs, Ca crosses SIC_th, and I_SIC fires.

        This is the demo's default path -- the spike->conductance fix made it live
        (pre-fix the presynaptic neuron sat at E_L and nothing propagated). Poisson
        PRNG-diverges from NEST, so this is a *law* not a parity check; a stronger
        weight + shorter window keeps it fast.
        """
        _t, v_pre, ip3, ca, isic = run(
            sim_time=2000.0, drive='poisson', poisson_rate=1500.0, poisson_weight=2.0,
            delta_IP3=0.2)
        self.assertGreater(float(np.max(_ms(ip3))), 1.0, 'Poisson spikes accumulate IP3')
        self.assertGreater(float(np.max(_ms(ca))), SIC_TH, 'Ca crosses the SIC threshold')
        self.assertGreater(float(np.max(_ms(isic))), 0.0, 'the returned SIC current fires')
        self.assertTrue(np.all(np.isfinite(_ms(v_pre))))

    def test_delta_arm_alone_raises_ip3_without_sic(self):
        """With the astro->post SIC arm decoupled, IP3 still rises but no I_SIC returns."""
        _t, _v, ip3, _ca, isic = run(**{**DET, 'w_astro2post': 0.0})
        self.assertGreater(float(np.max(_ms(ip3))), 5.0)
        self.assertTrue(np.allclose(_ms(isic), 0.0))

    def test_sic_modulates_post_voltage(self):
        """The SIC current measurably changes the postsynaptic membrane voltage.

        Isolated with ``w_pre2post=0`` so the post neuron is driven *only* by the
        astrocytic SIC: with the arm on its voltage departs from the SIC-off
        (resting) trace, and only after the SIC onset.
        """
        common = {**DET, 'w_pre2post': 0.0, 'record_v_post': True}
        v_off = _ms(run(**{**common, 'w_astro2post': 0.0})[-1])
        v_on = _ms(run(**common)[-1])
        diff = np.abs(v_on - v_off)
        self.assertGreater(float(diff.max()), 1e-2, 'SIC measurably modulates V_post')
        # Before any SIC can arrive (the sic delay), the two traces coincide.
        self.assertTrue(np.allclose(diff[:SIC_DELAY_STEPS], 0.0, atol=1e-9))

    def test_loop_lowers_with_stable_shapes(self):
        """The whole loop runs under the Simulator's for_loop with ``(T/dt,)`` traces."""
        _t, v_pre, ip3, ca, isic = run(**DET)
        n = int(round(DET['sim_time'] / DT))
        for tr in (_ms(v_pre), _ms(ip3), _ms(ca), _ms(isic)):
            self.assertEqual(tr.shape, (n,))


# --- Live-NEST parity (deterministic constant-current drive) -----------------------

def _nest_interaction_det(sim_time, I_e, delta_IP3, tau_syn_ex,
                          w_pre2post, w_pre2astro, w_astro2post, conn_delay):
    """NEST: I_e-driven pre -> {post, astro}; astro -> post sic; (V_pre, IP3, Ca, I_SIC)."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    pre = nest.Create('aeif_cond_alpha_astro', 1, params={'tau_syn_ex': tau_syn_ex, 'I_e': I_e})
    post = nest.Create('aeif_cond_alpha_astro', 1, params={'tau_syn_ex': tau_syn_ex})
    astro = nest.Create('astrocyte_lr_1994', params={'delta_IP3': delta_IP3})
    nest.Connect(pre, post, syn_spec={'weight': w_pre2post, 'delay': conn_delay})
    nest.Connect(pre, astro, syn_spec={'weight': w_pre2astro, 'delay': conn_delay})
    nest.Connect(astro, post, syn_spec={'synapse_model': 'sic_connection', 'weight': w_astro2post})
    mm_pre = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': DT})
    mm_a = nest.Create('multimeter', params={'record_from': ['IP3', 'Ca_astro'], 'interval': DT})
    mm_p = nest.Create('multimeter', params={'record_from': ['I_SIC'], 'interval': DT})
    nest.Connect(mm_pre, pre)
    nest.Connect(mm_a, astro)
    nest.Connect(mm_p, post)
    nest.Simulate(sim_time)
    return (np.asarray(mm_pre.events['V_m'], dtype=float),
            np.asarray(mm_a.events['IP3'], dtype=float),
            np.asarray(mm_a.events['Ca_astro'], dtype=float),
            np.asarray(mm_p.events['I_SIC'], dtype=float))


@requires_nest
class TestAstrocyteInteractionParity(unittest.TestCase):
    """The driven neuron<->astro loop (V/IP3/Ca/I_SIC) matches live NEST."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_loop_traces_match_nest(self):
        """V_pre (CAT_A) and IP3/Ca/I_SIC (ASTRO_TOL) all track NEST."""
        n_v, n_ip3, n_ca, n_isic = _nest_interaction_det(
            DET['sim_time'], DET['I_e'], DET['delta_IP3'], DET['tau_syn_ex'],
            DET['w_pre2post'], DET['w_pre2astro'], DET['w_astro2post'], DET['conn_delay'])
        _t, b_v, b_ip3, b_ca, b_isic = run(**DET)
        b_v, b_ip3, b_ca, b_isic = _ms(b_v), _ms(b_ip3), _ms(b_ca), _ms(b_isic)
        self.assertGreater(float(np.max(n_isic)), 1.0)       # SIC arm fired in the loop
        nv = min(n_v.size, b_v.size)
        compare_trace(n_v[:nv], b_v[:nv], tol=CAT_A, metric='V_pre').assert_()
        for nm, ref, cand in (('IP3', n_ip3, b_ip3), ('Ca', n_ca, b_ca), ('I_SIC', n_isic, b_isic)):
            n = min(ref.size, cand.size)
            compare_trace(ref[:n], cand[:n], tol=ASTRO_TOL, metric=nm).assert_()


if __name__ == '__main__':
    unittest.main()
