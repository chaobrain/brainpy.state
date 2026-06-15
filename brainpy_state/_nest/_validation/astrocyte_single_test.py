# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Demo-port parity for ``examples/nest/astrocyte_single.py`` (goal 17b).

NEST's ``astrocyte_single`` drives one ``astrocyte_lr_1994`` with a Poisson spike
train and records IP3 / cytosolic calcium. NEST's astrocyte exposes no ``SIC``
recordable (only ``IP3`` / ``Ca_astro`` / ``h_IP3R``), so to surface the slow
inward current this port adds one downstream ``aeif_cond_alpha_astro`` connected by
a ``sic_connection`` and records its ``I_SIC``.

* **Law class** (always runs, no NEST): IP3 climbs with the spike train then
  relaxes; ``I_SIC`` is zero until the ``sic_connection`` delay then equals
  ``ln((Ca - SIC_th) * 1000)``; ``sic_weight=0`` and sub-threshold Ca both give
  ``I_SIC == 0``; the astrocyte's own ``SIC`` recordable equals the downstream
  ``I_SIC`` shifted by ``delay_steps``; the whole model lowers under the
  Simulator's ``for_loop`` with ``(T/dt,)`` traces.
* **Parity class** (``@requires_nest``): a *deterministic* ``spike_generator`` drive
  (Poisson PRNG-diverges) makes IP3 / Ca / I_SIC match live NEST 3.9.0 within
  ``ASTRO_TOL`` -- the 15d SIC-response micro-parity, repackaged as the demo port.
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

from examples.nest.astrocyte_single import run
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

DT = 0.1
SIC_DELAY_STEPS = 10
SIC_TH = 0.19669

#: IP3/Ca/I_SIC trace tolerance (15d ASTRO_TOL): IP3/Ca dimensionless (µM), I_SIC
#: pA, so atol is a plain float; align_steps=3 absorbs the spike->IP3 + sic-delivery
#: integer pipeline offset.
ASTRO_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                           note='astrocyte_single IP3/Ca/I_SIC vs live NEST')

#: Deterministic Pillar-1 drive: Ca initialised at 1.0 µM (>> SIC_th) so the
#: astrocyte emits a graded SIC every step and the pathway is exercised.
DET = dict(sim_time=60.0, spike_times=[5., 6., 7., 8., 9., 10.], spike_weight=2.0,
           delta_IP3=0.5, ip3_init=1.0, ca_init=1.0, h_init=1.0,
           sic_weight=1.0, sic_delay_steps=SIC_DELAY_STEPS)


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float).reshape(-1)


class TestAstrocyteSingleLaw(unittest.TestCase):
    """SIC-response invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        # Each run() is a fresh trace+compile; bound the JAX cache per test.
        jax.clear_caches()
        gc.collect()

    def test_ip3_rises_then_relaxes_and_sic_onsets_at_delay(self):
        """IP3 climbs with the spike train then relaxes; SIC onsets at the delay."""
        _t, ip3, ca, isic = run(**DET)
        ip3, isic = _ms(ip3), _ms(isic)
        self.assertAlmostEqual(float(ip3[0]), 1.0, places=2)
        self.assertGreater(float(np.max(ip3)), 6.0)          # train accumulates IP3
        self.assertLess(float(ip3[-1]), float(np.max(ip3)))  # relaxes after the train
        nz = np.flatnonzero(isic > 0.0)
        self.assertTrue(nz.size > 0, 'SIC must be delivered for Ca above threshold')
        self.assertEqual(int(nz[0]), SIC_DELAY_STEPS,
                         'first nonzero I_SIC lands at the sic-connection delay')
        self.assertAlmostEqual(float(isic[nz[0]]),
                               float(np.log((1.0 - SIC_TH) * 1000.0)), places=2)

    def test_weight_zero_sic_is_decoupled(self):
        """``sic_weight=0`` delivers no current downstream (decoupled astro)."""
        _t, _ip3, _ca, isic = run(**{**DET, 'sic_weight': 0.0})
        self.assertTrue(np.allclose(_ms(isic), 0.0))

    def test_subthreshold_ca_emits_no_sic(self):
        """Ca held below ``SIC_th`` (no drive, resting init) emits no SIC."""
        _t, _ip3, ca, isic = run(sim_time=20.0, spike_times=[], spike_weight=2.0,
                                 delta_IP3=0.5, ip3_init=0.16, ca_init=0.073, h_init=0.793,
                                 sic_weight=1.0, sic_delay_steps=SIC_DELAY_STEPS)
        self.assertLess(float(np.max(_ms(ca))), SIC_TH)
        self.assertTrue(np.allclose(_ms(isic), 0.0))

    def test_astro_sic_recordable_matches_downstream_isic(self):
        """The astrocyte's own ``SIC`` recordable == downstream ``I_SIC`` shifted by the delay."""
        _t, _ip3, _ca, isic, astro_sic = run(**DET, return_astro_sic=True)
        isic, astro_sic = _ms(isic), _ms(astro_sic)
        d = SIC_DELAY_STEPS
        np.testing.assert_allclose(isic[d:d + 20], astro_sic[:20], atol=1e-6)

    def test_poisson_drive_path_accumulates_ip3(self):
        """The default Poisson drive (``spike_times=None``) runs end-to-end: a
        fixed-seed high-rate train climbs IP3 far above baseline.

        Exercises the demo's actual ``poisson_generator`` branch (the parity test uses
        a deterministic ``spike_generator``); Poisson PRNG-diverges from NEST so this is
        a law, not a parity check.
        """
        _t, ip3, _ca, _isic = run(sim_time=80.0, poisson_rate=2000.0, poisson_weight=1.0,
                                  delta_IP3=0.5, seed=1)
        ip3 = _ms(ip3)
        self.assertEqual(ip3.shape, (int(round(80.0 / DT)),))
        self.assertTrue(np.all(np.isfinite(ip3)))
        self.assertGreater(float(np.max(ip3)), 5.0, 'Poisson spikes accumulate IP3')

    def test_loop_lowers_with_stable_trace_shapes(self):
        """The whole model runs under the Simulator's for_loop with ``(T/dt,)`` traces."""
        _t, ip3, ca, isic = run(**DET)
        n = int(round(DET['sim_time'] / DT))
        for tr in (_ms(ip3), _ms(ca), _ms(isic)):
            self.assertEqual(tr.shape, (n,))


# --- Live-NEST parity (deterministic spike drive) ----------------------------------

def _nest_single_det(spike_times, spike_weight, delta_IP3, sim_time):
    """NEST: spike_generator -> astro -> sic -> post; (IP3, Ca, I_SIC) traces."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT})
    astro = nest.Create('astrocyte_lr_1994',
                        params={'delta_IP3': delta_IP3, 'IP3': 1.0, 'Ca_astro': 1.0, 'h_IP3R': 1.0})
    post = nest.Create('aeif_cond_alpha_astro', 1)
    sg = nest.Create('spike_generator',
                     params={'spike_times': list(spike_times),
                             'spike_weights': [spike_weight] * len(spike_times)})
    mm_a = nest.Create('multimeter', params={'record_from': ['IP3', 'Ca_astro'], 'interval': DT})
    mm_p = nest.Create('multimeter', params={'record_from': ['I_SIC'], 'interval': DT})
    nest.Connect(sg, astro, syn_spec={'weight': 1.0, 'delay': DT})
    nest.Connect(astro, post, syn_spec={'synapse_model': 'sic_connection', 'weight': 1.0})
    nest.Connect(mm_a, astro)
    nest.Connect(mm_p, post)
    nest.Simulate(sim_time)
    return (np.asarray(mm_a.events['IP3'], dtype=float),
            np.asarray(mm_a.events['Ca_astro'], dtype=float),
            np.asarray(mm_p.events['I_SIC'], dtype=float))


@requires_nest
class TestAstrocyteSingleParity(unittest.TestCase):
    """Deterministic SIC pathway (IP3/Ca/I_SIC) matches live NEST per-sample."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_ip3_ca_isic_match_nest(self):
        """IP3, Ca, and the delivered ``I_SIC`` track NEST within ``ASTRO_TOL``."""
        n_ip3, n_ca, n_isic = _nest_single_det(DET['spike_times'], DET['spike_weight'],
                                               DET['delta_IP3'], DET['sim_time'])
        _t, b_ip3, b_ca, b_isic = run(**DET)
        b_ip3, b_ca, b_isic = _ms(b_ip3), _ms(b_ca), _ms(b_isic)
        self.assertGreater(float(np.max(n_isic)), 1.0)       # SIC actually exercised
        for nm, ref, cand in (('IP3', n_ip3, b_ip3), ('Ca', n_ca, b_ca), ('I_SIC', n_isic, b_isic)):
            n = min(ref.size, cand.size)
            compare_trace(ref[:n], cand[:n], tol=ASTRO_TOL, metric=nm).assert_()


if __name__ == '__main__':
    unittest.main()
