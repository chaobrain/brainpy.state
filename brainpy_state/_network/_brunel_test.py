# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Integration test: Brunel network using the Network API."""
import unittest

import brainstate
import jax.numpy as jnp
import brainunit as u

from brainpy_state import (
    Builder, LIF, Expon, COBA,
    FixedIndegreeProj,
)


class TestBrunelIntegration(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_brunel_small_runs_end_to_end(self):
        N_E, N_I = 80, 20
        eps = 0.1  # connection probability target -> K = eps * N
        K_E = int(eps * N_E)
        K_I = int(eps * N_I)

        b = Builder()
        exc = b.add('exc', LIF(N_E, tau=20 * u.ms,
                               V_th=-50 * u.mV, V_reset=-60 * u.mV,
                               V_rest=-65 * u.mV))
        inh = b.add('inh', LIF(N_I, tau=20 * u.ms,
                               V_th=-50 * u.mV, V_reset=-60 * u.mV,
                               V_rest=-65 * u.mV))

        for src, tgt, w, K, label in [
            (exc, exc, 0.1 * u.nS, K_E, 'e2e'),
            (exc, inh, 0.1 * u.nS, K_E, 'e2i'),
            (inh, exc, -0.5 * u.nS, K_I, 'i2e'),
            (inh, inh, -0.5 * u.nS, K_I, 'i2i'),
        ]:
            b.connect(
                src,
                tgt,
                rule=FixedIndegreeProj,
                K=K,
                weight=w,
                syn=Expon.desc(tgt.in_size, tau=5 * u.ms),
                out=COBA.desc(E=0 * u.mV),
                seed=42,
                allow_multapses=False
            )

        brainstate.nn.init_all_states(b)
        out = b.simulate(
            5 * u.ms,
            monitor={
                'exc_V': lambda n: u.get_mantissa(n.exc.V.value),
                'inh_V': lambda n: u.get_mantissa(n.inh.V.value),
            },
        )

        # 50 timesteps (5 ms / 0.1 ms dt)
        self.assertEqual(out['exc_V'].shape, (50, N_E))
        self.assertEqual(out['inh_V'].shape, (50, N_I))
        self.assertFalse(bool(jnp.any(jnp.isnan(out['exc_V']))))
        self.assertFalse(bool(jnp.any(jnp.isnan(out['inh_V']))))
