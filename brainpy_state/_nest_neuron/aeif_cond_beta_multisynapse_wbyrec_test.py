# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""TDD for the additive ``w_by_rec`` blob seam on ``aeif_cond_beta_multisynapse``.

The multi-receptor ``connect(receptor_type=k)`` seam routes blob deposits through
the Simulator bridge, which calls ``neuron.update(w_by_rec=<(N, n_receptors) nS
mantissa>)``. These NEST-free unit tests pin:

* ``w_by_rec`` is a per-receptor (nS-mantissa) bypass that reproduces the
  equivalent eager ``spike_events`` routing of the same per-port jumps;
* the ``receptor_input_unit`` class attribute (nS for aeif, pA for iaf) the bridge
  uses to scale the gathered conductance/current;
* the ``w_by_rec`` path stays JIT-traceable (it is driven inside the simulate loop).
"""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy_state import aeif_cond_beta_multisynapse, iaf_psc_exp_multisynapse


def _make(n=1):
    # n=1: the realistic single-neuron demo case (the bridge passes varshape+(n_rec,)).
    # (aeif has a latent n>1 multi-receptor broadcasting limitation, out of scope here.)
    # V_peak/V_th huge -> no spike, so V/g/dg/w evolve deterministically.
    return aeif_cond_beta_multisynapse(
        n,
        V_peak=1e6 * u.mV, V_th=1e6 * u.mV, V_reset=0.0 * u.mV, Delta_T=0.0 * u.mV,
        tau_rise=[2.0, 5.0, 10.0] * u.ms, tau_decay=[6.0, 20.0, 40.0] * u.ms,
        E_rev=[0.0, 0.0, -85.0] * u.mV,
        V_initializer=braintools.init.Constant(-65.0 * u.mV),
    )


class TestAeifWByRec(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_receptor_input_unit_class_attrs(self):
        self.assertIs(aeif_cond_beta_multisynapse.receptor_input_unit, u.nS)
        self.assertIs(iaf_psc_exp_multisynapse.receptor_input_unit, u.pA)

    def test_wbyrec_equiv_spike_events(self):
        # Path A: eager spike_events to receptor 2 (1-based), weight 0.5 nS.
        na = _make()
        brainstate.nn.init_all_states(na)
        with brainstate.environ.context(t=0.0 * u.ms):
            na.update(spike_events=[(2, 0.5 * u.nS)])

        # Path B: equivalent blob (column index 1 == receptor 2) via w_by_rec.
        nb = _make()
        brainstate.nn.init_all_states(nb)
        arr = np.zeros((1, 3))
        arr[:, 1] = 0.5
        with brainstate.environ.context(t=0.0 * u.ms):
            nb.update(w_by_rec=arr)

        npt.assert_allclose(np.asarray(na.V.value / u.mV), np.asarray(nb.V.value / u.mV), atol=1e-12)
        npt.assert_allclose(np.asarray(na.g.value / u.nS), np.asarray(nb.g.value / u.nS), atol=1e-12)
        npt.assert_allclose(np.asarray(na.dg.value), np.asarray(nb.dg.value), atol=1e-12)
        npt.assert_allclose(np.asarray(na.w.value / u.pA), np.asarray(nb.w.value / u.pA), atol=1e-12)

    def test_wbyrec_only_targets_named_port(self):
        # Driving only port 3 (index 2, inhibitory E_rev=-85) must leave ports 1,2 at 0.
        n = _make()
        brainstate.nn.init_all_states(n)
        arr = np.zeros((1, 3))
        arr[:, 2] = 1.0
        with brainstate.environ.context(t=0.0 * u.ms):
            n.update(w_by_rec=arr)
        dg = np.asarray(n.dg.value)  # (N, n_receptors), nS/ms mantissa
        self.assertTrue(np.all(dg[:, 0] == 0.0))
        self.assertTrue(np.all(dg[:, 1] == 0.0))
        self.assertTrue(np.all(dg[:, 2] > 0.0))

    def test_wbyrec_is_jittable(self):
        n = _make()
        brainstate.nn.init_all_states(n)
        arr = jnp.zeros((1, 3)).at[:, 0].set(0.3)
        update_jit = brainstate.transform.jit(n.update)
        with brainstate.environ.context(t=0.0 * u.ms):
            out = update_jit(w_by_rec=arr)
        self.assertEqual(tuple(np.asarray(out).shape), (1,))


if __name__ == '__main__':
    unittest.main()
