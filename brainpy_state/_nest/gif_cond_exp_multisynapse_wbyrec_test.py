# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""TDD for the additive ``w_by_rec`` blob seam on ``gif_cond_exp_multisynapse``.

``gif_cond_exp_multisynapse`` joins the blob camp (symmetric with
``aeif_cond_beta_multisynapse``) because its legacy ``key.split('_')`` receptor
parser is incompatible with the GLIF ``label='receptor_k'`` convention the router
emits. The additive ``w_by_rec`` bypass lets the Simulator bridge drive it while its
documented per-key ``add_delta_input('receptor_k', ...)`` API stays intact.

These NEST-free unit tests pin:

* ``update(w_by_rec=arr)`` reproduces the legacy per-key routing of the same jumps
  (per-port conductance state ``g[k]`` matches);
* the legacy ``add_delta_input('receptor_k', ...)`` path is unchanged when
  ``w_by_rec`` is None (back-compat);
* ``receptor_input_unit`` is nS;
* the ``w_by_rec`` path stays JIT-traceable.

The receptor jump is applied *after* the integrator (NEST ring-buffer ordering),
so after a single step it shows up in ``g[k]`` rather than ``V`` — hence the
equivalence is asserted on the conductance states. ``lambda_0=0`` disables
stochastic spiking, making ``V``/``g`` deterministic.
"""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import saiunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import gif_cond_exp_multisynapse


def _make(n=2):
    # gif takes bare float sequences for tau_syn (ms) and E_rev (mV), per its API.
    return gif_cond_exp_multisynapse(
        n,
        tau_syn=(4.0, 8.0),
        E_rev=(0.0, -85.0),
        lambda_0=0.0,  # deterministic: no stochastic spikes
        V_initializer=braintools.init.Constant(-65.0 * u.mV),
    )


class TestGifWByRec(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_receptor_input_unit_class_attr(self):
        self.assertIs(gif_cond_exp_multisynapse.receptor_input_unit, u.nS)

    def test_wbyrec_equiv_legacy_keys(self):
        j0, j1 = 0.7, 2.0  # nS jumps for ports 0 and 1

        # Path A: blob bypass.
        na = _make()
        brainstate.nn.init_all_states(na)
        arr = np.zeros((2, 2))
        arr[:, 0] = j0
        arr[:, 1] = j1
        with brainstate.environ.context(t=0.0 * u.ms):
            na.update(w_by_rec=arr)

        # Path B: legacy per-key routing ('receptor_0' -> port 0, 'receptor_1' -> port 1).
        nb = _make()
        brainstate.nn.init_all_states(nb)
        nb.add_delta_input('receptor_0', j0 * u.nS)
        nb.add_delta_input('receptor_1', j1 * u.nS)
        with brainstate.environ.context(t=0.0 * u.ms):
            nb.update()

        for k in range(2):
            npt.assert_allclose(
                np.asarray(na.g[k].value / u.nS), np.asarray(nb.g[k].value / u.nS), atol=1e-12,
                err_msg=f'g[{k}] mismatch between w_by_rec and legacy-key routing',
            )
        npt.assert_allclose(np.asarray(na.V.value / u.mV), np.asarray(nb.V.value / u.mV), atol=1e-12)

    def test_legacy_key_path_unchanged_when_wbyrec_none(self):
        # Back-compat: only port 1 driven via legacy key -> g[1] jumps, g[0] stays 0.
        n = _make()
        brainstate.nn.init_all_states(n)
        n.add_delta_input('receptor_1', 3.0 * u.nS)
        with brainstate.environ.context(t=0.0 * u.ms):
            n.update()
        self.assertTrue(np.all(np.asarray(n.g[0].value / u.nS) == 0.0))
        self.assertTrue(np.all(np.asarray(n.g[1].value / u.nS) > 0.0))

    def test_wbyrec_is_jittable(self):
        n = _make()
        brainstate.nn.init_all_states(n)
        arr = jnp.zeros((2, 2)).at[:, 0].set(0.5)
        update_jit = brainstate.transform.jit(n.update)
        with brainstate.environ.context(t=0.0 * u.ms):
            out = update_jit(w_by_rec=arr)
        self.assertEqual(tuple(np.asarray(out).shape), (2,))


if __name__ == '__main__':
    unittest.main()
