# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Per-generator weight vectors (Extension D2): multi-channel generator view.

``create(poisson_generator, k, rate=[...])`` returns a ``k``-segment NodeView
(one independent scalar-rate channel per segment), and
``connect(gen, neuron, weight=[w0..w_{k-1}])`` applies ``weight[i]`` to channel
``i`` (signed: positive excitatory, negative inhibitory), summed in the neuron.
"""
import unittest

import braintools
import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import iaf_psc_alpha, poisson_generator, voltmeter
from brainpy_state._nest_network import Simulator
from brainpy_state._nest_network._simulator import (
    _n_channels, _is_len_vector, _index_channel)

# A neuron that never spikes (V_th unreachable): pure subthreshold integration so
# the recorded V_m reflects the net (signed) synaptic drive directly.
NOSPIKE = dict(C_m=250. * u.pF, tau_m=10. * u.ms, tau_syn_ex=2. * u.ms,
               tau_syn_in=2. * u.ms, t_ref=2. * u.ms, E_L=-70. * u.mV,
               V_reset=-70. * u.mV, V_th=1e6 * u.mV,
               V_initializer=braintools.init.Constant(-70. * u.mV))


def _vm(res, rec):
    return np.asarray(u.get_mantissa(res.trace(rec, 'V_m') / u.mV)).reshape(-1)


class TestPerGeneratorWeightVector(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_multichannel_generator_creates_one_segment_per_channel(self):
        sim = Simulator(dt=0.1 * u.ms)
        gen = sim.create(poisson_generator, 2, rate=[8000., 2000.] * u.Hz)
        self.assertEqual(len(gen.segments), 2)
        r0 = gen.segments[0].spec.params['rate']
        r1 = gen.segments[1].spec.params['rate']
        self.assertAlmostEqual(float(r0 / u.Hz), 8000.)
        self.assertAlmostEqual(float(r1 / u.Hz), 2000.)

    def test_signed_weight_summation_lowers_vm(self):
        # Two channels share the same excitatory train (segment 0, ordinal 0 ->
        # same derived seed across both fresh sims); only the second channel's
        # weight changes. Adding inhibition must pull mean V_m below the
        # excitation-only case.
        def run(weights):
            sim = Simulator(dt=0.1 * u.ms)
            neu = sim.create(iaf_psc_alpha, 1, params=NOSPIKE)
            gen = sim.create(poisson_generator, 2, rate=[5000., 5000.] * u.Hz)
            vm = sim.create(voltmeter)
            sim.connect(gen, neu, weight=weights, delay=1. * u.ms)
            sim.connect(vm, neu)
            return float(_vm(sim.simulate(100. * u.ms), vm).mean())

        exc_only = run([5.0, 0.0] * u.pA)
        balanced = run([5.0, -5.0] * u.pA)
        self.assertGreater(exc_only, -69.5)      # excitation depolarizes past rest
        self.assertLess(balanced, exc_only)      # inhibition pulls mean V_m down

    def test_scalar_weight_broadcasts_across_channels(self):
        # A scalar weight must behave like a same-valued vector across channels.
        def run(weight):
            sim = Simulator(dt=0.1 * u.ms)
            neu = sim.create(iaf_psc_alpha, 1, params=NOSPIKE)
            gen = sim.create(poisson_generator, 2, rate=[4000., 4000.] * u.Hz)
            vm = sim.create(voltmeter)
            sim.connect(gen, neu, weight=weight, delay=1. * u.ms)
            sim.connect(vm, neu)
            return _vm(sim.simulate(50. * u.ms), vm)

        v_scalar = run(3.0 * u.pA)
        v_vector = run([3.0, 3.0] * u.pA)
        npt = float(np.max(np.abs(v_scalar - v_vector)))
        self.assertLess(npt, 1e-9, f"scalar vs vector weight mismatch {npt}")


class TestWeightVectorHelpers(unittest.TestCase):
    """The channel-splitting helpers behind ``create``/``connect`` (Extension D2)."""

    def test_n_channels_flattens_size_specs(self):
        self.assertEqual(_n_channels(3), 3)          # scalar size
        self.assertEqual(_n_channels((2, 3)), 6)     # tuple -> product
        self.assertEqual(_n_channels([4]), 4)        # list -> product

    def test_is_len_vector_detects_length_k_across_types(self):
        self.assertTrue(_is_len_vector([1., 2.], 2))             # list
        self.assertTrue(_is_len_vector((1., 2.), 2))            # tuple
        self.assertTrue(_is_len_vector(np.array([1., 2.]), 2))  # ndarray
        self.assertTrue(_is_len_vector([1., 2.] * u.pA, 2))     # Quantity vector
        self.assertFalse(_is_len_vector([1., 2., 3.], 2))       # wrong length
        self.assertFalse(_is_len_vector(5.0, 2))               # bare scalar
        self.assertFalse(_is_len_vector(5.0 * u.pA, 2))        # scalar Quantity

    def test_index_channel_extracts_or_broadcasts(self):
        # Quantity vector -> channel ``i`` preserving units.
        self.assertAlmostEqual(
            float(_index_channel([10., 20.] * u.pA, 1, 2) / u.pA), 20.)
        # Plain sequence vector -> element ``i``.
        self.assertEqual(_index_channel([10., 20.], 0, 2), 10.)
        # Scalar -> broadcast unchanged (float and Quantity).
        self.assertEqual(_index_channel(7.0, 0, 2), 7.0)
        self.assertAlmostEqual(
            float(_index_channel(7.0 * u.pA, 1, 2) / u.pA), 7.0)


if __name__ == '__main__':
    unittest.main()
