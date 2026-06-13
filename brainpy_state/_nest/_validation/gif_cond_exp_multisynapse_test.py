# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/gif_cond_exp_multisynapse.py``.

Like ``aeif_cond_beta_multisynapse`` this drives the multi-receptor routing seam
(``connect(receptor_type=k)``), here on a *generalized-IAF* neuron whose two
exponential-conductance receptors have opposite reversal potentials. Validated on
two fronts:

* **Receptor routing (NEST-free).** Driving a single port ``k`` in isolation must
  raise *only* ``g_k`` (peaking near that port's weight) while the other stays at
  machine zero, and the reversal potentials must give the right sign — port 1
  (``E_rev=0``) depolarizes, port 2 (``E_rev=-85``) hyperpolarizes. NEST's
  ``gif_cond_exp_multisynapse`` exposes only ``['E_sfa', 'I_stc', 'V_m']`` as
  recordables (not the per-port conductances), so ``g_1``/``g_2`` are validated
  here rather than against NEST.

* **Live-NEST V_m parity.** With both ports driven the single presynaptic spike
  stays sub-threshold (the escape-rate spiking never triggers; both sides record
  zero spikes), so ``V_m`` is deterministic and must match NEST per sample. With
  the standard one-step recorder alignment (``t=0`` sample dropped candidate-side,
  ``align_steps=1`` for the remaining shift) the agreement is exact to machine
  precision (category A).
"""
import unittest

import brainstate
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:
    nest = None

import saiunit as u

from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

from examples.nest.gif_cond_exp_multisynapse import (
    run_traces, MODEL_PARAMS, DELAYS, WEIGHTS, SPIKE_TIME)

# Category A with the one-step recorder-offset alignment (RKF45 subthreshold V_m).
CAT_A_V = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="A",
                         note="gif V_m, one-step recorder alignment")

_REST = -70.0  # shared NEST/brainpy default E_L (not overridden in MODEL_PARAMS)


def _nest_Vm(dt=0.1, simtime=100.0):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    neu = nest.Create('gif_cond_exp_multisynapse',
                      params={'E_rev': list(MODEL_PARAMS['E_rev']),
                              'tau_syn': list(MODEL_PARAMS['tau_syn'])})
    spk = nest.Create('spike_generator', params={'spike_times': np.array([SPIKE_TIME])})
    mm = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': dt})
    sr = nest.Create('spike_recorder')
    for syn in range(2):
        nest.Connect(spk, neu, syn_spec={
            'synapse_model': 'static_synapse', 'receptor_type': 1 + syn,
            'weight': WEIGHTS[syn], 'delay': DELAYS[syn]})
    nest.Connect(mm, neu)
    nest.Connect(neu, sr)
    nest.Simulate(simtime)
    return np.asarray(mm.events['V_m']), int(sr.n_events)


class TestGifReceptorSeam(unittest.TestCase):
    """NEST-free: connect(receptor_type=k) deposits into exactly port k."""

    def test_receptor_routing_isolates_ports(self):
        for k in (1, 2):
            tr = run_traces(ports=(k,), dt=0.1, simtime=100.0)
            driven = tr[f'g_{k}'].max()
            other = tr[f'g_{2 if k == 1 else 1}'].max()
            with self.subTest(port=k):
                # driven conductance peaks near its weight; the other stays zero
                self.assertGreater(driven, 0.9 * WEIGHTS[k - 1])
                self.assertLess(other, 1e-9)

    def test_excitatory_vs_inhibitory_ports(self):
        # port 1 (E_rev=0) depolarizes above rest; port 2 (E_rev=-85) hyperpolarizes.
        ex = run_traces(ports=(1,), dt=0.1, simtime=100.0)
        self.assertGreater(ex['V_m'].max(), _REST + 0.05)
        inh = run_traces(ports=(2,), dt=0.1, simtime=100.0)
        self.assertLess(inh['V_m'].min(), _REST - 0.05)


@requires_nest
class TestGifNestParity(unittest.TestCase):
    def test_Vm_matches_nest(self):
        bp = run_traces(ports=(1, 2), dt=0.1, simtime=100.0)
        ns_Vm, ns_spikes = _nest_Vm(dt=0.1, simtime=100.0)
        self.assertEqual(ns_spikes, 0)          # sub-threshold: V_m is deterministic
        # brainpy includes the t=0 initial sample; drop it, align_steps=1 absorbs
        # the remaining one-step multimeter offset.
        compare_trace(ns_Vm, bp['V_m'][1:], tol=CAT_A_V, metric='gif V_m').assert_()


if __name__ == '__main__':
    unittest.main()
