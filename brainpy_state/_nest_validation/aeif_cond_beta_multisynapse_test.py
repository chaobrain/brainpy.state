# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``examples/nest/aeif_cond_beta_multisynapse.py``.

This is the first demo to exercise the multi-receptor routing seam
(``connect(receptor_type=k)``) end to end, so it is validated on two fronts:

* **Receptor routing (NEST-free).** Driving a single port ``k`` in isolation must
  raise *only* ``g_k`` (the other three conductances stay at machine zero), and the
  reversal potentials must give the right sign — excitatory ports (1–3, ``E_rev=0``)
  depolarize while the inhibitory port (4, ``E_rev=-85``) hyperpolarizes. This is
  the direct proof that ``receptor_type=k`` deposits into exactly port ``k``.

* **Live-NEST trace parity.** With all four ports driven, ``V_m`` and the four
  per-port conductances ``g_1..g_4`` must match NEST per sample. The ``aeif`` rides
  an adaptive RKF45 integrator; with the standard one-step multimeter-offset
  alignment (brainpy samples the ``t=0`` initial state, dropped here, then a further
  one-step recorder shift via ``align_steps=1``) the agreement is to machine
  precision for the conductances and ~1e-6 mV for ``V_m`` (category A).
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

import brainunit as u

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance

from examples.nest.aeif_cond_beta_multisynapse import (
    run_traces, MODEL_PARAMS, DELAYS, WEIGHTS, SPIKE_TIME)

# Category A with the one-step recorder-offset alignment (RKF45 conductance trace,
# t=0 sample dropped candidate-side; align_steps absorbs the remaining shift).
CAT_A_V = TraceTolerance(1e-3 * u.mV, 1e-3, align_steps=1, label="A",
                         note="aeif V_m, one-step recorder alignment")
CAT_A_G = TraceTolerance(1e-3, 1e-3, align_steps=1, label="A",
                         note="aeif per-port conductance g_k, one-step aligned")


def _nest_traces(dt=0.1, simtime=1000.0):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    neu = nest.Create('aeif_cond_beta_multisynapse', params={
        'V_peak': 0.0, 'a': 4.0, 'b': 80.5,
        'E_rev': [0.0, 0.0, 0.0, -85.0],
        'tau_decay': [50.0, 20.0, 20.0, 20.0],
        'tau_rise': [10.0, 10.0, 1.0, 1.0]})
    spk = nest.Create('spike_generator', params={'spike_times': np.array([SPIKE_TIME])})
    mm = nest.Create('multimeter', params={
        'record_from': ['V_m', 'g_1', 'g_2', 'g_3', 'g_4'], 'interval': dt})
    for syn in range(4):
        nest.Connect(spk, neu, syn_spec={
            'synapse_model': 'static_synapse', 'receptor_type': 1 + syn,
            'weight': WEIGHTS[syn], 'delay': DELAYS[syn]})
    nest.Connect(mm, neu)
    nest.Simulate(simtime)
    ev = mm.events
    return {k: np.asarray(ev[k]) for k in ('V_m', 'g_1', 'g_2', 'g_3', 'g_4')}


class TestAeifReceptorSeam(unittest.TestCase):
    """NEST-free: connect(receptor_type=k) deposits into exactly port k."""

    def test_receptor_routing_isolates_ports(self):
        for k in (1, 2, 3, 4):
            tr = run_traces(ports=(k,), dt=0.1, simtime=1000.0)
            driven = tr[f'g_{k}'].max()
            others = [tr[f'g_{j}'].max() for j in (1, 2, 3, 4) if j != k]
            with self.subTest(port=k):
                self.assertGreater(driven, 0.9)             # driven port responds (~weight)
                self.assertLess(max(others), 1e-9)          # the rest stay at machine zero

    def test_excitatory_vs_inhibitory_ports(self):
        rest = -70.6  # shared NEST/brainpy default E_L (not overridden in MODEL_PARAMS)
        # Excitatory port (1, E_rev=0): V_m rises above rest.
        ex = run_traces(ports=(1,), dt=0.1, simtime=400.0)
        self.assertGreater(ex['V_m'].max(), rest + 0.05)
        # Inhibitory port (4, E_rev=-85): V_m dips below rest.
        inh = run_traces(ports=(4,), dt=0.1, simtime=1000.0)
        self.assertLess(inh['V_m'].min(), rest - 0.05)


@requires_nest
class TestAeifNestParity(unittest.TestCase):
    def test_all_ports_Vm_and_g_match_nest(self):
        bp = run_traces(ports=(1, 2, 3, 4), dt=0.1, simtime=1000.0)
        ns = _nest_traces(dt=0.1, simtime=1000.0)
        # brainpy includes the t=0 initial sample; drop it, then align_steps=1
        # absorbs the remaining one-step multimeter offset.
        compare_trace(ns['V_m'], bp['V_m'][1:], tol=CAT_A_V,
                      metric='aeif V_m').assert_()
        for k in (1, 2, 3, 4):
            compare_trace(ns[f'g_{k}'], bp[f'g_{k}'][1:], tol=CAT_A_G,
                          metric=f'aeif g_{k}').assert_()


if __name__ == '__main__':
    unittest.main()
