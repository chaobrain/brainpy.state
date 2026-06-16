# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
# -*- coding: utf-8 -*-
r"""Oracle-anchor tests for the State-based ``cm_default.update()`` path.

The legacy host-loop ``cm_default.step()`` is a comprehensively-tested,
pure-Python Crank-Nicolson/Hines reference (see ``cm_default_test.py``).  This
module pins the *new* vectorised, ``for_loop``-lowerable ``update()`` path to
that reference: both are driven with an identical spike/current schedule on the
same morphology and their compartment-voltage traces must agree to ~machine
precision.  Because the host path is the oracle, passing this test validates the
JAX math *before* any NEST comparison.
"""

import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import unittest

import brainunit as u
import numpy as np

from brainpy_state._nest.cm_default import cm_default


# Three-compartment tree exercising all four receptor kinds:
#   soma (0): active Na/K               -> AMPA receptor (idx 2)
#   dend (1): passive, child of soma    -> AMPA_NMDA receptor (idx 0), NMDA (idx 3)
#   dend (2): passive, child of soma    -> GABA receptor (idx 1)
_SOMA = {
    'C_m': 89.245535, 'g_C': 0.0, 'g_L': 8.924572508, 'e_L': -75.0, 'v_comp': -75.0,
    'gbar_Na': 4608.698576715, 'e_Na': 60.0, 'gbar_K': 956.112772900, 'e_K': -90.0,
}
_DEND = {'C_m': 1.928, 'g_C': 1.260, 'g_L': 0.192, 'e_L': -70.0, 'v_comp': -70.0}

_COMPARTMENTS = [
    {'parent_idx': -1, 'params': _SOMA},
    {'parent_idx': 0, 'params': _DEND},
    {'parent_idx': 0, 'params': _DEND},
]
# Receptor add-order fixes the global index (NEST syn_idx).
_RECEPTORS = [
    {'comp_idx': 1, 'receptor_type': 'AMPA_NMDA'},                 # idx 0
    {'comp_idx': 2, 'receptor_type': 'GABA'},                      # idx 1
    {'comp_idx': 0, 'receptor_type': 'AMPA',
     'params': {'tau_r_AMPA': 0.2, 'tau_d_AMPA': 3.0, 'e_AMPA': 0.0}},  # idx 2
    {'comp_idx': 1, 'receptor_type': 'NMDA'},                      # idx 3
]
_V_TH = -50.0
_DT = 0.1
_N_STEPS = 800

# Deterministic input schedule: {step: [(receptor_idx, weight_nS), ...]}
# Strong somatic AMPA drive to force action potentials (exercise the Na/K
# nonlinearity), plus dendritic excitation/inhibition on the slower receptors.
_SPIKES = {
    50: [(2, 30.0)], 60: [(2, 30.0)], 70: [(2, 30.0)], 80: [(2, 30.0)],
    120: [(0, 5.0), (3, 2.0)], 130: [(0, 5.0)], 160: [(1, 4.0)],
    300: [(2, 40.0)], 305: [(2, 40.0)], 310: [(2, 40.0)],
    400: [(0, 8.0), (3, 3.0)], 410: [(1, 6.0)], 500: [(2, 50.0)],
}
# {step: [(comp_idx, current_pA), ...]}
_CURRENTS = {s: [(0, 80.0)] for s in range(600, 700)}  # 10 ms soma current pulse


def _build_host():
    h = cm_default(V_th=_V_TH)
    for comp in _COMPARTMENTS:
        h.add_compartment(comp['parent_idx'], comp['params'])
    for rec in _RECEPTORS:
        h.add_receptor(rec['comp_idx'], rec['receptor_type'], rec.get('params'))
    h.pre_run_hook(dt=_DT)
    return h


def _build_state():
    brainstate.environ.set(dt=_DT * u.ms)
    return cm_default(1, V_th=_V_TH, compartments=_COMPARTMENTS, receptors=_RECEPTORS)


class TestCmDefaultStateOracle(unittest.TestCase):
    """The vectorised update() must reproduce the host step() trace exactly."""

    def _run_both(self):
        host = _build_host()
        state = _build_state()
        n_comp = host.num_compartments
        v_host = np.zeros((_N_STEPS, n_comp))
        v_state = np.zeros((_N_STEPS, n_comp))
        spk_host = np.zeros(_N_STEPS, dtype=bool)
        spk_state = np.zeros(_N_STEPS, dtype=bool)

        for t in range(_N_STEPS):
            spikes = _SPIKES.get(t, [])
            currents = _CURRENTS.get(t, [])
            # host: queue inputs, then step
            for ridx, w in spikes:
                host.add_spike(ridx, w)
            for cidx, cur in currents:
                host.add_current(cidx, cur)
            spk_host[t] = host.step()
            v_host[t] = host.get_voltages()
            # state: identical inputs via update() direct-event path
            s = state.update(
                spike_events=[(ridx, w * u.nS) for ridx, w in spikes],
                current_events=[(cidx, cur * u.pA) for cidx, cur in currents],
            )
            spk_state[t] = bool(np.asarray(s)[0] > 0.5)
            v_state[t] = np.asarray(state.V.value)[0]
        return v_host, v_state, spk_host, spk_state

    def test_voltage_traces_match_host(self):
        v_host, v_state, _, _ = self._run_both()
        # Same Crank-Nicolson system, same exact-integration gating -> machine precision.
        np.testing.assert_allclose(v_state, v_host, atol=1e-7, rtol=1e-7)

    def test_spikes_match_host(self):
        _, _, spk_host, spk_state = self._run_both()
        self.assertGreater(spk_host.sum(), 0, 'schedule should evoke at least one AP')
        np.testing.assert_array_equal(spk_state, spk_host)

    def test_final_state_is_finite(self):
        state = _build_state()
        for t in range(200):
            state.update(spike_events=[(2, 30.0 * u.nS)] if t % 20 == 0 else None)
        self.assertTrue(np.all(np.isfinite(np.asarray(state.V.value))))


class TestCmDefaultSeamValidation(unittest.TestCase):
    """Routing/recordable resolvers and their validation branches (NEST-free).

    The morphology has ``n_comp=3`` and ``n_rec=4`` (see ``_RECEPTORS``): receptor
    indices 0-3 are valid; compartment indices 0-2 are valid current ports.
    """

    def setUp(self):
        brainstate.environ.set(dt=_DT * u.ms)

    def test_delta_label_for_receptor(self):
        # Spike routing: valid receptor index -> unique label; out-of-range / non-int
        # rejected (the Simulator resolves this once at connect()).
        s = _build_state()
        self.assertEqual(s.delta_label_for_receptor(0), 'cm_rcpt0#')
        self.assertEqual(s.delta_label_for_receptor(3), 'cm_rcpt3#')
        for bad in (-1, 4, 'x'):
            with self.assertRaises(ValueError):
                s.delta_label_for_receptor(bad)

    def test_current_compartment_for_receptor(self):
        # Current routing: receptor_type IS the compartment index (NEST convention);
        # out-of-range / non-int rejected. NCOMP exposes the compartment count.
        s = _build_state()
        self.assertEqual(s.NCOMP, 3)
        self.assertEqual(s.current_compartment_for_receptor(0), 0)
        self.assertEqual(s.current_compartment_for_receptor(2), 2)
        for bad in (-1, 3, 'x'):
            with self.assertRaises(ValueError):
                s.current_compartment_for_receptor(bad)

    def test_read_recordable_totals_and_unknown(self):
        # The total-conductance recordables (g_AMPA_/g_GABA_/g_NMDA_) resolve via the
        # first receptor on each compartment; an unknown name raises KeyError.
        s = _build_state()
        for name in ('g_AMPA_0', 'g_GABA_2', 'g_NMDA_1'):
            self.assertEqual(np.asarray(s.read_recordable(name)).shape, (1,))
        with self.assertRaises(KeyError):
            s.read_recordable('not_a_recordable')

    def test_init_state_is_noop_in_host_mode(self):
        # A bare (host-loop) instance has no morphology/State yet, so init_state()
        # must be a no-op rather than allocating against an empty topology.
        h = cm_default(V_th=_V_TH)
        h.init_state()
        self.assertFalse(getattr(h, '_state_ready', False))


if __name__ == '__main__':
    unittest.main()
