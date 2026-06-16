# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Unit tests for the recordable-alias resolver in the Simulator.

NEST exposes recordables under names that differ from the brainpy.state State
attributes (``Act_m`` vs ``m``, ``E_sfa`` vs ``_sfa_val_state``, …), and some
recordables are derived (per-port conductance ``g_1`` from a ``g_syn`` list,
``ASCurrents_sum`` summed from per-mechanism states). ``_read_recordable`` maps a
NEST recordable name to its State value via ``_RECORDABLE_ALIAS``, supporting both
tuple-of-candidate-attrs entries and callable entries.

These tests pin the resolver mechanism with a fake population; the *semantic*
correctness of each alias (does ``_sfa_val_state`` equal NEST's ``E_sfa``?) is
verified by the per-demo live-NEST parity tests, not here.
"""
import unittest

import numpy as np
import numpy.testing as npt

from brainpy_state._nest_network.simulator import _read_recordable


class _FakeState:
    def __init__(self, value):
        self.value = value


class _FakePop:
    """Exposes the union of HH/GIF/GLIF State attributes the aliases target."""

    def __init__(self, with_asc_sum=True):
        self.V = _FakeState(np.array([-65.0]))
        # HH gating
        self.m = _FakeState(np.array([0.05]))
        self.h = _FakeState(np.array([0.6]))
        self.n = _FakeState(np.array([0.32]))
        # GIF adaptation
        self._sfa_val_state = _FakeState(np.array([-50.0]))
        self._stc_val_state = _FakeState(np.array([1.5]))
        # GLIF threshold components
        self._threshold_state = _FakeState(np.array([10.0]))
        self._threshold_spike_state = _FakeState(np.array([2.0]))
        self._threshold_voltage_state = _FakeState(np.array([3.0]))
        # GLIF per-port conductance (glif_cond)
        self.g_syn = [_FakeState(np.array([0.7])), _FakeState(np.array([0.2]))]
        # GLIF after-spike currents: glif_cond has a precomputed sum state,
        # glif_psc only the per-mechanism list.
        self._asc_states = [_FakeState(np.array([0.4])), _FakeState(np.array([0.1]))]
        if with_asc_sum:
            self._asc_sum_state = _FakeState(np.array([0.5]))
        # glif_psc per-port post-synaptic current (PSC, pA): I_syn = sum over ports.
        self.y2 = [_FakeState(np.array([12.0])), _FakeState(np.array([3.0]))]
        # glif_psc / iaf injected (current-generator) current, NEST recordable ``I``.
        self.I_stim = _FakeState(np.array([400.0]))
        # izhikevich recovery variable (NEST ``U_m`` -> brainpy ``U``).
        self.U = _FakeState(np.array([-13.0]))


class _FakeMatPop:
    """mat2_psc_exp: adaptive threshold is *computed* as omega + V_th_1 + V_th_2.

    ``omega`` is a parameter (absolute mV, no ``.value``); ``V_th_1``/``V_th_2``
    are States that jump on the model's own spikes and decay.
    """

    def __init__(self):
        self.V = _FakeState(np.array([-70.0]))
        self.omega = np.array([-51.0])              # parameter, absolute mV
        self.V_th_1 = _FakeState(np.array([5.0]))
        self.V_th_2 = _FakeState(np.array([2.0]))


class _FakeAmatPop:
    """amat2_psc_exp: adds the voltage-dependent component V_th_v to the sum."""

    def __init__(self):
        self.V = _FakeState(np.array([-70.0]))
        self.omega = np.array([-51.0])
        self.V_th_1 = _FakeState(np.array([5.0]))
        self.V_th_2 = _FakeState(np.array([2.0]))
        self.V_th_v = _FakeState(np.array([1.0]))   # voltage-dependent component


class _FakeClopathPop:
    """aeif_psc_delta_clopath: stores the adaptive threshold directly as ``V_th``.

    No MAT ``omega``; the ``V_th`` recordable must fall back to this State so the
    MAT callable does not hijack a model that already exposes ``V_th`` directly.
    """

    def __init__(self):
        self.V = _FakeState(np.array([-55.0]))
        self.V_th = _FakeState(np.array([-50.4]))


class _FakeMcPop:
    """iaf_cond_alpha_mc: each compartment stacked on the last axis (SOMA/PROX/DIST)."""

    def __init__(self):
        # shape (n_neurons=1, n_compartments=3): columns = soma, proximal, distal.
        self.V = _FakeState(np.array([[-65.0, -64.0, -63.0]]))
        self.g_ex = _FakeState(np.array([[0.1, 0.2, 0.3]]))
        self.g_in = _FakeState(np.array([[0.4, 0.5, 0.6]]))


class _FakeAeifPop:
    """aeif_cond_beta_multisynapse: one ``g`` State, receptor on the last axis."""

    def __init__(self):
        self.V = _FakeState(np.array([-70.0]))
        self.g = _FakeState(np.array([[0.1, 0.2, 0.3, 0.4]]))   # (1, n_receptors)


class _FakeGifPop:
    """gif_cond_exp_multisynapse: ``g`` is a *list* of per-receptor States."""

    def __init__(self):
        self.V = _FakeState(np.array([-65.0]))
        self.g = [_FakeState(np.array([0.7])), _FakeState(np.array([0.2]))]


class _FakeDoubleAlphaPop:
    """glif_psc_double_alpha: PSC split across y2_fast/y2_slow, summed by get_I_syn()."""

    def __init__(self):
        self.V = _FakeState(np.array([-78.85]))
        # No flat ``y2`` list — the fast/slow split is summed by the model method.
        self.y2_fast = [_FakeState(np.array([5.0])), _FakeState(np.array([2.0]))]
        self.y2_slow = [_FakeState(np.array([1.0])), _FakeState(np.array([0.5]))]

    def get_I_syn(self):
        return sum(f.value + s.value for f, s in zip(self.y2_fast, self.y2_slow))


class TestRecordableAlias(unittest.TestCase):
    def test_backcompat_V_m(self):
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'V_m'), pop.V.value)

    def test_hh_gating_tuple_aliases(self):
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'Act_m'), pop.m.value)
        self.assertIs(_read_recordable(pop, 'Inact_h'), pop.h.value)
        self.assertIs(_read_recordable(pop, 'Act_n'), pop.n.value)

    def test_gif_adaptation_aliases(self):
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'E_sfa'), pop._sfa_val_state.value)
        self.assertIs(_read_recordable(pop, 'I_stc'), pop._stc_val_state.value)

    def test_glif_threshold_aliases(self):
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'threshold'), pop._threshold_state.value)
        self.assertIs(_read_recordable(pop, 'threshold_spike'), pop._threshold_spike_state.value)
        self.assertIs(_read_recordable(pop, 'threshold_voltage'), pop._threshold_voltage_state.value)

    def test_glif_per_port_conductance_callable(self):
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'g_1'), pop.g_syn[0].value)
        self.assertIs(_read_recordable(pop, 'g_2'), pop.g_syn[1].value)

    def test_aeif_per_port_conductance_last_axis(self):
        # aeif stores one g State of shape (*, n_receptors); g_k indexes the last axis.
        pop = _FakeAeifPop()
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_1')), [0.1])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_3')), [0.3])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_4')), [0.4])

    def test_gif_per_port_conductance_list(self):
        # gif stores g as a list of per-receptor States; g_k indexes the list.
        pop = _FakeGifPop()
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_1')), [0.7])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_2')), [0.2])

    def test_ascurrents_sum_prefers_precomputed_state(self):
        pop = _FakePop(with_asc_sum=True)
        self.assertIs(_read_recordable(pop, 'ASCurrents_sum'), pop._asc_sum_state.value)

    def test_ascurrents_sum_falls_back_to_summing_states(self):
        pop = _FakePop(with_asc_sum=False)
        got = _read_recordable(pop, 'ASCurrents_sum')
        npt.assert_allclose(np.asarray(got), np.array([0.5]))  # 0.4 + 0.1

    def test_i_syn_sums_per_port_psc(self):
        # glif_psc exposes per-port PSC states ``y2``; NEST ``I_syn`` is their sum
        # (glif_psc.cpp: ``S_.I_syn_ += S_.y2_[i]`` over all receptors).
        pop = _FakePop()
        got = _read_recordable(pop, 'I_syn')
        npt.assert_allclose(np.asarray(got), np.array([15.0]))  # 12.0 + 3.0

    def test_i_syn_prefers_get_I_syn_method(self):
        # glif_psc_double_alpha splits the PSC across y2_fast/y2_slow and exposes a
        # get_I_syn() summing both; the resolver must call it (there is no flat y2).
        pop = _FakeDoubleAlphaPop()
        got = _read_recordable(pop, 'I_syn')
        npt.assert_allclose(np.asarray(got), np.array([8.5]))  # (5+1) + (2+0.5)

    def test_injected_current_I_maps_to_I_stim(self):
        # NEST records the current-generator input as ``I`` (S_.I_ = currents_);
        # brainpy buffers it on the I_stim ShortTermState.
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'I'), pop.I_stim.value)

    def test_g_port_without_conductance_state_raises(self):
        # ``g_k`` on a population exposing neither ``g_syn`` (glif_cond) nor ``g``
        # (aeif/gif) is a misconfiguration, not a silent zero. The double-alpha
        # pop is current-based (PSC, no conductance), so g_1 must raise clearly.
        pop = _FakeDoubleAlphaPop()
        with self.assertRaisesRegex(KeyError, 'neither g_syn nor g'):
            _read_recordable(pop, 'g_1')

    def test_izhikevich_U_m_maps_to_U(self):
        # NEST izhikevich exposes the recovery variable as ``U_m``; brainpy ``U``.
        pop = _FakePop()
        self.assertIs(_read_recordable(pop, 'U_m'), pop.U.value)

    def test_mat_V_th_sums_omega_and_threshold_components(self):
        # mat2_psc_exp threshold is computed: omega + V_th_1 + V_th_2 (absolute mV).
        pop = _FakeMatPop()
        got = _read_recordable(pop, 'V_th')
        npt.assert_allclose(np.asarray(got), np.array([-44.0]))  # -51 + 5 + 2

    def test_amat_V_th_includes_voltage_dependent_component(self):
        # amat2_psc_exp adds the voltage-dependent V_th_v to the threshold sum.
        pop = _FakeAmatPop()
        got = _read_recordable(pop, 'V_th')
        npt.assert_allclose(np.asarray(got), np.array([-43.0]))  # -51 + 5 + 2 + 1

    def test_amat_V_th_v_resolves_by_name(self):
        # The voltage-dependent component is itself recordable under its own name.
        pop = _FakeAmatPop()
        self.assertIs(_read_recordable(pop, 'V_th_v'), pop.V_th_v.value)

    def test_V_th_falls_back_to_direct_state_when_not_mat(self):
        # A model storing the threshold directly as a V_th State (clopath) must not
        # be hijacked by the MAT sum (which would AttributeError on V_th_1).
        pop = _FakeClopathPop()
        self.assertIs(_read_recordable(pop, 'V_th'), pop.V_th.value)

    def test_V_th_without_omega_or_state_raises(self):
        # A population that is neither a MAT model nor exposes a V_th State is a
        # misconfiguration, not a silent value.
        pop = _FakePop()  # has neither omega+V_th_1 nor a V_th State
        with self.assertRaisesRegex(KeyError, 'V_th'):
            _read_recordable(pop, 'V_th')

    def test_mc_per_compartment_V_m(self):
        # iaf_cond_alpha_mc: V_m.s/.p/.d index the compartment on the last axis.
        pop = _FakeMcPop()
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'V_m.s')), [-65.0])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'V_m.p')), [-64.0])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'V_m.d')), [-63.0])

    def test_mc_per_compartment_conductances(self):
        pop = _FakeMcPop()
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_ex.s')), [0.1])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_ex.d')), [0.3])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_in.s')), [0.4])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_in.p')), [0.5])
        npt.assert_allclose(np.asarray(_read_recordable(pop, 'g_in.d')), [0.6])

    def test_mc_compartment_without_state_raises(self):
        # Asking for a compartment recordable on a pop lacking that stacked State
        # is a misconfiguration.
        pop = _FakePop()  # no g_ex/g_in/V stacked-compartment states
        with self.assertRaises(KeyError):
            _read_recordable(pop, 'g_ex.s')

    def test_unknown_recordable_raises_keyerror(self):
        pop = _FakePop()
        with self.assertRaises(KeyError):
            _read_recordable(pop, 'definitely_not_a_recordable')


if __name__ == '__main__':
    unittest.main()
