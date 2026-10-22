# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of ``iaf_cond_exp`` and ``aeif_cond_alpha`` (fixed-substep RK4); NEST parity in nest_parity_test."""
import math

import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.integrators import RK
from brainpy_state._kernel.models import (
    AEIF_COND_ALPHA_SUBSTEPS,
    IAF_COND_EXP_SUBSTEPS,
    aeif_cond_alpha,
    iaf_cond_exp,
)
from brainpy_state._kernel.models._testing import rollout, step_inputs
from brainpy_state._kernel.ports import Event

DT = 0.1 * u.ms


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _ctx(i=0, dt=DT):
    return StepContext.create(i * dt, dt, i)


def _V(tr, k='V_m'):
    return np.asarray(tr[k].to_decimal(u.mV))


def _psp_inputs(n_steps):
    return step_inputs(n_steps, 1, {'ex': ([2, 30, 31], [8.0, 6.0, 6.0], u.nS),
                                    'in_': ([60], [10.0], u.nS)}, dtype=jnp.float64)


class TestIafCondExp:
    def test_declaration(self):
        assert iaf_cond_exp.integrator == RK(4, IAF_COND_EXP_SUBSTEPS)
        assert iaf_cond_exp.spiking.when == 'step'
        assert iaf_cond_exp.inputs['ex'] == Event(unit=u.nS, into='g_ex')  # receptor port (D-b)
        assert iaf_cond_exp.inputs['in_'] == Event(unit=u.nS, into='g_in')
        assert iaf_cond_exp.outputs['spike'].binary
        assert aeif_cond_alpha.inputs['ex'] == Event(unit=u.nS, into='g_ex')
        assert aeif_cond_alpha.inputs['in_'] == Event(unit=u.nS, into='g_in')
        assert 'on_input' not in iaf_cond_exp.__dict__  # the default receptor jump
        assert list(iaf_cond_exp.all_state_decl()) == ['V_m', 'g_ex', 'g_in', 'latch_I', 'spike',
                                                       'refractory_count']

    def test_conductance_jump_after_propagation(self, x64):
        m = iaf_cond_exp(1, precision='float64')
        m.update(_ctx(), {'ex': 5.0 * u.nS})
        assert float(m.dyn_states['g_ex'].value.to_decimal(u.nS)[0]) == 5.0
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) == -70.0
        m.update(_ctx(1))
        assert float(m.dyn_states['g_ex'].value.to_decimal(u.nS)[0]) == pytest.approx(
            5.0 * math.exp(-0.1 / 0.2), rel=1e-4)  # RK4 (2 substeps) error of exp decay
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) > -70.0

    def test_self_convergence_order(self, x64):
        n = 200
        ref = _V(rollout(iaf_cond_exp(1, precision='float64', integrator=RK(4, 16)), n, DT,
                         _psp_inputs(n), record=('V_m',))[1])
        errs = []
        for k in (1, 2, 4):
            m = iaf_cond_exp(1, precision='float64', integrator=RK(4, k))
            errs.append(np.max(np.abs(_V(rollout(m, n, DT, _psp_inputs(n), record=('V_m',))[1]) - ref)))
        ratios = np.array(errs[:-1]) / np.array(errs[1:])
        assert np.all(ratios > 8.0), (errs, ratios)  # ~16 for a 4th-order method

    def test_derivative_bounds_V_at_threshold(self):
        m = iaf_cond_exp(1)
        s, p = m.read()
        inp = m.prepare_inputs()
        d_hi = m.derivatives(s.replace(V_m=jnp.array([-40.0]) * u.mV), p, inp, _ctx())['V_m']
        d_th = m.derivatives(s.replace(V_m=jnp.array([-55.0]) * u.mV), p, inp, _ctx())['V_m']
        assert u.math.allclose(d_hi, d_th)

    @pytest.mark.parametrize('t_ref, steps', [(2.0, 20), (0.25, 3), (0.0, 0)])
    def test_one_spike_per_step_and_refractory(self, x64, t_ref, steps):
        m = iaf_cond_exp(1, params=dict(t_ref=t_ref * u.ms, I_e=1e6 * u.pA), precision='float64')
        out, tr = rollout(m, 50, DT, record=('V_m',))
        spk = np.asarray(out['spike'])[:, 0]
        assert spk.max() == 1.0
        idx = np.nonzero(spk)[0]
        assert np.all(np.diff(idx) == steps + 1)
        V = _V(tr)[:, 0]
        for k in range(1, steps + 1):
            assert V[idx[0] + k] == -60.0

    def test_current_latched(self, x64):
        m = iaf_cond_exp(1, precision='float64')
        m.update(_ctx(), {'I': 500.0 * u.pA})
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) == -70.0
        m.update(_ctx(1))
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) > -70.0

    def test_float32_heterogeneous(self):
        m = iaf_cond_exp(3, params=dict(I_e=[0.0, 300.0, 600.0] * u.pA))
        out, tr = rollout(m, 500, DT, record=('V_m',))
        counts = np.asarray(out['spike']).sum(0)
        assert counts[0] == 0 and 0 < counts[1] < counts[2]
        assert tr['V_m'].dtype == jnp.float32 and np.all(np.isfinite(_V(tr)))


class TestAeifCondAlpha:
    def test_declaration(self):
        assert aeif_cond_alpha.integrator == RK(4, AEIF_COND_ALPHA_SUBSTEPS)
        assert aeif_cond_alpha.outputs['spike'].max_multiplicity is None
        assert aeif_cond_alpha.spiking.when == 'substep'

    def test_several_spikes_per_step_and_w_jump(self, x64):
        m = aeif_cond_alpha(1, params=dict(I_e=5e4 * u.pA, tau_w=1e6 * u.ms, a=0.0 * u.nS),
                            precision='float64', integrator=RK(4, 8))
        out = m.update(_ctx(dt=1.0 * u.ms))
        n = float(out['spike'][0])
        assert n >= 2.0 and n == int(n)
        # b = 80.5 pA per spike, negligible decay with tau_w = 1e6 ms
        assert float(m.dyn_states['w'].value.to_decimal(u.pA)[0]) == pytest.approx(80.5 * n, rel=1e-5)

    def test_refractory_blocks_in_step_spikes(self, x64):
        m = aeif_cond_alpha(1, params=dict(I_e=5e4 * u.pA, t_ref=0.5 * u.ms), precision='float64',
                            integrator=RK(4, 8))
        out, tr = rollout(m, 30, DT, record=('V_m',))
        spk = np.asarray(out['spike'])[:, 0]
        assert spk.max() == 1.0
        assert np.all(np.diff(np.nonzero(spk)[0]) >= 5)

    def test_delta_T_zero_is_iaf_limit(self, x64):
        m = aeif_cond_alpha(1, params=dict(Delta_T=0.0 * u.mV, I_e=1000.0 * u.pA), precision='float64')
        out, tr = rollout(m, 300, DT, record=('V_m',))
        V = _V(tr)[:, 0]
        assert np.all(np.isfinite(V)) and float(out['spike'].sum()) > 0
        assert V.max() < -50.4 + 1e-9  # threshold V_th, no exponential upswing

        def f(dT):
            mm = aeif_cond_alpha(1, precision='float64')
            s, p = mm.read()
            p = p.replace(Delta_T=jnp.full(1, dT) * u.mV)
            return mm.step(s, p, mm.prepare_inputs(), _ctx())[0].V_m.to_decimal(u.mV)[0]

        assert np.isfinite(float(jax.grad(f)(0.0)))

    def test_alpha_conductance_jump(self, x64):
        m = aeif_cond_alpha(1, precision='float64')
        m.update(_ctx(), {'ex': 2.0 * u.nS})
        assert float(m.dyn_states['dg_ex'].value.to_decimal(u.nS / u.ms)[0]) == pytest.approx(
            2.0 * math.e / 0.2)
        assert float(m.dyn_states['g_ex'].value.to_decimal(u.nS)[0]) == 0.0

    def test_self_convergence(self, x64):
        n = 200
        ref = _V(rollout(aeif_cond_alpha(1, precision='float64', integrator=RK(4, 16)), n, DT,
                         _psp_inputs(n), record=('V_m',))[1])
        errs = [np.max(np.abs(_V(rollout(aeif_cond_alpha(1, precision='float64', integrator=RK(4, k)),
                                         n, DT, _psp_inputs(n), record=('V_m',))[1]) - ref))
                for k in (1, 2, 4)]
        assert errs[0] > errs[1] > errs[2]

    def test_float32_strong_drive_finite(self):
        m = aeif_cond_alpha(4, params=dict(I_e=[0.0, 800.0, 3000.0, 2e4] * u.pA))
        out, tr = rollout(m, 400, DT, record=('V_m', 'w'))
        assert np.all(np.isfinite(_V(tr))) and np.all(np.isfinite(np.asarray(tr['w'].mantissa)))
        counts = np.asarray(out['spike']).sum(0)
        assert counts[0] == 0 and np.all(np.diff(counts) > 0)

    def test_surrogate_override(self):
        m = aeif_cond_alpha(1, surrogate=braintools.surrogate.Sigmoid())
        assert isinstance(m.spiking.surrogate, braintools.surrogate.Sigmoid)
        assert isinstance(aeif_cond_alpha.spiking.surrogate, braintools.surrogate.ReluGrad)
