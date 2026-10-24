# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-3: one receptor-port encoding (decision D-b).

A neuron-owned receptor port is an ``Event`` (or ``TimedEvent``) input port naming
the state it filters into (``Event(unit=u.nS, into='g_ex')``); projection-owned
kinetics (``synapse=Expon(...)``) only reach plain ``Current`` / ``Conductance``
ports. The architecture doc §7 L1 ``iaf_cond_exp`` example is reproduced here and
must compile and run; the K0 encoding (``filtered=True``) is gone.
"""
from __future__ import annotations

from typing import ClassVar

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import Exact, Integrator, RK
from brainspike._kernel.model import NeuronModel
from brainspike._kernel.models import (
    aeif_cond_alpha,
    iaf_cond_exp,
    iaf_psc_alpha,
    iaf_psc_exp_ps,
    spike_generator,
)
from brainspike._kernel.network import Network
from brainspike._kernel.params import Init, Normal, Params, Record, States, Uniform
from brainspike._kernel.ports import Conductance, Current, Event, PortError, Ports, TimedEvent
from brainspike._kernel.spiking import Clamp, SpikeRule
from brainspike._kernel.synapse import Expon
from brainspike._kernel.units import UnitMismatchError

DT = 0.1


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


# --------------------------------------------------------------------------- the §7 example

class iaf_cond_exp_doc(NeuronModel):
    """Architecture doc §7 L1 ``NeuronModel`` example (corrected to the kernel API)."""

    params = Params(C_m=250 * u.pF, g_L=16.6667 * u.nS, E_ex=0 * u.mV, E_in=-85 * u.mV,
                    E_L=-70 * u.mV, V_th=-55 * u.mV, V_reset=-60 * u.mV,
                    tau_syn_ex=0.2 * u.ms, tau_syn_in=2 * u.ms, I_e=0 * u.pA, t_ref=2 * u.ms)
    states = States(V_m=Init(-70 * u.mV), g_ex=Init(0 * u.nS), g_in=Init(0 * u.nS))
    inputs = Ports(ex=Event(unit=u.nS, into='g_ex'),    # neuron-owned receptor ports (D-b)
                   in_=Event(unit=u.nS, into='g_in'),
                   I=Current(u.pA, latch=True))        # NEST CurrentEvent: next step (D-a)
    outputs = Ports(spike=Event())
    spiking = SpikeRule(condition=lambda s, p: s.V_m >= p.V_th,
                        jump=lambda s, p: s.replace(V_m=p.V_reset),
                        refractory=Clamp('V_m', 't_ref'))
    integrator = RK(order=4, substeps=2)  # fixed substeps; 'exact' only for linear models

    def derivatives(self, s, p, inp, ctx):  # continuous part (g_ex, g_in decay here)
        V = u.math.minimum(s.V_m, p.V_th)
        I_syn = s.g_ex * (V - p.E_ex) + s.g_in * (V - p.E_in)
        return {'V_m': (-p.g_L * (V - p.E_L) - I_syn + p.I_e + inp.I) / p.C_m,
                'g_ex': -s.g_ex / p.tau_syn_ex,
                'g_in': -s.g_in / p.tau_syn_in}
    # on_input: the default jump g += w of every receptor port


EX_TIMES = [1.0, 3.0, 3.0, 7.5]  # ms; a doublet at 3 ms (multiplicity 2)
IN_TIMES = [5.0]


def _example_net(model, *, w_ex=6.0, w_in=8.0, I_e=0.0):
    net = Network(dt=DT * u.ms)
    n = net.add(model, 2, params=dict(I_e=np.array([0.0, I_e]) * u.pA), name='n')
    sg_ex = net.add(spike_generator, 1, params=dict(spike_times=EX_TIMES * u.ms), name='sg_ex')
    sg_in = net.add(spike_generator, 1, params=dict(spike_times=IN_TIMES * u.ms), name='sg_in')
    net.connect(sg_ex, n, port='ex', weight=w_ex * u.nS)
    net.connect(sg_in, n, port='in_', weight=w_in * u.nS)
    rec = net.observe(n, 'V_m', 'g_ex', 'g_in', 'spike')
    return net, rec


def _run(model, precision='float64', **kw):
    net, rec = _example_net(model, **kw)
    res = net.compile(precision).runner().simulate(12.0 * u.ms)
    return {k: np.asarray(u.get_mantissa(res[rec][k])) for k in ('V_m', 'g_ex', 'g_in', 'spike')}


class TestArchitectureExample:
    def test_declaration(self):
        assert iaf_cond_exp_doc.inputs['ex'].into == 'g_ex'
        assert iaf_cond_exp_doc.inputs['in_'].into == 'g_in'
        assert list(iaf_cond_exp_doc.all_state_decl())[:3] == ['V_m', 'g_ex', 'g_in']

    def test_compiles_and_runs(self, x64):
        out = _run(iaf_cond_exp_doc)
        g = out['g_ex'][:, 0]
        # spike at 1.0 ms -> arrives one step later (delay dt), jump g_ex += 6 nS after propagation
        k = int(round(1.0 / DT))  # observed step index whose end value holds the jump
        assert g[k - 1] == 0.0 and g[k] == pytest.approx(6.0, abs=1e-12)
        # the doublet at 3 ms: multiplicity carried (one jump of 2 * 6 nS)
        k3 = int(round(3.0 / DT))
        assert g[k3] - g[k3 - 1] * np.exp(-DT / 0.2) == pytest.approx(12.0, rel=1e-6)
        assert out['g_in'][int(round(5.0 / DT)), 0] == pytest.approx(8.0, abs=1e-12)
        assert np.all(out['g_ex'] >= 0.0) and np.all(np.isfinite(out['V_m']))

    def test_equals_builtin_iaf_cond_exp_subthreshold(self, x64):
        # same equations, same RK(4, 2): the example reproduces the library model
        ours, ref = _run(iaf_cond_exp_doc), _run(iaf_cond_exp)
        for k in ('V_m', 'g_ex', 'g_in'):
            np.testing.assert_allclose(ours[k], ref[k], rtol=1e-12, atol=1e-12)
        assert ours['spike'].sum() == 0 == ref['spike'].sum()

    def test_spiking_neuron_runs(self):
        out = _run(iaf_cond_exp_doc, precision='float32', I_e=1000.0)
        assert out['spike'][:, 1].sum() >= 1 and out['spike'][:, 0].sum() == 0

    def test_previous_example_synapse_into_receptor_port_raises(self):
        # K0 wrote ``synapse=Expon(...)`` into iaf_cond_exp.ex: kinetics would be applied twice
        for model in (iaf_cond_exp, iaf_cond_exp_doc, aeif_cond_alpha, iaf_psc_alpha):
            net = Network(dt=DT * u.ms)
            sg = net.add(spike_generator, 1)
            n = net.add(model, 1)
            w = 1.0 * (u.pA if model is iaf_psc_alpha else u.nS)
            with pytest.raises(PortError, match=rf'{model.__name__}\.ex: .*neuron-owned receptor'):
                net.connect(sg, n, port='ex', weight=w, synapse=Expon(5.0 * u.ms))


# --------------------------------------------------------------------------- declarations

class TestDeclarationChecks:
    def test_migrated_models(self):
        assert iaf_psc_alpha.inputs['ex'] == Event(unit=u.pA, into='I_syn_ex')
        assert iaf_psc_alpha.inputs['in_'] == Event(unit=u.pA, into='I_syn_in')
        for m in (iaf_cond_exp, aeif_cond_alpha):
            assert m.inputs['ex'] == Event(unit=u.nS, into='g_ex')
            assert m.inputs['in_'] == Event(unit=u.nS, into='g_in')
            assert not any(isinstance(k, Conductance) for k in m.inputs.values())

    def test_undeclared_state(self):
        with pytest.raises(PortError, match=r"into 'g_nmda', which is not a declared state"):
            class _Bad(NeuronModel):  # noqa: F841
                states = States(V=Init(0.0 * u.mV))
                inputs = Ports(ex=Event(unit=u.nS, into='g_nmda'))

    def test_dimension_mismatch(self):
        with pytest.raises(PortError, match='does not match the dimension'):
            class _Bad(NeuronModel):  # noqa: F841
                states = States(g=Init(0.0 * u.nS))
                inputs = Ports(ex=Event(unit=u.pA, into='g'))
        with pytest.raises(PortError, match='does not match the dimension'):
            class _BadTimed(NeuronModel):  # noqa: F841
                states = States(g=Init(0.0 * u.nS))
                inputs = Ports(ex=TimedEvent(4, unit=u.mV, into='g'))

    def test_integer_state(self):
        with pytest.raises(PortError, match='float state'):
            class _Bad(NeuronModel):  # noqa: F841
                states = States(n=Init(0, dtype=np.int32))
                inputs = Ports(ex=Event(into='n'))

    def test_same_dimension_other_unit_and_callable_init_accepted(self):
        class _Ok(NeuronModel):
            states = States(g=Init(0.0 * u.nS), h=Init(lambda p: 0.0 * u.uS))
            inputs = Ports(a=Event(unit=u.uS, into='g'), b=Event(unit=u.nS, into='h'))
            outputs = Ports(spike=Event(into='g'))  # into on an output port: no effect

            def derivatives(self, s, p, inp, ctx):
                return {}

        assert _Ok.inputs['a'].into == 'g'

    def test_with_capacity_keeps_into(self):
        class _Timed(iaf_psc_exp_ps):
            inputs = Ports(syn=TimedEvent(16, unit=u.pA, into='I_syn_ex'),
                           I=Current(u.pA, latch=True))

        k8 = _Timed.with_capacity(8)
        assert k8.inputs['syn'].capacity == 8 and k8.inputs['syn'].into == 'I_syn_ex'
        # the library model's sign-routed ``syn`` port is not a receptor port of one state
        assert iaf_psc_exp_ps.inputs['syn'].into is None
        assert iaf_psc_exp_ps.with_capacity(4).inputs['syn'].into is None


# --------------------------------------------------------------------------- the default jump

class _TwoIntoOne(NeuronModel):
    """Two receptor ports into one state, one plain event port, exp-Euler decay."""

    params = Params(tau=1.0 * u.ms)
    states = States(g=Init(0.0 * u.nS), V=Init(0.0 * u.mV))
    inputs = Ports(a=Event(unit=u.nS, into='g'), b=Event(unit=u.nS, into='g'),
                   d=Event(unit=u.mV))
    outputs = Ports()
    integrator: ClassVar[Integrator] = RK(order=1, substeps=1)

    def derivatives(self, s, p, inp, ctx):
        return {'g': -s.g / p.tau, 'V': 0.0 * u.mV / u.ms}


class TestDefaultOnInput:
    def test_jumps_after_propagation_and_sum(self, x64):
        m = _TwoIntoOne(1, precision='float64')
        ctx = StepContext.create(0.0 * u.ms, DT * u.ms)
        m.update(ctx, {'a': 2.0 * u.nS, 'b': 3.0 * u.nS, 'd': 7.0 * u.mV})
        # jump after propagation: no decay in the arrival step; ``d`` is not a receptor port
        assert float(m.dyn_states['g'].value.to_decimal(u.nS)[0]) == 5.0
        assert float(m.dyn_states['V'].value.to_decimal(u.mV)[0]) == 0.0
        m.update(ctx)
        g = float(m.dyn_states['g'].value.to_decimal(u.nS)[0])
        assert g == pytest.approx(5.0 * (1.0 - DT), rel=1e-12)

    def test_no_receptor_ports_returns_state(self):
        class _Plain(NeuronModel):
            states = States(V=Init(0.0 * u.mV))
            inputs = Ports(d=Event(unit=u.mV))

            def derivatives(self, s, p, inp, ctx):
                return {}

        s = Record(V=1.0 * u.mV)
        assert _Plain(1).on_input(s, Record(), Record(d=2.0 * u.mV), None) is s

    def test_iaf_cond_exp_jump_unchanged(self, x64):
        m = iaf_cond_exp(1, precision='float64')
        m.update(StepContext.create(0.0 * u.ms, DT * u.ms), {'ex': 5.0 * u.nS, 'in_': 2.0 * u.nS})
        assert float(m.dyn_states['g_ex'].value.to_decimal(u.nS)[0]) == 5.0
        assert float(m.dyn_states['g_in'].value.to_decimal(u.nS)[0]) == 2.0


# --------------------------------------------------------------------------- connect-time checks

class TestConnectChecks:
    def _net(self, model):
        net = Network(dt=DT * u.ms)
        return net, net.add(spike_generator, 1), net.add(model, 2)

    def test_weight_unit_mismatch(self):
        net, sg, n = self._net(iaf_cond_exp)
        with pytest.raises(UnitMismatchError):
            net.connect(sg, n, port='ex', weight=1.0 * u.pA)
        net, sg, n = self._net(iaf_psc_alpha)
        with pytest.raises(UnitMismatchError):
            net.connect(sg, n, port='ex', weight=1.0 * u.nS)

    def test_conductance_receptor_sign(self):
        for model in (iaf_cond_exp, aeif_cond_alpha):
            net, sg, n = self._net(model)
            with pytest.raises(PortError, match='non-negative'):
                net.connect(sg, n, port='ex', weight=-1.0 * u.nS)
            with pytest.raises(PortError, match='clip it with low=0'):
                net.connect(sg, n, port='in_', weight=Normal(1.0 * u.nS, 0.5 * u.nS))
            with pytest.raises(PortError, match='non-negative'):
                net.connect(sg, n, port='ex', weight=Uniform(-1.0 * u.nS, 1.0 * u.nS))
            net.connect(sg, n, port='ex', weight=Normal(1.0 * u.nS, 0.5 * u.nS, low=0.0 * u.nS))

    def test_psc_receptor_keeps_signed_weights(self):
        net, sg, n = self._net(iaf_psc_alpha)
        net.connect(sg, n, port='in_', weight=-50.0 * u.pA)
        net.connect(sg, n, port='ex', weight=Normal(0.0 * u.pA, 10.0 * u.pA))

    def test_multiplicity_carried_iaf_psc_alpha(self, x64):
        # a source with max_multiplicity > 1: a doublet is one jump of 2 w
        def run(times):
            net = Network(dt=DT * u.ms)
            sg = net.add(spike_generator, 1, params=dict(spike_times=times * u.ms))
            n = net.add(iaf_psc_alpha, 1)
            net.connect(sg, n, port='ex', weight=100.0 * u.pA)
            rec = net.observe(n, 'I_syn_ex')
            res = net.compile('float64').runner().simulate(3.0 * u.ms)
            return np.asarray(res[rec]['I_syn_ex'].to_decimal(u.pA))[:, 0]

        one, two = run([1.0]), run([1.0, 1.0])
        assert np.abs(one).max() > 0
        np.testing.assert_allclose(two, 2.0 * one, rtol=1e-12, atol=1e-12)

    def test_plain_event_ports_still_accept_synapse_errors(self):
        # a plain value port keeps projection-owned kinetics (unchanged)
        net = Network(dt=DT * u.ms)
        sg = net.add(spike_generator, 1)
        n = net.add(iaf_cond_exp, 1)
        p = net.connect(sg, n, port='I', weight=1.0 * u.pA, synapse=Expon(2.0 * u.ms))
        assert isinstance(p.synapse, Expon)

    def test_describe_shows_receptor_state(self):
        net, sg, n = self._net(iaf_cond_exp)
        net.connect(sg, n, port='ex', weight=1.0 * u.nS)
        text = net.compile().describe(print_=False)
        assert 'inputs: ex=event(into g_ex) in_=event(into g_in) I=current(latch)' in text


class TestExactModelStillDeclaresReceptors:
    def test_iaf_psc_alpha_step_ignores_default(self):
        # hand-written Exact() steps apply their own jumps; on_input default is not used
        assert iaf_psc_alpha.integrator == Exact()
        m = iaf_psc_alpha(1)
        m.update(StepContext.create(0.0 * u.ms, DT * u.ms), {'ex': 100.0 * u.pA})
        assert float(m.dyn_states['I_syn_ex'].value.to_decimal(u.pA)[0]) == 0.0
        assert float(m.dyn_states['dI_syn_ex'].value.to_decimal(u.pA / u.ms)[0]) > 0.0
        assert jnp.isfinite(m.dyn_states['V_m'].value.to_decimal(u.mV)).all()
