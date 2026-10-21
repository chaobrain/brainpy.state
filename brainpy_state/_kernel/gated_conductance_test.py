# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Voltage-gated conductance ports and NMDA delivery (K0 T8).

* the gate is evaluated at every RK substep (vs a gate frozen per step);
* NMDA PSP vs the old ``_brainpy`` ``BioNMDA`` + ``MgBlock`` (unitless-g setup);
* unit invariance, float32/float64, jit / vmap / for_loop, network end to end;
* optional single-neuron comparison against Brian2.
"""
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.connectivity.rules import OneToOne
from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.gates import MgBlock
from brainpy_state._kernel.integrators import RK
from brainpy_state._kernel.model import with_conductance_ports
from brainpy_state._kernel.models import NMDA, aeif_cond_alpha, iaf_cond_exp, iaf_psc_alpha, spike_generator
from brainpy_state._kernel.models._testing import rollout, step_inputs
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.ports import Conductance, PortError
from brainpy_state._kernel.validation.markers import requires_brian2

DT = 0.1 * u.ms

# membrane parameters: no spikes (V_th far above), resting near the steep part of B(V)
P = dict(V_th=100.0 * u.mV, V_reset=-60.0 * u.mV, g_L=16.6667 * u.nS, C_m=250.0 * u.pF,
         E_L=-70.0 * u.mV, I_e=150.0 * u.pA)
E_NMDA = 0.0 * u.mV


def _cell(unit=u.nS, gate=MgBlock()):
    return with_conductance_ports(iaf_cond_exp,
                                  nmda=Conductance(unit, E_rev='E_nmda', gate=gate),
                                  params=dict(E_nmda=E_NMDA))


Cell = _cell()


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _g_sequence(n_steps, dt_ms=0.1):
    # piecewise-constant conductance (nS) with a fast rise and slow decay
    t = np.arange(n_steps) * dt_ms
    g = 40.0 * (1 - np.exp(-np.maximum(t - 2.0, 0) / 2.0)) * np.exp(-np.maximum(t - 2.0, 0) / 30.0)
    return jnp.asarray(g)


def _kernel_V(g_nS, substeps, precision='float64', model=Cell, unit=u.nS, scale=1.0, dt=DT):
    m = model(1, params=P, precision=precision, integrator=RK(4, substeps))
    _, tr = rollout(m, g_nS.shape[0], dt, {'nmda': u.Quantity(g_nS[:, None] * scale, unit=unit)},
                    record=('V_m',))
    return np.asarray(tr['V_m'].to_decimal(u.mV))[:, 0]


def _mg(v):  # Jahr-Stevens, mV
    return 1.0 / (1.0 + 1.2 / 3.57 * jnp.exp(-0.062 * v))


def _reference_V(g_nS, substeps, frozen=False, current=None, dt_ms=0.1):
    """Independent RK4 of C dV/dt = -g_L (V - E_L) + I_nmda + I_e (mantissas: mV, nS, pA, ms).

    ``frozen=True`` evaluates the gate once at the step start (what a port input
    computed outside the substeps would give); ``current(g, V)`` replaces the
    NMDA current (used for the old ``MgBlock``).
    """
    gL, EL, C, Ie, E = 16.6667, -70.0, 250.0, 150.0, 0.0
    h = dt_ms / substeps

    def rhs(V, g, Vgate):
        I = g * _mg(Vgate) * (E - V) if current is None else current(g, V)
        return (-gL * (V - EL) + I + Ie) / C

    def step(V, g):
        V0 = V
        for _ in range(substeps):  # Python-unrolled substeps inside one step
            gv = (lambda v: V0) if frozen else (lambda v: v)
            k1 = rhs(V, g, gv(V))
            k2 = rhs(V + h / 2 * k1, g, gv(V + h / 2 * k1))
            k3 = rhs(V + h / 2 * k2, g, gv(V + h / 2 * k2))
            k4 = rhs(V + h * k3, g, gv(V + h * k3))
            V = V + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return V, V

    _, Vs = jax.lax.scan(step, jnp.asarray(-70.0, g_nS.dtype), g_nS)
    return np.asarray(Vs)


class TestWithConductancePorts:
    def test_declaration(self):
        assert list(Cell.inputs) == ['ex', 'in_', 'I', 'nmda']
        assert 'E_nmda' in Cell.params and issubclass(Cell, iaf_cond_exp)
        assert Cell.__name__ == 'iaf_cond_exp_nmda'
        assert Cell.inputs['nmda'].gate == MgBlock() and not Cell.inputs['nmda'].filtered
        named = with_conductance_ports(aeif_cond_alpha, name='AdExNMDA',
                                       nmda=Conductance(u.nS, E_rev=0.0 * u.mV, gate=MgBlock()))
        assert named.__name__ == 'AdExNMDA'

    def test_validation(self):
        with pytest.raises(TypeError, match='NeuronModel subclass'):
            with_conductance_ports(object, nmda=Conductance(E_rev=0.0 * u.mV))
        with pytest.raises(TypeError, match='accepts_conductance_ports'):
            with_conductance_ports(iaf_psc_alpha, nmda=Conductance(E_rev=0.0 * u.mV))
        with pytest.raises(TypeError, match='at least one'):
            with_conductance_ports(iaf_cond_exp)
        with pytest.raises(TypeError, match='must be a Conductance'):
            with_conductance_ports(iaf_cond_exp, nmda=object())
        with pytest.raises(PortError, match='unfiltered'):
            with_conductance_ports(iaf_cond_exp, nmda=Conductance(E_rev=0.0 * u.mV, filtered=True))
        with pytest.raises(PortError, match='needs E_rev'):
            with_conductance_ports(iaf_cond_exp, nmda=Conductance())
        with pytest.raises(PortError, match='not declared'):
            with_conductance_ports(iaf_cond_exp, nmda=Conductance(E_rev='E_nmda'))
        with pytest.raises(PortError, match='already exists'):
            with_conductance_ports(iaf_cond_exp, ex=Conductance(E_rev=0.0 * u.mV))

    def test_no_extra_input_is_bit_identical_to_base(self, x64):
        n = 300
        inputs = step_inputs(n, 1, {'ex': ([5, 80], [30.0, 50.0], u.nS)}, dtype=jnp.float64)
        params = dict(I_e=300.0 * u.pA)
        base = rollout(iaf_cond_exp(1, params, precision='float64'), n, DT, inputs, record=('V_m',))
        ext = rollout(Cell(1, params, precision='float64'), n, DT, inputs, record=('V_m',))
        np.testing.assert_array_equal(np.asarray(base[1]['V_m'].to_decimal(u.mV)),
                                      np.asarray(ext[1]['V_m'].to_decimal(u.mV)))
        np.testing.assert_array_equal(np.asarray(base[0]['spike']), np.asarray(ext[0]['spike']))

    def test_ungated_port_is_plain_conductance(self, x64):
        plain = _cell(gate=None)
        g = _g_sequence(200)
        Mg0 = _kernel_V(g, 2, model=_cell(gate=MgBlock(Mg=0.0 * u.mM)))
        np.testing.assert_allclose(_kernel_V(g, 2, model=plain), Mg0, rtol=0, atol=1e-12)

    def test_aeif_cond_alpha_accepts_gated_port(self, x64):
        AdEx = with_conductance_ports(aeif_cond_alpha,
                                      nmda=Conductance(u.nS, E_rev=0.0 * u.mV, gate=MgBlock()))
        g = _g_sequence(300)
        params = dict(V_th=-30.0 * u.mV, Delta_T=0.0 * u.mV)
        base = rollout(aeif_cond_alpha(1, params, precision='float64'), 300, DT, record=('V_m',))
        m = AdEx(1, params, precision='float64')
        gated = rollout(m, 300, DT, {'nmda': g[:, None] * u.nS}, record=('V_m',))
        free = with_conductance_ports(aeif_cond_alpha, nmda=Conductance(u.nS, E_rev=0.0 * u.mV))
        ungated = rollout(free(1, params, precision='float64'), 300, DT,
                          {'nmda': g[:, None] * u.nS}, record=('V_m',))
        vb, vg, vu = (np.asarray(t[1]['V_m'].to_decimal(u.mV))[:, 0] for t in (base, gated, ungated))
        assert np.all(vg >= vb - 1e-12) and vg.max() > vb.max() + 0.1
        assert vu.max() > vg.max() + 1.0  # the Mg block removes most of the drive at rest


class TestGatePerSubstep:
    @pytest.mark.parametrize('substeps', [1, 2, 4])
    def test_kernel_equals_live_gate_reference(self, x64, substeps):
        g = _g_sequence(400)
        np.testing.assert_allclose(_kernel_V(g, substeps), _reference_V(g, substeps),
                                   rtol=0, atol=1e-10)

    def test_frozen_gate_differs(self, x64):
        g = _g_sequence(400)
        live = _kernel_V(g, 2)
        frozen = _reference_V(g, 2, frozen=True)
        assert np.max(np.abs(live - frozen)) > 1e-4

    def test_live_gate_converges_frozen_does_not(self, x64):
        # coarse steps (1 ms) so the RK4 error is far above round-off
        g = _g_sequence(100, dt_ms=1.0) * 3.0
        exact = _reference_V(g, 128, dt_ms=1.0)
        live = [np.max(np.abs(_kernel_V(g, k, dt=1.0 * u.ms) - exact)) for k in (1, 2, 4)]
        frozen = [np.max(np.abs(_reference_V(g, k, frozen=True, dt_ms=1.0) - exact))
                  for k in (1, 2, 4)]
        live_ratio = np.array(live[:-1]) / np.array(live[1:])
        assert np.all(live_ratio > 8.0), (live, live_ratio)  # ~16 for RK4
        # freezing the gate is an O(dt) error that substeps cannot remove
        assert frozen[-1] > 0.8 * frozen[0] and frozen[-1] > 100 * live[-1], (frozen, live)


class TestNMDAPSPvsOld:
    def test_psp_matches_old_bionmda_and_mgblock(self, x64):
        from brainpy_state._brainpy.synapse import BioNMDA
        from brainpy_state._brainpy.synouts import MgBlock as OldMgBlock
        n, g_max = 1500, 3.0  # nS
        spikes = np.zeros((n, 1))
        spikes[[20, 22, 24, 400]] = 1.0
        spikes = jnp.asarray(spikes)
        # old: unitless open fraction (g_initializer=0.), unitless MgBlock (V in mV mantissas)
        old = BioNMDA(1, T_dur=0.45 * u.ms, g_initializer=braintools.init.Constant(0.))
        with brainstate.environ.context(dt=DT):
            old.init_state()

            def body(i, sp):
                with brainstate.environ.context(t=i * DT):
                    return old.update(sp > 0)

            g_old = brainstate.transform.for_loop(body, jnp.arange(n), spikes)[:, 0]
        old_mg = OldMgBlock(E=0.0)
        V_ref = _reference_V(g_max * g_old, 2, current=lambda g, V: old_mg.update(g, V))
        # kernel: NMDA receptor -> s, delivered with weight g_max to the gated port
        rec = NMDA(1, params=dict(T_dur=0.45 * u.ms), precision='float64')
        _, tr = rollout(rec, n, DT, {'spike': spikes}, record=('s',))
        V = _kernel_V(g_max * jnp.asarray(tr['s'])[:, 0], 2)
        assert V.max() - V.min() > 1.0  # a real PSP
        np.testing.assert_allclose(V, V_ref, rtol=0, atol=1e-10)


def _nmda_net(weight, unit=u.nS, precision='float64', nmda_params=None, batch=None,
              cell=None, n_post=2):
    net = Network(dt=DT, seed=3)
    sg = net.add(spike_generator, 2, params=dict(spike_times=np.array([1.0, 1.2, 1.4, 20.0]) * u.ms))
    rec = net.add(NMDA, 2, params=nmda_params)
    post = net.add(cell or _cell(unit), n_post, params=P)
    net.connect(sg, rec, OneToOne(), port='spike')
    net.connect(rec, post, port='nmda', weight=weight, delay=0.0 * u.ms)
    o = net.observe(post, 'V_m')
    o2 = net.observe(rec, 's')
    res = net.compile(precision).runner(batch=batch).simulate(60.0 * u.ms)
    return np.asarray(res[o]['V_m'].to_decimal(u.mV)), np.asarray(res[o2]['s'])


class TestNetwork:
    def test_end_to_end_matches_chained_rollouts(self, x64):
        V, s = _nmda_net(2.0 * u.nS)
        # all-to-all from 2 identical pools: g = 2 * 2 nS * s
        np.testing.assert_array_equal(V[:, 0], V[:, 1])
        V_chain = _kernel_V(4.0 * jnp.asarray(s[:, 0]), 2)
        np.testing.assert_allclose(V[:, 0], V_chain, rtol=0, atol=1e-10)
        assert V.max() > -63.0

    def test_unit_invariance_weight_and_port_unit(self, x64):
        ref, _ = _nmda_net(2.0 * u.nS)
        in_uS, _ = _nmda_net(0.002 * u.uS)
        np.testing.assert_allclose(in_uS, ref, rtol=1e-12, atol=1e-10)
        port_uS, _ = _nmda_net(0.002 * u.uS, unit=u.uS)
        np.testing.assert_allclose(port_uS, ref, rtol=1e-12, atol=1e-10)
        rates_si, _ = _nmda_net(2.0 * u.nS, nmda_params=dict(alpha1=2000.0 / u.second,
                                                            beta1=10.0 / u.second))
        np.testing.assert_allclose(rates_si, ref, rtol=1e-12, atol=1e-10)

    def test_float32_close_to_float64(self, x64):
        ref, _ = _nmda_net(2.0 * u.nS)
        f32, _ = _nmda_net(2.0 * u.nS, precision='float32')
        assert f32.dtype == np.float32
        np.testing.assert_allclose(f32, ref, atol=2e-3)

    def test_batched_runner_vmap(self):
        ref, _ = _nmda_net(2.0 * u.nS, precision='float32')
        Vb, sb = _nmda_net(2.0 * u.nS, precision='float32', batch=2)
        assert Vb.shape == (2,) + ref.shape
        np.testing.assert_allclose(Vb[0], ref, atol=1e-5)
        np.testing.assert_allclose(Vb[1], ref, atol=1e-5)

    def test_event_source_into_unfiltered_port_rejected(self):
        net = Network(dt=DT)
        sg = net.add(spike_generator, 1)
        post = net.add(Cell, 1)
        with pytest.raises(PortError, match='cannot feed'):
            net.connect(sg, post, port='nmda', weight=1.0 * u.nS)

    def test_negative_gmax_rejected(self):
        net = Network(dt=DT)
        rec = net.add(NMDA, 1)
        post = net.add(Cell, 1)
        with pytest.raises(PortError, match='non-negative'):
            net.connect(rec, post, port='nmda', weight=-1.0 * u.nS)

    def test_wrong_weight_unit_rejected(self):
        from brainpy_state._kernel.units import UnitMismatchError
        net = Network(dt=DT)
        rec = net.add(NMDA, 1)
        post = net.add(Cell, 1)
        with pytest.raises(UnitMismatchError):
            net.connect(rec, post, port='nmda', weight=1.0 * u.pA)


class TestTransforms:
    def test_conductance_port_current_jit_and_vmap(self, x64):
        m = Cell(3, params=P, precision='float64')
        s, p = m.read()
        inp = m.prepare_inputs({'nmda': jnp.array([0.0, 5.0, 10.0]) * u.nS})
        V = jnp.array([-70.0, -40.0, -10.0]) * u.mV
        I = m.conductance_port_current(inp, V, p)
        expect = np.array([0.0, 5.0, 10.0]) * _mg(np.array([-70.0, -40.0, -10.0])) * \
            (0.0 - np.array([-70.0, -40.0, -10.0]))
        np.testing.assert_allclose(np.asarray(I.to_decimal(u.pA)), expect, rtol=1e-12)
        jitted = jax.jit(lambda v: m.conductance_port_current(inp, v, p).to_decimal(u.pA))(V)
        np.testing.assert_allclose(np.asarray(jitted), expect, rtol=1e-12)
        vm = jax.vmap(lambda v: m.conductance_port_current(inp, v, p).to_decimal(u.pA))(
            jnp.stack([V.to_decimal(u.mV)] * 2) * u.mV)
        np.testing.assert_allclose(np.asarray(vm), np.stack([expect] * 2), rtol=1e-12)

    def test_for_loop_rollout_float32(self):
        g = _g_sequence(200).astype(jnp.float32)
        V = _kernel_V(g, 2, precision='float32')
        assert V.dtype == np.float32 and np.all(np.isfinite(V))

    def test_jit_update(self):
        m = Cell(1, params=P)
        ctx = StepContext.create(0.0 * u.ms, DT)
        f = brainstate.transform.jit(lambda g: m.update(ctx, {'nmda': g})['spike'])
        f(jnp.array([20.0]) * u.nS)
        assert float(m.dyn_states['V_m'].value.to_decimal(u.mV)[0]) > -70.0


@requires_brian2
def test_single_neuron_vs_brian2(x64):
    """Brian2 rk4 with the same NMDA conductance held constant over each step."""
    import brian2 as b2
    n = 600
    g = np.asarray(_g_sequence(n))
    ours = _kernel_V(jnp.asarray(g), 1)  # RK(4, 1): same scheme as Brian2's rk4
    b2.start_scope()
    b2.prefs.codegen.target = 'numpy'
    b2.defaultclock.dt = 0.1 * b2.ms
    g_arr = b2.TimedArray(g * b2.nS, dt=0.1 * b2.ms)
    eqs = '''
    dv/dt = (gL * (EL - v) + g_nmda * (E_nmda - v) / (1 + Mg / beta * exp(-alpha * v)) + Ie) / C : volt
    g_nmda : siemens
    '''
    G = b2.NeuronGroup(1, eqs, method='rk4',
                       namespace=dict(gL=16.6667 * b2.nS, EL=-70 * b2.mV, E_nmda=0 * b2.mV,
                                      Mg=1.2 * b2.mmolar, beta=3.57 * b2.mmolar,
                                      alpha=0.062 / b2.mV, Ie=150 * b2.pA, C=250 * b2.pF,
                                      g_arr=g_arr))
    G.v = -70 * b2.mV
    G.run_regularly('g_nmda = g_arr(t)', when='start')
    M = b2.StateMonitor(G, 'v', record=0, when='end')
    b2.run(n * 0.1 * b2.ms)
    theirs = np.asarray(M.v[0] / b2.mV)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-9)
