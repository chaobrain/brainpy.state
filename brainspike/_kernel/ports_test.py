# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import dataclasses

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel import (
    Conductance,
    Current,
    Event,
    Graded,
    LogNormal,
    Normal,
    PortError,
    Ports,
    TimedEvent,
    TimedEventPayload,
    Uniform,
    UnitMismatchError,
    check_capacity,
    check_connection,
    check_kind_compatible,
    check_single_owner,
    check_weight_sign,
    check_weight_unit,
    expected_weight_unit,
    is_receptor,
)


# ---------------------------------------------------------------- kinds

class TestKinds:
    def test_event(self):
        e = Event()
        assert e.kind == 'event' and e.max_multiplicity == 1 and e.binary
        assert e.unit == u.UNITLESS and e.into is None and not e.nonnegative_weights
        assert not Event(None).binary and Event(None).max_multiplicity is None
        assert Event(3).max_multiplicity == 3
        assert Event(unit=u.pA).unit == u.pA
        for bad in (0, -1, 1.5, True):
            with pytest.raises(PortError):
                Event(bad)

    def test_event_zeros(self):
        z = Event(unit=u.pA).zeros(3)
        assert z.unit == u.pA and z.shape == (3,)
        z = Event().zeros(0)
        assert z.shape == (0,) and not isinstance(z, u.Quantity)

    def test_frozen_and_eq(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Event().max_multiplicity = 2
        assert Current() == Current(u.pA)
        assert Current(u.nA) != Current(u.pA)

    def test_timed_event(self):
        t = TimedEvent(4, unit=u.pA)
        assert t.kind == 'timed_event' and t.capacity == 4
        z = t.zeros(2)
        assert isinstance(z, TimedEventPayload) and z.capacity == 4
        assert z.weights.unit == u.pA
        for bad in (0, 2.5, False):
            with pytest.raises(PortError):
                TimedEvent(bad)

    def test_current(self):
        assert Current().unit == u.pA
        assert Current(u.nA).zeros(2, jnp.float32).unit == u.nA
        with pytest.raises(UnitMismatchError):
            Current(u.nS)
        with pytest.raises(TypeError):
            Current(1 * u.pA)

    def test_conductance(self):
        c = Conductance(E_rev=0 * u.mV, gate=lambda V: 1.0)
        assert c.kind == 'conductance' and c.nonnegative_weights
        assert Conductance(E_rev='E_ex').E_rev == 'E_ex'
        with pytest.raises(UnitMismatchError):
            Conductance(u.pA)
        with pytest.raises(UnitMismatchError):
            Conductance(E_rev=0 * u.nS)
        with pytest.raises(TypeError, match='gate'):
            Conductance(gate=3)
        assert c.into is None

    def test_filtered_flag_removed_everywhere(self):
        # D-b: one receptor encoding (Event/TimedEvent ``into=``); no ``filtered`` flag left
        for make in (lambda: Event(filtered=True), lambda: TimedEvent(2, filtered=True),
                     lambda: Current(filtered=True), lambda: Conductance(filtered=True),
                     lambda: Graded(u.pA, filtered=True)):
            with pytest.raises(TypeError):
                make()
        for kind in (Event(), TimedEvent(2), Current(), Conductance(), Graded(None)):
            assert not hasattr(kind, 'filtered')
        with pytest.raises(TypeError):
            Current(u.pA, False)  # the positional ``filtered`` slot is gone (latch is keyword-only)

    def test_receptor_event(self):
        ex = Event(unit=u.nS, into='g_ex')
        assert ex.into == 'g_ex' and is_receptor(ex) and ex.nonnegative_weights
        assert ex != Event(unit=u.nS) and ex == Event(unit=u.nS, into='g_ex')
        assert hash(ex) == hash(Event(unit=u.nS, into='g_ex'))
        psc = Event(unit=u.pA, into='I_syn_ex')
        assert is_receptor(psc) and not psc.nonnegative_weights  # signed PSC weights
        assert not Event(unit=u.nS).nonnegative_weights  # not a receptor: no sign rule
        assert not is_receptor(Event(unit=u.pA))
        for bad in ('', '1g', 'g ex', 3, b'g'):
            with pytest.raises(PortError, match='into'):
                Event(unit=u.nS, into=bad)
        with pytest.raises(TypeError):
            Event(1, u.nS, 'g_ex')  # into is keyword-only

    def test_receptor_timed_event(self):
        t = TimedEvent(4, unit=u.nS, into='g_ex')
        assert t.into == 'g_ex' and is_receptor(t) and t.nonnegative_weights
        assert not is_receptor(TimedEvent(4, unit=u.pA))
        assert not TimedEvent(4, unit=u.pA, into='I_syn').nonnegative_weights
        with pytest.raises(PortError, match='into'):
            TimedEvent(4, unit=u.nS, into='not a name')

    def test_value_kinds_are_never_receptors(self):
        for kind in (Current(), Conductance(), Graded(u.nS)):
            assert not is_receptor(kind) and kind.into is None

    def test_graded(self):
        g = Graded(u.mV)
        assert g.kind == 'graded' and g.unit == u.mV
        assert Graded(None).unit == u.UNITLESS


class TestPorts:
    def test_mapping_and_require(self):
        p = Ports(ex=Conductance(), I=Current())
        assert list(p) == ['ex', 'I'] and len(p) == 2
        assert p.require('ex') is p['ex']
        assert 'ex=Conductance' in repr(p)
        r = repr(Ports(ex=Event(unit=u.nS, into='g_ex'), I=Current(), d=Event()))
        assert r == "Ports(ex=Event(into='g_ex'), I=Current(latch=True), d=Event)"
        with pytest.raises(PortError, match=r"E->I: port 'in' does not exist; available: \['ex', 'I'\]"):
            p.require('in', where='E->I')

    def test_type_check(self):
        with pytest.raises(TypeError, match="'x'"):
            Ports(x=u.pA)

    def test_zeros(self):
        z = Ports(spike=Event(), I=Current(), te=TimedEvent(2)).zeros(5)
        assert z['spike'].shape == (5,) and z['I'].unit == u.pA
        assert z['te'].valid.shape == (5, 2)


# ---------------------------------------------------------------- payload

def ms(x):
    return jnp.asarray(x, jnp.float32) * u.ms


class TestTimedEventPayload:
    def test_empty(self):
        p = TimedEventPayload.empty(3, 2, weight_unit=u.pA)
        assert p.capacity == 2 and p.count.tolist() == [0, 0, 0]
        assert p.overflow.tolist() == [0, 0, 0]
        assert bool(jnp.all(jnp.isinf(p.offsets.mantissa)))

    def test_pack_sorts_and_overflows(self):
        p = TimedEventPayload.pack(
            ms([[0.3, 0.1, 0.2], [0.05, 0.0, 0.0]]),
            jnp.array([[3.0, 1.0, 2.0], [7.0, 8.0, 9.0]]) * u.pA,
            [[True, True, True], [True, False, False]],
            capacity=2,
        )
        np.testing.assert_allclose(p.offsets.mantissa[0], [0.1, 0.2])
        np.testing.assert_allclose(p.weights.mantissa[0], [1.0, 2.0])
        assert p.overflow.tolist() == [1, 0]
        assert p.count.tolist() == [2, 1]
        # invalid slot: inf offset, zero weight
        assert np.isinf(p.offsets.mantissa[1, 1]) and float(p.weights.mantissa[1, 1]) == 0
        assert float(p.weights.mantissa[1, 0]) == 7.0

    def test_pack_pads(self):
        p = TimedEventPayload.pack(ms([[0.2]]), jnp.array([[1.0]]), [[True]], capacity=3)
        assert p.capacity == 3 and p.count.tolist() == [1]
        assert p.overflow.tolist() == [0]

    def test_pack_offsets_must_be_time(self):
        with pytest.raises(UnitMismatchError):
            TimedEventPayload.pack(jnp.zeros((1, 1)), jnp.ones((1, 1)), [[True]], 1)

    def test_merge(self):
        a = TimedEventPayload.pack(ms([[0.4, 0.1]]), jnp.array([[4., 1.]]) * u.pA, [[True, True]], 2)
        b = TimedEventPayload.pack(ms([[0.2, 0.3]]), jnp.array([[2., 3.]]) * u.nA, [[True, False]], 2)
        m = a.merge(b)
        np.testing.assert_allclose(m.offsets.mantissa[0], [0.1, 0.2])
        np.testing.assert_allclose(m.weights.to_decimal(u.pA)[0], [1.0, 2000.0])
        assert m.overflow.tolist() == [1]  # 0.4 dropped, counted
        m2 = m.merge(m)
        assert m2.overflow.tolist() == [1 + 1 + 2]

    def test_merge_unit_mismatch(self):
        a = TimedEventPayload.empty(1, 1, weight_unit=u.pA)
        b = TimedEventPayload.empty(1, 1, weight_unit=u.nS)
        with pytest.raises(UnitMismatchError):
            a.merge(b)

    def test_jit_and_pytree(self):
        @jax.jit
        def f(p):
            return p.merge(p)

        p = TimedEventPayload.pack(ms([[0.1]]), jnp.array([[1.0]]), [[True]], 1)
        out = f(p)
        assert out.overflow.tolist() == [1]
        assert len(jax.tree_util.tree_leaves(p)) == 4

    def test_zero_targets(self):
        p = TimedEventPayload.empty(0, 3)
        assert p.count.shape == (0,)
        m = p.merge(p)
        assert m.valid.shape == (0, 3)


# ---------------------------------------------------------------- validation

class TestCompatibility:
    @pytest.mark.parametrize('src,dst', [
        (Event(), Event()),
        (Event(), TimedEvent(2)),
        (TimedEvent(2), TimedEvent(2)),
        (TimedEvent(2), Event()),
        (Current(), Current()),
        (Conductance(), Conductance()),
        (Graded(u.mV), Graded(u.mV)),
        (Graded(u.pA), Current()),
        (Graded(u.nS), Conductance()),
        # neuron-owned receptor ports receive events directly (the neuron owns the kinetics)
        (Event(), Event(unit=u.nS, into='g_ex')),
        (TimedEvent(2), Event(unit=u.nS, into='g_ex')),
        (Event(), TimedEvent(2, unit=u.pA, into='I_syn')),
    ])
    def test_allowed(self, src, dst):
        check_kind_compatible(src, dst)

    @pytest.mark.parametrize('src,dst', [
        (Event(), Current()),
        (Event(), Conductance()),
        # the K0 special case (events straight into a filtered value port) is gone (D-b)
        (TimedEvent(2), Conductance()),
        (TimedEvent(2), Current()),
        (Current(), Event(unit=u.pA, into='I_syn')),
        (Graded(u.nS), Event(unit=u.nS, into='g_ex')),
        (Event(), Graded(None)),
        (Current(), Event()),
        (Conductance(), Current()),
        (Graded(u.mV), Event()),
    ])
    def test_rejected(self, src, dst):
        with pytest.raises(PortError, match='cannot feed'):
            check_kind_compatible(src, dst, where='A->B')


class TestWeightUnit:
    def test_expected(self):
        assert expected_weight_unit(Event(), Event(unit=u.pA)) == u.pA
        assert u.get_dim(expected_weight_unit(Graded(u.mV), Current(u.pA))) == u.get_dim(u.nS)
        assert u.get_dim(expected_weight_unit(Current(), Current())) == u.get_dim(u.UNITLESS)

    def test_check(self):
        check_weight_unit(87.8 * u.pA, Event(), Event(unit=u.pA))
        check_weight_unit(0.1 * u.nA, Event(), Event(unit=u.pA))
        check_weight_unit(Normal(1 * u.nS, 0.1 * u.nS), Event(), Conductance())
        check_weight_unit(2.0, Current(), Current())
        check_weight_unit(1.0, Event(), Event())
        with pytest.raises(UnitMismatchError, match='E->I: weight'):
            check_weight_unit(1 * u.mV, Event(), Event(unit=u.pA), where='E->I')
        with pytest.raises(UnitMismatchError):
            check_weight_unit(1.0, Event(), Conductance())
        with pytest.raises(UnitMismatchError):
            check_weight_unit(Normal(1 * u.mV, 1 * u.mV), Event(), Conductance())


class TestWeightSign:
    def test_signed_ports_accept_anything(self):
        check_weight_sign(-1 * u.pA, Event(unit=u.pA))
        check_weight_sign(Normal(0 * u.pA, 1 * u.pA), Current())

    def test_conductance_values(self):
        check_weight_sign(jnp.array([0.0, 1.0]) * u.nS, Conductance())
        check_weight_sign(jnp.zeros((0,)) * u.nS, Conductance())
        with pytest.raises(PortError, match='non-negative'):
            check_weight_sign(jnp.array([1.0, -0.1]) * u.nS, Conductance(), where='E->I')

    def test_conductance_receptor_values(self):
        ex = Event(unit=u.nS, into='g_ex')
        check_weight_sign(jnp.array([0.0, 1.0]) * u.nS, ex)
        with pytest.raises(PortError, match='event weights must be non-negative'):
            check_weight_sign(jnp.array([1.0, -0.1]) * u.nS, ex, where='E->I.ex')
        with pytest.raises(PortError, match='clip it with low=0'):
            check_weight_sign(Normal(1 * u.nS, 0.1 * u.nS), ex)
        check_weight_sign(Normal(1 * u.nS, 0.1 * u.nS, low=0 * u.nS), ex)
        with pytest.raises(PortError, match='non-negative'):
            check_weight_sign(-1 * u.nS, TimedEvent(2, unit=u.nS, into='g'))
        # a current receptor keeps signed weights (NEST: inhibitory PSC weights < 0)
        check_weight_sign(Normal(0 * u.pA, 1 * u.pA), Event(unit=u.pA, into='I_syn_ex'))

    def test_conductance_distributions(self):
        check_weight_sign(Uniform(0 * u.nS, 1 * u.nS), Conductance())
        check_weight_sign(LogNormal(0.0, 1.0, unit=u.nS), Conductance())
        check_weight_sign(Normal(1 * u.nS, 0.1 * u.nS, low=0 * u.nS), Conductance())
        with pytest.raises(PortError, match='clip it with low=0'):
            check_weight_sign(Normal(1 * u.nS, 0.1 * u.nS), Conductance())
        with pytest.raises(PortError):
            check_weight_sign(Uniform(-1 * u.nS, 1 * u.nS), Conductance())

    def test_traced_values_are_skipped(self):
        def f(w):
            check_weight_sign(w * u.nS, Conductance())
            return w

        jax.jit(f)(jnp.array([-1.0]))


class TestCapacityAndOwner:
    def test_capacity(self):
        assert check_capacity(TimedEvent(4), 4)
        assert check_capacity(Event(), 10 ** 6)
        with pytest.raises(PortError, match='capacity is 4'):
            check_capacity(TimedEvent(4), 5)
        assert check_capacity(TimedEvent(4), 5, strict=False) is False

    def test_single_owner(self):
        ex = Event(unit=u.nS, into='g_ex')
        check_single_owner(ex, synapse_filters=False)
        check_single_owner(Conductance(), synapse_filters=True)
        check_single_owner(Current(), synapse_filters=True)
        check_single_owner(Event(unit=u.nS), synapse_filters=True)  # not a receptor
        with pytest.raises(PortError, match=r"P -> iaf_cond_exp\.ex: .*neuron-owned receptor.*'g_ex'"):
            check_single_owner(ex, synapse_filters=True, where='P -> iaf_cond_exp.ex')
        with pytest.raises(PortError, match='receptor'):
            check_single_owner(TimedEvent(2, unit=u.pA, into='I'), synapse_filters=True)


class TestCheckConnection:
    def test_ok(self):
        assert check_connection(Event(), Event(unit=u.pA), 10 * u.pA)
        assert check_connection(Event(), TimedEvent(2, u.pA), 10 * u.pA, max_events=2)
        assert not check_connection(Event(), TimedEvent(2, u.pA), max_events=3, strict_capacity=False)

    def test_each_check_fires(self):
        with pytest.raises(PortError, match='cannot feed'):
            check_connection(Event(), Current())
        with pytest.raises(PortError, match='receptor'):
            check_connection(Event(), Event(unit=u.nS, into='g'), synapse_filters=True)
        with pytest.raises(UnitMismatchError):
            check_connection(Event(), Event(unit=u.pA), 1 * u.nS)
        with pytest.raises(PortError, match='non-negative'):
            check_connection(Conductance(), Conductance(), -1.0)
        with pytest.raises(PortError, match='capacity'):
            check_connection(TimedEvent(1), TimedEvent(1), max_events=2)
