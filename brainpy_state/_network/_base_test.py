# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest
import brainstate
import saiunit as u

from brainpy_state._network._base import Network
from brainpy_state import LIF


class TestNetworkSkeleton(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_network_is_a_brainstate_module(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(10)

        net = Net()
        self.assertIsInstance(net, brainstate.nn.Module)
        self.assertIs(net.pop, net.nodes()[('pop',)])

    def test_module_attribute_is_brainpy_state(self):
        self.assertEqual(Network.__module__, 'brainpy.state')


from brainpy_state._base import Dynamics
from brainpy_state._brainpy.projection import Projection


class _TraceNode(Dynamics):
    def __init__(self, calls, tag):
        super().__init__(in_size=1)
        self._calls = calls
        self._tag = tag

    def init_state(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        self._calls.append(self._tag)


class _TraceProj(Projection):
    def __init__(self, calls, tag):
        super().__init__()
        self._calls = calls
        self._tag = tag

    def update(self, *args, **kwargs):
        self._calls.append(self._tag)


class TestNetworkUpdateOrder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_projections_run_before_dynamics(self):
        calls = []

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.neuron = _TraceNode(calls, 'neuron')
                self.proj = _TraceProj(calls, 'proj')

        net = Net()
        net.update()
        self.assertEqual(calls, ['proj', 'neuron'])

    def test_introspection_properties(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.neuron = LIF(5)
                self.proj = _TraceProj([], 'proj')

        net = Net()
        self.assertIn('neuron', net.populations)
        self.assertIn('proj', net.projections)
        self.assertEqual(net.devices, {})


import jax.numpy as jnp


class TestNetworkSimulate(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_simulate_runs_for_loop(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(5)

        net = Net()
        brainstate.nn.init_all_states(net)
        out = net.simulate(1.0 * u.ms)
        self.assertEqual(out, {})

    def test_simulate_update_parity(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net_a = Net()
        net_b = Net()
        brainstate.nn.init_all_states(net_a)
        brainstate.nn.init_all_states(net_b)

        net_a.simulate(0.5 * u.ms)

        times = u.math.arange(0 * u.ms, 0.5 * u.ms, 0.1 * u.ms)
        for t in times:
            with brainstate.environ.context(t=t):
                net_b.update(t)

        self.assertTrue(jnp.allclose(
            u.get_mantissa(net_a.pop.V.value),
            u.get_mantissa(net_b.pop.V.value)))


class TestNetworkMonitor(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_monitor_attribute_path(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net = Net()
        brainstate.nn.init_all_states(net)
        out = net.simulate(0.5 * u.ms, monitor=['pop.V'])
        self.assertIn('pop.V', out)
        # 5 timesteps, 3 neurons
        self.assertEqual(out['pop.V'].shape, (5, 3))

    def test_monitor_callable(self):
        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(3)

        net = Net()
        brainstate.nn.init_all_states(net)
        out = net.simulate(
            0.5 * u.ms,
            monitor={'mean_V': lambda n: jnp.mean(u.get_mantissa(n.pop.V.value))},
        )
        self.assertIn('mean_V', out)
        self.assertEqual(out['mean_V'].shape, (5,))
