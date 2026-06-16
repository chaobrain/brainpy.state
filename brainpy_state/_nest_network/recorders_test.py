# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import brainstate
import jax.numpy as jnp
import brainunit as u

from brainpy_state import LIF
from brainpy_state._nest_network.base import Network
from brainpy_state._nest_network.recorders import Recorder


class TestRecorder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_recorder_string_attr_forwards_state(self):
        captured = {'v': None}

        class FakeDevice:
            def update(self, val=None, **kw):
                captured['v'] = val

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(5)
                self.rec = Recorder(source=self.pop, attr='V',
                                    device=FakeDevice())

        net = Net()
        brainstate.nn.init_all_states(net)
        net.update()
        self.assertIsNotNone(captured['v'])
        self.assertEqual(captured['v'].shape, (5,))

    def test_recorder_callable_attr(self):
        captured = {'spikes': None}

        class FakeDevice:
            def update(self, val=None, **kw):
                captured['spikes'] = val

        class Net(Network):
            def __init__(self):
                super().__init__()
                self.pop = LIF(5)
                self.rec = Recorder(
                    source=self.pop,
                    attr=lambda s: s.get_spike(s.V.value),
                    device=FakeDevice(),
                )

        net = Net()
        brainstate.nn.init_all_states(net)
        net.update()
        self.assertIsNotNone(captured['spikes'])
        self.assertEqual(captured['spikes'].shape, (5,))

    def test_recorder_attr_missing_raises_at_update(self):
        class FakeDevice:
            def update(self, x=None, **kw):
                pass

        pop = LIF(3)
        rec = Recorder(source=pop, attr='nonexistent', device=FakeDevice())
        brainstate.nn.init_all_states(pop)
        with self.assertRaises(AttributeError):
            rec.update()
