# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests for the voltmeter recording device (a multimeter preset for V_m)."""
import unittest

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')


class TestVoltmeter(unittest.TestCase):
    def test_importable_from_top_level(self):
        from brainpy_state import voltmeter  # noqa: F401
        from brainpy.state import voltmeter as vm2  # noqa: F401
        self.assertIs(voltmeter, vm2)

    def test_records_only_v_m_by_default(self):
        from brainpy_state import voltmeter
        vm = voltmeter()
        self.assertEqual(vm.record_from, ('V_m',))

    def test_is_a_multimeter(self):
        from brainpy_state import voltmeter, multimeter
        from brainpy_state._nest_base._base import NESTDevice
        vm = voltmeter()
        self.assertIsInstance(vm, multimeter)
        self.assertIsInstance(vm, NESTDevice)

    def test_interval_and_timing_passthrough(self):
        from brainpy_state import voltmeter
        vm = voltmeter(interval=0.5 * u.ms, start=2.0 * u.ms)
        self.assertTrue(u.math.allclose(vm.interval, 0.5 * u.ms))
        self.assertTrue(u.math.allclose(vm.start, 2.0 * u.ms))


if __name__ == '__main__':
    unittest.main()
