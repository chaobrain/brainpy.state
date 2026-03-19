# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import unittest

import brainstate
import saiunit as u
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

from brainpy.state import volume_transmitter


def _as_pairs(spikes):
    return [(float(s.spike_time), float(s.multiplicity)) for s in spikes]


def _to_float_1d(x):
    if isinstance(x, u.Quantity):
        x = u.get_mantissa(x)
    dftype = brainstate.environ.dftype()
    return np.asarray(u.math.asarray(x), dtype=dftype).reshape(-1)


def _to_int_1d(x):
    arr = _to_float_1d(x)
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=1e-12, rtol=1e-12):
        raise ValueError('Expected integer-like values.')
    return rounded.astype(np.int64)


def _schedule_reference(pending, spikes, multiplicities, stamp_steps, stamp_now):
    if spikes is None:
        return

    spike_arr = _to_float_1d(spikes)
    if spike_arr.size == 0:
        return

    n_items = spike_arr.size
    if multiplicities is None:
        rounded = np.rint(spike_arr)
        is_integer_like = np.allclose(spike_arr, rounded, atol=1e-12, rtol=1e-12)
        if is_integer_like:
            counts = np.maximum(rounded.astype(np.int64), 0)
        else:
            counts = (spike_arr > 0.0).astype(np.int64)
    else:
        mult = _to_int_1d(multiplicities)
        if mult.size != n_items:
            raise ValueError('multiplicities size mismatch')
        if np.any(mult < 0):
            raise ValueError('negative multiplicity')
        counts = np.where(spike_arr > 0.0, mult, 0)

    if stamp_steps is None:
        ditype = brainstate.environ.ditype()
        stamps = np.full((n_items,), stamp_now, dtype=ditype)
    else:
        stamps = _to_int_1d(stamp_steps)
        if stamps.size != n_items:
            raise ValueError('stamp_steps size mismatch')
        if np.any(stamps < stamp_now):
            raise ValueError('past stamp in reference schedule')

    for i in range(n_items):
        c = int(counts[i])
        if c <= 0:
            continue
        s = int(stamps[i])
        pending[s] = float(pending.get(s, 0.0) + c)


def _run_reference_trace(n_steps, dt_ms, deliver_interval, min_delay_ms, per_step_payload):
    min_delay_steps = int(np.rint(min_delay_ms / dt_ms))
    period = int(deliver_interval) * int(min_delay_steps)
    if period < 1:
        raise ValueError('invalid period')

    pending = {}
    spike_hist = [(0.0, 0.0)]
    out = []

    for step in range(n_steps):
        stamp = step + 1
        payload = per_step_payload.get(step, {})
        _schedule_reference(
            pending=pending,
            spikes=payload.get('spikes', None),
            multiplicities=payload.get('multiplicities', None),
            stamp_steps=payload.get('stamp_steps', None),
            stamp_now=stamp,
        )

        multiplicity = float(pending.pop(stamp, 0.0))
        if multiplicity > 0.0:
            spike_hist.append((float(stamp) * dt_ms, multiplicity))

        triggered = (stamp % period) == 0
        t_trig = float(stamp) * dt_ms if triggered else None
        delivered = tuple(spike_hist) if triggered else tuple()

        if triggered:
            spike_hist = [(t_trig, 0.0)]

        out.append(
            {
                'triggered': bool(triggered),
                't_trig': t_trig,
                'delivered_spikes': delivered,
                'spike_history': tuple(spike_hist),
            }
        )

    return out


class TestVolumeTransmitter(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _run_model_trace(self, n_steps, vt, per_step_payload):
        out = []
        with brainstate.environ.context(dt=self.dt):
            vt.init_state()
            for step in range(n_steps):
                payload = per_step_payload.get(step, {})
                with brainstate.environ.context(t=step * self.dt):
                    y = vt.update(**payload)
                out.append(
                    {
                        'triggered': bool(y['triggered']),
                        't_trig': y['t_trig'],
                        'delivered_spikes': tuple(_as_pairs(y['delivered_spikes'])),
                        'spike_history': tuple(_as_pairs(y['spike_history'])),
                    }
                )
        return out

    def test_nest_default_parameters_and_initial_pseudo_spike(self):
        vt = volume_transmitter()
        self.assertEqual(vt.deliver_interval, 1)
        self.assertTrue(u.math.allclose(vt.min_delay, 1.0 * u.ms))
        self.assertEqual(vt.get_local_device_id(), 0)
        self.assertEqual(vt.n_deliveries, 0)
        self.assertEqual(vt.last_delivery_time, 0.0)
        self.assertEqual(vt.last_delivery_spikes, ())
        self.assertEqual(_as_pairs(vt.deliver_spikes()), [(0.0, 0.0)])

    def test_receptor_and_local_device_id_helpers_match_nest_intent(self):
        vt = volume_transmitter()
        self.assertEqual(vt.handles_test_event(0), 0)
        with self.assertRaises(ValueError):
            vt.handles_test_event(1)

        vt.set_local_device_id(7)
        self.assertEqual(vt.get_local_device_id(), 7)
        self.assertEqual(vt.local_device_id, 7)

    def test_step_dynamics_match_nest_cpp_update_order(self):
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()
        per_step_payload = {
            0: {
                'spikes': np.array([1.0, 1.0], dtype=dftype),
                'multiplicities': np.array([1, 2], dtype=ditype),
            },
            1: {
                'spikes': np.array([1.0], dtype=dftype),
                'multiplicities': np.array([2], dtype=ditype),
                'stamp_steps': np.array([4], dtype=ditype),
            },
            3: {
                'spikes': np.array([1.0], dtype=dftype),
                'multiplicities': np.array([1], dtype=ditype),
            },
            5: {
                'spikes': np.array([1.0], dtype=dftype),
                'multiplicities': np.array([5], dtype=ditype),
            },
        }

        n_steps = 7
        vt = volume_transmitter(deliver_interval=2, min_delay=0.3 * u.ms)
        model_out = self._run_model_trace(n_steps=n_steps, vt=vt, per_step_payload=per_step_payload)
        ref_out = _run_reference_trace(
            n_steps=n_steps,
            dt_ms=self.dt_ms,
            deliver_interval=2,
            min_delay_ms=0.3,
            per_step_payload=per_step_payload,
        )

        for step in range(n_steps):
            self.assertEqual(model_out[step]['triggered'], ref_out[step]['triggered'])
            self.assertEqual(model_out[step]['t_trig'], ref_out[step]['t_trig'])
            self.assertEqual(model_out[step]['delivered_spikes'], ref_out[step]['delivered_spikes'])
            self.assertEqual(model_out[step]['spike_history'], ref_out[step]['spike_history'])

        # Core ordering check from NEST C++: spike at trigger step is included
        # in delivered history before the history is reset to pseudo spike.
        self.assertEqual(
            model_out[5]['delivered_spikes'],
            ((0.0, 0.0), (0.1, 3.0), (0.4, 3.0), (0.6000000000000001, 5.0)),
        )
        self.assertEqual(model_out[5]['spike_history'], ((0.6000000000000001, 0.0),))

    def test_input_validation_and_multiplicity_aggregation(self):
        vt = volume_transmitter(deliver_interval=1, min_delay=0.2 * u.ms)
        with brainstate.environ.context(dt=self.dt):
            vt.init_state()

            with brainstate.environ.context(t=0.0 * u.ms):
                dftype = brainstate.environ.dftype()
                ditype = brainstate.environ.ditype()
                vt.update(
                    spikes=np.array([1.0, 1.0, 1.0], dtype=dftype),
                    multiplicities=np.array([2, 3, 4], dtype=ditype),
                    stamp_steps=np.array([2, 2, 1], dtype=ditype),
                )

            with brainstate.environ.context(t=0.1 * u.ms):
                out = vt.update()

            self.assertTrue(out['triggered'])
            self.assertEqual(_as_pairs(out['delivered_spikes']), [(0.0, 0.0), (0.1, 4.0), (0.2, 5.0)])

            with brainstate.environ.context(t=0.2 * u.ms):
                with self.assertRaises(ValueError):
                    vt.update(
                        spikes=np.array([1.0], dtype=dftype),
                        stamp_steps=np.array([2], dtype=ditype),
                    )

                with self.assertRaises(ValueError):
                    vt.update(
                        spikes=np.array([1.0], dtype=dftype),
                        multiplicities=np.array([-1], dtype=ditype),
                    )

        with self.assertRaises(ValueError):
            volume_transmitter(deliver_interval=0)


if __name__ == '__main__':
    unittest.main()
