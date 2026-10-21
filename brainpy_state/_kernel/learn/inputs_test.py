# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of ``ExternalInput`` (data relay of learning networks)."""
from __future__ import annotations

import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainpy_state._kernel import StepContext
from brainpy_state._kernel.learn import ExternalInput
from brainpy_state._kernel.models import LIF
from brainpy_state._kernel.network import Network

DT = 0.1 * u.ms


def test_relay_is_stateless_and_exact():
    relay = ExternalInput(3)
    ctx = StepContext.create(0.0 * u.ms, DT)
    assert len(relay.dyn_states) == 0 and len(relay.param_states) == 0
    out = relay.update(ctx, {'x': jnp.array([0.0, 0.25, 3.0])})
    np.testing.assert_array_equal(np.asarray(out['spike']), [0.0, 0.25, 3.0])
    np.testing.assert_array_equal(np.asarray(relay.update(ctx)['spike']), [0.0, 0.0, 0.0])


def test_relay_in_a_network_uses_the_float_path():
    net = Network(dt=DT)
    x = net.add(ExternalInput, 2, name='x')
    h = net.add(LIF, 2, name='h')
    p = net.connect(x, h, port='delta', weight=0.4 * u.mV, delay=0.0 * u.ms)
    plan = net.compile()
    assert plan.projection(p).path == 'float'
    prog = plan.runner().program
    # fractional and multi-count payloads are transmitted exactly (0.4 * 2.5 = 1 mV -> spike)
    _, cur = prog.step(prog.read_params(), prog.init_state(), 0, jnp.uint32(0),
                       ext={'x': {'x': jnp.array([2.5, 0.0])}})
    np.testing.assert_array_equal(np.asarray(cur[1]['spike']), [1.0, 1.0])
