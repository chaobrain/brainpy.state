# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Learners over a ``Program`` (decision D-c): surrogate-gradient BPTT and braintrace."""
from brainpy_state._kernel.learn.bptt import (
    BPTT,
    membrane_mse,
    rate_mse,
    spike_count_mse,
    trainable_states,
)
from brainpy_state._kernel.learn.etrace import (
    EtraceError,
    EtraceNetwork,
    EtraceReport,
    Online,
    check_etrace,
    check_plan,
)
from brainpy_state._kernel.learn.inputs import ExternalInput

__all__ = [
    'BPTT',
    'trainable_states',
    'spike_count_mse',
    'rate_mse',
    'membrane_mse',
    'EtraceError',
    'EtraceReport',
    'EtraceNetwork',
    'Online',
    'check_plan',
    'check_etrace',
    'ExternalInput',
]
