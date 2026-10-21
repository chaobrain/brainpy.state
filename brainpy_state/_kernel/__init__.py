# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K0 kernel (private): model contracts and runtime of the redesigned architecture.

Nothing here is exported through ``brainpy.state``; internal code imports
``brainpy_state._kernel`` directly. See ``docs/specs/2026-10-21-k0-kernel-spec.md``.
"""
from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.params import (
    ConstParamState,
    Distribution,
    Init,
    LogNormal,
    Normal,
    Param,
    Params,
    Record,
    States,
    Uniform,
    name_key,
    read_states,
    write_states,
)
from brainpy_state._kernel.ports import (
    Conductance,
    Current,
    Event,
    Graded,
    PortError,
    PortKind,
    Ports,
    TimedEvent,
    TimedEventPayload,
    check_capacity,
    check_connection,
    check_kind_compatible,
    check_single_owner,
    check_weight_sign,
    check_weight_unit,
    expected_weight_unit,
)
from brainpy_state._kernel.units import (
    PRECISIONS,
    UnitMismatchError,
    as_unit,
    check_unit,
    from_mantissa,
    index_dtype,
    resolve_dtype,
    same_dim,
    to_mantissa,
    unit_of,
)

__all__ = [
    # context
    'StepContext',
    # params / states
    'ConstParamState', 'Distribution', 'Init', 'LogNormal', 'Normal', 'Param', 'Params',
    'Record', 'States', 'Uniform', 'name_key', 'read_states', 'write_states',
    # ports
    'Conductance', 'Current', 'Event', 'Graded', 'PortError', 'PortKind', 'Ports',
    'TimedEvent', 'TimedEventPayload', 'check_capacity', 'check_connection',
    'check_kind_compatible', 'check_single_owner', 'check_weight_sign', 'check_weight_unit',
    'expected_weight_unit',
    # units
    'PRECISIONS', 'UnitMismatchError', 'as_unit', 'check_unit', 'from_mantissa',
    'index_dtype', 'resolve_dtype', 'same_dim', 'to_mantissa', 'unit_of',
]
