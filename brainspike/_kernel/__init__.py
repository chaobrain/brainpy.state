# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K0 kernel (private): model contracts and runtime of the redesigned architecture.

Nothing here is exported through ``brainspike``; internal code imports
``brainspike._kernel`` directly. See ``docs/specs/2026-10-21-k0-kernel-spec.md``.
"""
from brainspike._kernel.context import StepContext
from brainspike._kernel.integrators import RK, Exact, ExpEuler, Integrator, integrate
from brainspike._kernel.model import (
    Capabilities,
    Device,
    NeuronModel,
    conductance_current,
    latch_inputs,
    latch_name,
    latched_ports,
)
from brainspike._kernel.params import (
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
from brainspike._kernel.ports import (
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
    is_latched,
    is_receptor,
)
from brainspike._kernel.propagators import (
    alpha_propagator_p31_p32,
    exp_decay,
    iaf_psc_alpha_propagators,
    phi1,
    phi2,
    propagator_exp,
)
from brainspike._kernel.spiking import Block, Clamp, Refractory, SpikeCarry, SpikeRule, count_ste
from brainspike._kernel.units import (
    PRECISIONS,
    UnitMismatchError,
    as_unit,
    check_unit,
    from_mantissa,
    grid_steps,
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
    'expected_weight_unit', 'is_latched', 'is_receptor',
    # integrators
    'Exact', 'ExpEuler', 'Integrator', 'RK', 'integrate',
    # model
    'Capabilities', 'Device', 'NeuronModel', 'conductance_current', 'latch_inputs',
    'latch_name', 'latched_ports',
    # propagators
    'alpha_propagator_p31_p32', 'exp_decay', 'iaf_psc_alpha_propagators', 'phi1', 'phi2',
    'propagator_exp',
    # spiking
    'Block', 'Clamp', 'Refractory', 'SpikeCarry', 'SpikeRule', 'count_ste',
    # units
    'PRECISIONS', 'UnitMismatchError', 'as_unit', 'check_unit', 'from_mantissa',
    'grid_steps', 'index_dtype', 'resolve_dtype', 'same_dim', 'to_mantissa', 'unit_of',
]
