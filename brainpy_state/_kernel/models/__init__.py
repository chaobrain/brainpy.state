# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K0 models on the kernel contracts (private; spec §2, task T3).

Neurons: :class:`LIF` (BrainPy numerics), :class:`iaf_psc_alpha` (exact
propagators), :class:`iaf_cond_exp` and :class:`aeif_cond_alpha` (fixed-substep
RK4). Devices: :class:`poisson_generator`, :class:`spike_generator`,
:class:`dc_generator`. Receptors: :class:`NMDA` (K0 T8). See ``docs/specs/2026-10-21-k0-t3-models.md``.
Precise timing (T7, ``timed_event`` ports): :class:`iaf_psc_exp_ps`, :class:`poisson_generator_ps`,
:class:`spike_generator_ps` (``docs/specs/2026-10-22-k0-t7-precise-timing.md``).
"""
from brainpy_state._kernel.models._common import SpikingNeuron, replace_rule
from brainpy_state._kernel.models.aeif_cond_alpha import AEIF_COND_ALPHA_SUBSTEPS, aeif_cond_alpha
from brainpy_state._kernel.models.devices import dc_generator, poisson_generator, spike_generator
from brainpy_state._kernel.models.iaf_cond_exp import IAF_COND_EXP_SUBSTEPS, iaf_cond_exp
from brainpy_state._kernel.models.iaf_psc_alpha import iaf_psc_alpha
from brainpy_state._kernel.models.lif import LIF
from brainpy_state._kernel.models.nmda import NMDA, nmda_kinetics
from brainpy_state._kernel.models.precise import iaf_psc_exp_ps, poisson_generator_ps, spike_generator_ps

__all__ = [
    'LIF', 'iaf_psc_alpha', 'iaf_cond_exp', 'aeif_cond_alpha',
    'poisson_generator', 'spike_generator', 'dc_generator',
    'iaf_psc_exp_ps', 'poisson_generator_ps', 'spike_generator_ps',
    'IAF_COND_EXP_SUBSTEPS', 'AEIF_COND_ALPHA_SUBSTEPS',
    'SpikingNeuron', 'replace_rule',
    'NMDA', 'nmda_kinetics',
]
