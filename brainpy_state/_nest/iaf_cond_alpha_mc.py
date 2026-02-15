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

from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron

__all__ = [
    'iaf_cond_alpha_mc',
]


class iaf_cond_alpha_mc(Neuron):
    r"""NEST-compatible ``iaf_cond_alpha_mc`` neuron model.

    Short description
    -----------------

    Three-compartment conductance-based leaky integrate-and-fire neuron with
    alpha-shaped synapses, following NEST ``models/iaf_cond_alpha_mc.{h,cpp}``.

    Description
    -----------

    ``iaf_cond_alpha_mc`` is the multicompartment extension of
    ``iaf_cond_alpha`` with compartments

    - soma (``s``),
    - proximal dendrite (``p``),
    - distal dendrite (``d``).

    Compartments are coupled by passive conductances ``g_sp`` and ``g_pd``.
    Each compartment has one excitatory and one inhibitory alpha synapse.
    Spike threshold and reset are applied at the soma only.

    This implementation mirrors NEST source behavior, including:

    - adaptive RKF45 integration (state-wide, persistent internal step),
    - one-step delayed current buffering per compartment,
    - receptor-specific spike and current routing,
    - NEST update ordering in ``update()``.

    Membrane and synaptic dynamics
    ..............................

    For compartment :math:`c \in \{s,p,d\}`:

    .. math::

       C_{m,c}\frac{dV_c}{dt} =
       -g_{L,c}(V_c - E_{L,c})
       -g_{\mathrm{ex},c}(V_c - E_{\mathrm{ex},c})
       -g_{\mathrm{in},c}(V_c - E_{\mathrm{in},c})
       -I_{\mathrm{conn},c}
       + I_{\mathrm{stim},c} + I_{e,c}.

    Coupling currents are

    .. math::

       I_{\mathrm{conn},s} = g_{sp}(V_s - V_p),

       I_{\mathrm{conn},p} = g_{sp}(V_p - V_s) + g_{pd}(V_p - V_d),

       I_{\mathrm{conn},d} = g_{pd}(V_d - V_p).

    Alpha-synapse states per compartment follow

    .. math::

       \frac{d\,dg_{\mathrm{ex},c}}{dt} = -\frac{dg_{\mathrm{ex},c}}{\tau_{\mathrm{syn,ex},c}},
       \qquad
       \frac{dg_{\mathrm{ex},c}}{dt} = dg_{\mathrm{ex},c} - \frac{g_{\mathrm{ex},c}}{\tau_{\mathrm{syn,ex},c}},

    .. math::

       \frac{d\,dg_{\mathrm{in},c}}{dt} = -\frac{dg_{\mathrm{in},c}}{\tau_{\mathrm{syn,in},c}},
       \qquad
       \frac{dg_{\mathrm{in},c}}{dt} = dg_{\mathrm{in},c} - \frac{g_{\mathrm{in},c}}{\tau_{\mathrm{syn,in},c}}.

    Incoming spike weight :math:`w` on a receptor port adds to ``dg`` as

    .. math::

       dg \leftarrow dg + \frac{e}{\tau_{\mathrm{syn}}} w.

    Spike and refractory semantics
    ..............................

    - Spike is emitted if somatic membrane potential satisfies
      :math:`V_s \ge V_{th}` after integration.
    - On spike: somatic voltage is reset to ``V_reset`` and refractory counter
      is set to ``ceil(t_ref / dt)``.
    - During refractory period, the ODE uses ``V_reset`` as somatic effective
      voltage and keeps all membrane derivatives at zero, matching NEST C++
      implementation.

    NEST receptor types
    ...................

    Spike receptors (must have non-negative weights):

    - ``soma_exc`` = 1
    - ``soma_inh`` = 2
    - ``proximal_exc`` = 3
    - ``proximal_inh`` = 4
    - ``distal_exc`` = 5
    - ``distal_inh`` = 6

    Current receptors:

    - ``soma_curr`` = 7
    - ``proximal_curr`` = 8
    - ``distal_curr`` = 9

    Update order (NEST semantics)
    .............................

    Per simulation step:

    1. Integrate ODEs on :math:`(t, t+dt]` using RKF45 with adaptive substeps.
    2. Apply incoming spike events to ``dg_ex`` / ``dg_in`` per receptor type.
    3. Apply refractory countdown / threshold test / reset / spike emission.
    4. Store incoming currents into delayed buffer ``I_stim`` for next step.

    Parameters
    ----------

    ===================== ============================= =====================================================
    **Parameter**         **Default**                   **Description**
    ===================== ============================= =====================================================
    ``in_size``           (required)                    Population shape
    ``V_th``              -55 mV                        Somatic threshold
    ``V_reset``           -60 mV                        Somatic reset potential
    ``t_ref``             2 ms                          Absolute refractory duration
    ``g_sp``              2.5 nS                        Soma-proximal coupling conductance
    ``g_pd``              1.0 nS                        Proximal-distal coupling conductance
    ``soma``              NEST defaults                 Per-compartment parameter dict for soma
    ``proximal``          NEST defaults                 Per-compartment parameter dict for proximal dendrite
    ``distal``            NEST defaults                 Per-compartment parameter dict for distal dendrite
    ``V_initializer``     ``None``                      Initial membrane potentials; default uses each ``E_L``
    ``spk_fun``           ReluGrad()                    Surrogate spike function
    ``spk_reset``         ``'hard'``                    Reset mode; hard reset matches NEST behavior
    ``ref_var``           ``False``                     If True, expose boolean refractory indicator
    ===================== ============================= =====================================================

    Per-compartment dictionaries support keys:
    ``g_L``, ``C_m``, ``E_ex``, ``E_in``, ``E_L``, ``tau_syn_ex``,
    ``tau_syn_in``, ``I_e``.

    State variables
    ---------------

    - ``V``: compartment membrane potentials ``[..., 3]`` in order ``(s, p, d)``.
    - ``dg_ex`` / ``dg_in``: alpha auxiliary conductance states ``[..., 3]``.
    - ``g_ex`` / ``g_in``: excitatory/inhibitory conductances ``[..., 3]``.
    - ``I_stim``: one-step delayed current buffer per compartment ``[..., 3]``.
    - ``refractory_step_count``: somatic refractory countdown.
    - ``integration_step``: persistent RKF45 internal step size.
    - ``last_spike_time``: last emitted spike time.

    Notes
    -----

    - NEST marks ``iaf_cond_alpha_mc`` as deprecated in favor of ``cm_default``.
    - As in NEST, spike weights are receptor-routed and must be non-negative.
    - The model is implemented for source-level behavioral parity with
      ``models/iaf_cond_alpha_mc.cpp``.

    References
    ----------
    .. [1] Meffin H, Burkitt AN, Grayden DB (2004). An analytical model for
           the large, fluctuating synaptic conductance state typical of
           neocortical neurons in vivo. Journal of Computational Neuroscience,
           16:159-175. DOI: https://doi.org/10.1023/B:JCNS.0000014108.03012.81
    .. [2] Bernander O, Douglas RJ, Martin KAC, Koch C (1991). Synaptic
           background activity influences spatiotemporal integration in single
           pyramidal cells. PNAS, 88(24):11569-11573.
           DOI: https://doi.org/10.1073/pnas.88.24.11569
    """

    __module__ = 'brainpy.state'

    SOMA = 0
    PROX = 1
    DIST = 2
    NCOMP = 3

    V_M = 0
    DG_EXC = 1
    G_EXC = 2
    DG_INH = 3
    G_INH = 4
    NSTATE_COMP = 5

    _ATOL = 1e-3
    _MIN_H = 1e-8  # ms
    _MAX_ITERS = 10000

    SPIKE_RECEPTOR_TYPES = {
        'soma_exc': 1,
        'soma_inh': 2,
        'proximal_exc': 3,
        'proximal_inh': 4,
        'distal_exc': 5,
        'distal_inh': 6,
    }
    CURRENT_RECEPTOR_TYPES = {
        'soma_curr': 7,
        'proximal_curr': 8,
        'distal_curr': 9,
    }
    RECEPTOR_TYPES = {
        **SPIKE_RECEPTOR_TYPES,
        **CURRENT_RECEPTOR_TYPES,
    }
    RECORDABLES = (
        'V_m.s', 'g_ex.s', 'g_in.s',
        'V_m.p', 'g_ex.p', 'g_in.p',
        'V_m.d', 'g_ex.d', 'g_in.d',
        't_ref_remaining',
    )

    def __init__(
        self,
        in_size: Size,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -60. * u.mV,
        t_ref: ArrayLike = 2. * u.ms,
        g_sp: ArrayLike = 2.5 * u.nS,
        g_pd: ArrayLike = 1.0 * u.nS,
        soma: dict | None = None,
        proximal: dict | None = None,
        distal: dict | None = None,
        V_initializer: Callable | dict | ArrayLike | None = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.g_sp = braintools.init.param(g_sp, self.varshape)
        self.g_pd = braintools.init.param(g_pd, self.varshape)

        self._compartments = self._build_compartment_parameters(soma, proximal, distal)
        self.soma = self._compartments['soma']
        self.proximal = self._compartments['proximal']
        self.distal = self._compartments['distal']

        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @classmethod
    def _default_compartment_parameters(cls):
        return {
            'soma': {
                'g_L': 10.0 * u.nS,
                'C_m': 150.0 * u.pF,
                'E_ex': 0.0 * u.mV,
                'E_in': -85.0 * u.mV,
                'E_L': -70.0 * u.mV,
                'tau_syn_ex': 0.5 * u.ms,
                'tau_syn_in': 2.0 * u.ms,
                'I_e': 0.0 * u.pA,
            },
            'proximal': {
                'g_L': 5.0 * u.nS,
                'C_m': 75.0 * u.pF,
                'E_ex': 0.0 * u.mV,
                'E_in': -85.0 * u.mV,
                'E_L': -70.0 * u.mV,
                'tau_syn_ex': 0.5 * u.ms,
                'tau_syn_in': 2.0 * u.ms,
                'I_e': 0.0 * u.pA,
            },
            'distal': {
                'g_L': 10.0 * u.nS,
                'C_m': 150.0 * u.pF,
                'E_ex': 0.0 * u.mV,
                'E_in': -85.0 * u.mV,
                'E_L': -70.0 * u.mV,
                'tau_syn_ex': 0.5 * u.ms,
                'tau_syn_in': 2.0 * u.ms,
                'I_e': 0.0 * u.pA,
            },
        }

    def _build_compartment_parameters(self, soma, proximal, distal):
        defaults = self._default_compartment_parameters()
        overrides = {
            'soma': soma,
            'proximal': proximal,
            'distal': distal,
        }

        result = {}
        for comp in ('soma', 'proximal', 'distal'):
            cfg = dict(defaults[comp])
            override = overrides[comp]
            if override is not None:
                if not isinstance(override, dict):
                    raise TypeError(f'`{comp}` must be a dict when provided.')
                unknown = set(override) - set(cfg)
                if unknown:
                    raise ValueError(f'Unknown keys in `{comp}`: {sorted(unknown)}')
                cfg.update(override)
            result[comp] = {
                key: braintools.init.param(value, self.varshape)
                for key, value in cfg.items()
            }

        return result

    @property
    def receptor_types(self):
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    @staticmethod
    def _state_index(comp: int, elem: int):
        return comp * iaf_cond_alpha_mc.NSTATE_COMP + elem

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        if isinstance(receptor, str):
            receptor = receptor.strip()
            if receptor in cls.SPIKE_RECEPTOR_TYPES:
                return cls.SPIKE_RECEPTOR_TYPES[receptor]
            if receptor.isdigit():
                receptor = int(receptor)
            else:
                raise ValueError(f'Unknown spike receptor label: {receptor}')

        receptor = int(receptor)
        if receptor < 1 or receptor > 6:
            raise ValueError(f'Spike receptor type must be in [1, 6], got {receptor}.')
        return receptor

    @classmethod
    def _normalize_current_compartment_index(cls, receptor_or_label):
        if isinstance(receptor_or_label, str):
            key = receptor_or_label.strip()
            if key in ('soma', 'proximal', 'distal'):
                return {'soma': cls.SOMA, 'proximal': cls.PROX, 'distal': cls.DIST}[key]
            if key in cls.CURRENT_RECEPTOR_TYPES:
                return cls.CURRENT_RECEPTOR_TYPES[key] - 7
            if key.isdigit():
                receptor_or_label = int(key)
            else:
                raise ValueError(f'Unknown current receptor label: {receptor_or_label}')

        receptor_or_label = int(receptor_or_label)
        if 0 <= receptor_or_label <= 2:
            return receptor_or_label
        if 7 <= receptor_or_label <= 9:
            return receptor_or_label - 7
        raise ValueError(
            f'Current receptor must be in [7, 9] (or compartment index [0, 2]), '
            f'got {receptor_or_label}.'
        )

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time cannot be negative.')

        for comp in ('soma', 'proximal', 'distal'):
            cm = self._to_numpy(self._compartments[comp]['C_m'], u.pF)
            tau_ex = self._to_numpy(self._compartments[comp]['tau_syn_ex'], u.ms)
            tau_in = self._to_numpy(self._compartments[comp]['tau_syn_in'], u.ms)
            if np.any(cm <= 0.0):
                raise ValueError(f'Capacitance ({comp}) must be strictly positive.')
            if np.any(tau_ex <= 0.0) or np.any(tau_in <= 0.0):
                raise ValueError(f'All time constants ({comp}) must be strictly positive.')

    def _safe_dt(self):
        try:
            return brainstate.environ.get_dt()
        except KeyError:
            return 0.1 * u.ms

    def _initial_membrane_potential(self, batch_size):
        state_shape = self._state_shape(batch_size)

        if self.V_initializer is None:
            init_cfg = {
                'soma': self.soma['E_L'],
                'proximal': self.proximal['E_L'],
                'distal': self.distal['E_L'],
            }
        elif isinstance(self.V_initializer, dict):
            init_cfg = {
                'soma': self.soma['E_L'],
                'proximal': self.proximal['E_L'],
                'distal': self.distal['E_L'],
            }
            unknown = set(self.V_initializer) - {'soma', 'proximal', 'distal'}
            if unknown:
                raise ValueError(f'Unknown keys in `V_initializer`: {sorted(unknown)}')
            init_cfg.update(self.V_initializer)
        else:
            init_cfg = {
                'soma': self.V_initializer,
                'proximal': self.V_initializer,
                'distal': self.V_initializer,
            }

        v_s = braintools.init.param(init_cfg['soma'], self.varshape, batch_size)
        v_p = braintools.init.param(init_cfg['proximal'], self.varshape, batch_size)
        v_d = braintools.init.param(init_cfg['distal'], self.varshape, batch_size)

        v_s = np.broadcast_to(np.asarray(u.math.asarray(v_s / u.mV), dtype=np.float64), state_shape)
        v_p = np.broadcast_to(np.asarray(u.math.asarray(v_p / u.mV), dtype=np.float64), state_shape)
        v_d = np.broadcast_to(np.asarray(u.math.asarray(v_d / u.mV), dtype=np.float64), state_shape)

        v_stack = np.stack([
            v_s,
            v_p,
            v_d,
        ], axis=-1) * u.mV
        return v_stack

    def _state_shape(self, batch_size):
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        return np.asarray(u.math.asarray(ref_steps), dtype=np.int32).shape

    def _stack_compartment_parameter(self, key: str, unit, state_shape):
        vals = []
        for comp in ('soma', 'proximal', 'distal'):
            vals.append(
                self._broadcast_to_state(self._to_numpy(self._compartments[comp][key], unit), state_shape)
            )
        return np.stack(vals, axis=-1)

    def init_state(self, batch_size: int = None, **kwargs):
        V = self._initial_membrane_potential(batch_size)
        state_shape = self._state_shape(batch_size)
        zeros_comp = np.zeros(state_shape + (self.NCOMP,), dtype=np.float64)

        self.V = brainstate.HiddenState(V)
        self.dg_ex = brainstate.ShortTermState(zeros_comp.copy())
        self.g_ex = brainstate.HiddenState(zeros_comp.copy() * u.nS)
        self.dg_in = brainstate.ShortTermState(zeros_comp.copy())
        self.g_in = brainstate.HiddenState(zeros_comp.copy() * u.nS)

        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count = brainstate.ShortTermState(u.math.asarray(ref_steps, dtype=jnp.int32))

        dt = self._safe_dt()
        self.integration_step = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(dt), self.varshape, batch_size)
        )
        self.I_stim = brainstate.ShortTermState(zeros_comp.copy() * u.pA)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory = brainstate.ShortTermState(refractory)

    def reset_state(self, batch_size: int = None, **kwargs):
        V = self._initial_membrane_potential(batch_size)
        state_shape = self._state_shape(batch_size)
        zeros_comp = np.zeros(state_shape + (self.NCOMP,), dtype=np.float64)

        self.V.value = V
        self.dg_ex.value = zeros_comp.copy()
        self.g_ex.value = zeros_comp.copy() * u.nS
        self.dg_in.value = zeros_comp.copy()
        self.g_in.value = zeros_comp.copy() * u.nS

        self.last_spike_time.value = braintools.init.param(
            braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size
        )

        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        self.refractory_step_count.value = u.math.asarray(ref_steps, dtype=jnp.int32)

        dt = self._safe_dt()
        self.integration_step.value = braintools.init.param(
            braintools.init.Constant(dt), self.varshape, batch_size
        )
        self.I_stim.value = zeros_comp.copy() * u.pA

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape, batch_size)
            self.refractory.value = refractory

    def get_spike(self, V: ArrayLike = None):
        if V is None:
            V = self.V.value[..., self.SOMA]
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        dt = brainstate.environ.get_dt()
        return u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events, state_shape):
        out = np.zeros(state_shape + (6,), dtype=np.float64)
        if spike_events is None:
            return out

        for ev in spike_events:
            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 1))
                weight = ev.get('weight', 0.0 * u.nS)
            else:
                receptor, weight = ev

            receptor_id = self._normalize_spike_receptor(receptor)
            weight_np = np.asarray(u.math.asarray(weight / u.nS), dtype=np.float64)
            if np.any(weight_np < 0.0):
                raise ValueError('All spike weights must be non-negative for `iaf_cond_alpha_mc`.')
            out[..., receptor_id - 1] += np.broadcast_to(weight_np, state_shape)

        return out

    def _parse_current_events(self, current_events, state_shape):
        out = np.zeros(state_shape + (self.NCOMP,), dtype=np.float64)
        if current_events is None:
            return out

        for ev in current_events:
            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 'soma_curr'))
                current = ev.get('current', ev.get('weight', 0.0 * u.pA))
            else:
                receptor, current = ev

            comp_idx = self._normalize_current_compartment_index(receptor)
            current_np = np.asarray(u.math.asarray(current / u.pA), dtype=np.float64)
            out[..., comp_idx] += np.broadcast_to(current_np, state_shape)

        return out

    def _parse_registered_spike_inputs(self, state_shape):
        out = np.zeros(state_shape + (6,), dtype=np.float64)
        if self.delta_inputs is None:
            return out

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)
            receptor = self.SPIKE_RECEPTOR_TYPES['soma_exc'] if label is None else self._normalize_spike_receptor(label)

            val_np = np.asarray(u.math.asarray(val / u.nS), dtype=np.float64)
            if np.any(val_np < 0.0):
                raise ValueError('All spike weights must be non-negative for `iaf_cond_alpha_mc`.')
            out[..., receptor - 1] += np.broadcast_to(val_np, state_shape)

        return out

    def _parse_registered_current_inputs(self, state_shape):
        out = np.zeros(state_shape + (self.NCOMP,), dtype=np.float64)
        if self.current_inputs is None:
            return out

        for key in tuple(self.current_inputs.keys()):
            val = self.current_inputs[key]
            if callable(val):
                val = val()
            else:
                self.current_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)
            comp_idx = self.SOMA if label is None else self._normalize_current_compartment_index(label)

            val_np = np.asarray(u.math.asarray(val / u.pA), dtype=np.float64)
            out[..., comp_idx] += np.broadcast_to(val_np, state_shape)

        return out

    def _parse_current_argument(self, x, state_shape):
        out = np.zeros(state_shape + (self.NCOMP,), dtype=np.float64)

        if isinstance(x, dict):
            for key, val in x.items():
                comp_idx = self._normalize_current_compartment_index(key)
                val_np = np.asarray(u.math.asarray(val / u.pA), dtype=np.float64)
                out[..., comp_idx] += np.broadcast_to(val_np, state_shape)
            return out

        x_np = np.asarray(u.math.asarray(x / u.pA), dtype=np.float64)
        if x_np.shape == state_shape + (self.NCOMP,):
            return x_np
        if x_np.shape == (self.NCOMP,):
            return np.broadcast_to(x_np, state_shape + (self.NCOMP,))

        if x_np.ndim == 0 or x_np.shape == state_shape:
            out[..., self.SOMA] = np.broadcast_to(x_np, state_shape)
            return out

        try:
            return np.broadcast_to(x_np, state_shape + (self.NCOMP,))
        except ValueError as exc:
            raise ValueError(
                'Current input `x` must be scalar/soma-shaped, have trailing 3 '
                'compartments, or be a compartment-labeled dict.'
            ) from exc

    @classmethod
    def _dynamics_scalar(cls, y, is_refractory, i_stim, p):
        f = np.zeros_like(y)

        for n in range(cls.NCOMP):
            if n == cls.SOMA:
                v_eff = p['V_reset'] if is_refractory else min(y[cls._state_index(cls.SOMA, cls.V_M)], p['V_th'])
                i_conn = p['g_sp'] * (v_eff - y[cls._state_index(cls.PROX, cls.V_M)])
            elif n == cls.PROX:
                v_eff = y[cls._state_index(cls.PROX, cls.V_M)]
                i_conn = (
                    p['g_sp'] * (v_eff - y[cls._state_index(cls.SOMA, cls.V_M)])
                    + p['g_pd'] * (v_eff - y[cls._state_index(cls.DIST, cls.V_M)])
                )
            else:
                v_eff = y[cls._state_index(cls.DIST, cls.V_M)]
                i_conn = p['g_pd'] * (v_eff - y[cls._state_index(cls.PROX, cls.V_M)])

            i_syn_ex = y[cls._state_index(n, cls.G_EXC)] * (v_eff - p['E_ex'][n])
            i_syn_in = y[cls._state_index(n, cls.G_INH)] * (v_eff - p['E_in'][n])
            i_leak = p['g_L'][n] * (v_eff - p['E_L'][n])

            f[cls._state_index(n, cls.V_M)] = 0.0 if is_refractory else (
                -i_leak - i_syn_ex - i_syn_in - i_conn + i_stim[n] + p['I_e'][n]
            ) / p['C_m'][n]

            f[cls._state_index(n, cls.DG_EXC)] = -y[cls._state_index(n, cls.DG_EXC)] / p['tau_syn_ex'][n]
            f[cls._state_index(n, cls.G_EXC)] = (
                y[cls._state_index(n, cls.DG_EXC)] - y[cls._state_index(n, cls.G_EXC)] / p['tau_syn_ex'][n]
            )

            f[cls._state_index(n, cls.DG_INH)] = -y[cls._state_index(n, cls.DG_INH)] / p['tau_syn_in'][n]
            f[cls._state_index(n, cls.G_INH)] = (
                y[cls._state_index(n, cls.DG_INH)] - y[cls._state_index(n, cls.G_INH)] / p['tau_syn_in'][n]
            )

        return f

    def _rkf45_integrate_scalar(self, y0, is_refractory, i_stim, h0, dt, p):
        t = 0.0
        h = max(h0, self._MIN_H)
        y = np.asarray(y0, dtype=np.float64)
        iters = 0

        while t < dt and iters < self._MAX_ITERS:
            iters += 1
            h = max(self._MIN_H, min(h, dt - t))

            k1 = self._dynamics_scalar(y, is_refractory, i_stim, p)
            k2 = self._dynamics_scalar(y + h * (1.0 / 4.0) * k1, is_refractory, i_stim, p)
            k3 = self._dynamics_scalar(
                y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0),
                is_refractory, i_stim, p
            )
            k4 = self._dynamics_scalar(
                y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0),
                is_refractory, i_stim, p
            )
            k5 = self._dynamics_scalar(
                y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0),
                is_refractory, i_stim, p
            )
            k6 = self._dynamics_scalar(
                y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0),
                is_refractory, i_stim, p
            )

            y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0)
            y5 = y + h * (
                16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0
            )
            err = float(np.max(np.abs(y5 - y4)))

            if err <= self._ATOL or h <= self._MIN_H:
                y = y5
                t += h
                fac = 5.0 if err == 0.0 else min(5.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.2))
                h = max(self._MIN_H, h * fac)
            else:
                fac = min(1.0, max(0.2, 0.9 * (self._ATOL / err) ** 0.25))
                h = max(self._MIN_H, h * fac)

        return y, h

    def update(self, x=0. * u.pA, spike_events=None, current_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        dt = float(u.math.asarray(dt_q / u.ms))

        r_raw = np.asarray(u.math.asarray(self.refractory_step_count.value), dtype=np.int32)
        state_shape = r_raw.shape
        comp_shape = state_shape + (self.NCOMP,)

        V = self._broadcast_to_state(np.asarray(self._to_numpy(self.V.value, u.mV), dtype=np.float64), comp_shape)

        dg_ex = self._broadcast_to_state(np.asarray(self.dg_ex.value, dtype=np.float64), comp_shape)
        g_ex = self._broadcast_to_state(np.asarray(self._to_numpy(self.g_ex.value, u.nS), dtype=np.float64), comp_shape)
        dg_in = self._broadcast_to_state(np.asarray(self.dg_in.value, dtype=np.float64), comp_shape)
        g_in = self._broadcast_to_state(np.asarray(self._to_numpy(self.g_in.value, u.nS), dtype=np.float64), comp_shape)

        r = self._broadcast_to_state(r_raw, state_shape)
        i_stim = self._broadcast_to_state(np.asarray(self._to_numpy(self.I_stim.value, u.pA), dtype=np.float64), comp_shape)
        h_int = self._broadcast_to_state(self._to_numpy(self.integration_step.value, u.ms), state_shape)

        p = {
            'V_th': self._broadcast_to_state(self._to_numpy(self.V_th, u.mV), state_shape),
            'V_reset': self._broadcast_to_state(self._to_numpy(self.V_reset, u.mV), state_shape),
            'g_sp': self._broadcast_to_state(self._to_numpy(self.g_sp, u.nS), state_shape),
            'g_pd': self._broadcast_to_state(self._to_numpy(self.g_pd, u.nS), state_shape),
            'g_L': self._stack_compartment_parameter('g_L', u.nS, state_shape),
            'C_m': self._stack_compartment_parameter('C_m', u.pF, state_shape),
            'E_ex': self._stack_compartment_parameter('E_ex', u.mV, state_shape),
            'E_in': self._stack_compartment_parameter('E_in', u.mV, state_shape),
            'E_L': self._stack_compartment_parameter('E_L', u.mV, state_shape),
            'tau_syn_ex': self._stack_compartment_parameter('tau_syn_ex', u.ms, state_shape),
            'tau_syn_in': self._stack_compartment_parameter('tau_syn_in', u.ms, state_shape),
            'I_e': self._stack_compartment_parameter('I_e', u.pA, state_shape),
        }

        refr_counts = self._broadcast_to_state(
            np.asarray(u.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )

        weights = self._parse_spike_events(spike_events, state_shape)
        weights += self._parse_registered_spike_inputs(state_shape)

        new_i_stim = self._parse_current_argument(x, state_shape)
        new_i_stim += self._parse_current_events(current_events, state_shape)
        new_i_stim += self._parse_registered_current_inputs(state_shape)

        pscon_ex = np.e / p['tau_syn_ex']
        pscon_in = np.e / p['tau_syn_in']

        v_for_spike = np.empty(state_shape, dtype=np.float64)
        spike_mask = np.zeros(state_shape, dtype=bool)

        V_next = np.empty_like(V)
        dg_ex_next = np.empty_like(dg_ex)
        g_ex_next = np.empty_like(g_ex)
        dg_in_next = np.empty_like(dg_in)
        g_in_next = np.empty_like(g_in)
        r_next = np.empty_like(r)
        h_next = np.empty_like(h_int)

        for idx in np.ndindex(state_shape):
            local_p = {
                'V_th': p['V_th'][idx],
                'V_reset': p['V_reset'][idx],
                'g_sp': p['g_sp'][idx],
                'g_pd': p['g_pd'][idx],
                'g_L': p['g_L'][idx],
                'C_m': p['C_m'][idx],
                'E_ex': p['E_ex'][idx],
                'E_in': p['E_in'][idx],
                'E_L': p['E_L'][idx],
                'tau_syn_ex': p['tau_syn_ex'][idx],
                'tau_syn_in': p['tau_syn_in'][idx],
                'I_e': p['I_e'][idx],
            }
            is_refractory = r[idx] > 0

            y0 = np.empty(self.NCOMP * self.NSTATE_COMP, dtype=np.float64)
            for c in range(self.NCOMP):
                y0[self._state_index(c, self.V_M)] = V[idx + (c,)]
                y0[self._state_index(c, self.DG_EXC)] = dg_ex[idx + (c,)]
                y0[self._state_index(c, self.G_EXC)] = g_ex[idx + (c,)]
                y0[self._state_index(c, self.DG_INH)] = dg_in[idx + (c,)]
                y0[self._state_index(c, self.G_INH)] = g_in[idx + (c,)]

            y, h_i = self._rkf45_integrate_scalar(
                y0,
                is_refractory,
                i_stim[idx],
                h_int[idx],
                dt,
                local_p,
            )

            for c in range(self.NCOMP):
                y[self._state_index(c, self.DG_EXC)] += pscon_ex[idx + (c,)] * weights[idx + (2 * c,)]
                y[self._state_index(c, self.DG_INH)] += pscon_in[idx + (c,)] * weights[idx + (2 * c + 1,)]

            v_soma = y[self._state_index(self.SOMA, self.V_M)]
            if is_refractory:
                v_for_spike[idx] = local_p['V_reset']
                y[self._state_index(self.SOMA, self.V_M)] = local_p['V_reset']
                r_i = r[idx] - 1
            else:
                v_for_spike[idx] = v_soma
                if v_soma >= local_p['V_th']:
                    spike_mask[idx] = True
                    y[self._state_index(self.SOMA, self.V_M)] = local_p['V_reset']
                    r_i = refr_counts[idx]
                else:
                    r_i = 0

            for c in range(self.NCOMP):
                V_next[idx + (c,)] = y[self._state_index(c, self.V_M)]
                dg_ex_next[idx + (c,)] = y[self._state_index(c, self.DG_EXC)]
                g_ex_next[idx + (c,)] = y[self._state_index(c, self.G_EXC)]
                dg_in_next[idx + (c,)] = y[self._state_index(c, self.DG_INH)]
                g_in_next[idx + (c,)] = y[self._state_index(c, self.G_INH)]

            r_next[idx] = r_i
            h_next[idx] = h_i

        self.V.value = V_next * u.mV
        self.dg_ex.value = dg_ex_next
        self.g_ex.value = g_ex_next * u.nS
        self.dg_in.value = dg_in_next
        self.g_in.value = g_in_next * u.nS
        self.refractory_step_count.value = jnp.asarray(r_next, dtype=jnp.int32)
        self.integration_step.value = h_next * u.ms
        self.I_stim.value = new_i_stim * u.pA
        self.last_spike_time.value = jax.lax.stop_gradient(
            u.math.where(spike_mask, t + dt_q, self.last_spike_time.value)
        )

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return self.get_spike(u.math.asarray(v_for_spike, dtype=jnp.float32) * u.mV)
