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

    **Short Description**

    Three-compartment conductance-based leaky integrate-and-fire neuron with
    alpha-shaped synapses, following NEST ``models/iaf_cond_alpha_mc.{h,cpp}``.

    **Description**

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

    **1. Membrane and Synaptic Dynamics**

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

    **2. Spike and Refractory Semantics**

    - Spike is emitted if somatic membrane potential satisfies
      :math:`V_s \ge V_{th}` after integration.
    - On spike: somatic voltage is reset to ``V_reset`` and refractory counter
      is set to ``ceil(t_ref / dt)``.
    - During refractory period, the ODE uses ``V_reset`` as somatic effective
      voltage and keeps all membrane derivatives at zero, matching NEST C++
      implementation.

    **3. NEST Receptor Types**

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

    **4. Update Order (NEST Semantics)**

    Per simulation step:

    1. Integrate ODEs on :math:`(t, t+dt]` using RKF45 with adaptive substeps.
    2. Apply incoming spike events to ``dg_ex`` / ``dg_in`` per receptor type.
    3. Apply refractory countdown / threshold test / reset / spike emission.
    4. Store incoming currents into delayed buffer ``I_stim`` for next step.

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape (required). Defines the dimensionality of the neuron population.
        Scalar for 1D populations, tuple for multi-dimensional arrays.
    V_th : ArrayLike, optional
        Somatic spike threshold potential. Must be greater than ``V_reset``.
        Default: ``-55.0 * u.mV``.
    V_reset : ArrayLike, optional
        Somatic reset potential after spike emission. Must be less than ``V_th``.
        Default: ``-60.0 * u.mV``.
    t_ref : ArrayLike, optional
        Absolute refractory period duration. Must be non-negative.
        Default: ``2.0 * u.ms``.
    g_sp : ArrayLike, optional
        Soma-proximal coupling conductance. Controls current flow between soma and proximal dendrite.
        Default: ``2.5 * u.nS``.
    g_pd : ArrayLike, optional
        Proximal-distal coupling conductance. Controls current flow between proximal and distal dendrites.
        Default: ``1.0 * u.nS``.
    soma : dict or None, optional
        Per-compartment parameters for soma. Overrides default values.
        Supported keys: ``g_L``, ``C_m``, ``E_ex``, ``E_in``, ``E_L``, ``tau_syn_ex``, ``tau_syn_in``, ``I_e``.
        Default: ``None`` (uses NEST defaults).
    proximal : dict or None, optional
        Per-compartment parameters for proximal dendrite. Same keys as ``soma``.
        Default: ``None`` (uses NEST defaults).
    distal : dict or None, optional
        Per-compartment parameters for distal dendrite. Same keys as ``soma``.
        Default: ``None`` (uses NEST defaults).
    V_initializer : Callable, dict, ArrayLike, or None, optional
        Initial membrane potential specification. Can be:

        - ``None``: use each compartment's ``E_L``
        - dict with keys ``'soma'``, ``'proximal'``, ``'distal'``: per-compartment initialization
        - Callable or ArrayLike: same value for all compartments

        Default: ``None``.
    spk_fun : Callable, optional
        Surrogate gradient function for differentiable spike generation.
        Must accept voltage scaled as ``(V - V_th) / (V_th - V_reset)`` and return spike probability.
        Default: ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Spike reset mode. Options:

        - ``'hard'``: stop gradient at reset (matches NEST behavior)
        - ``'soft'``: allow gradient through reset

        Default: ``'hard'``.
    ref_var : bool, optional
        If ``True``, expose boolean ``refractory`` state variable indicating refractory status.
        Default: ``False``.
    name : str or None, optional
        Optional name for the neuron population. Default: ``None``.

    **Parameter Mapping (Per-Compartment Defaults)**

    The following table shows default per-compartment parameters matching NEST:

    ================== =============== ================= ================= =============================================
    **Parameter**      **Soma**        **Proximal**      **Distal**        **Description**
    ================== =============== ================= ================= =============================================
    ``g_L``            10.0 nS         5.0 nS            10.0 nS           Leak conductance
    ``C_m``            150.0 pF        75.0 pF           150.0 pF          Membrane capacitance
    ``E_ex``           0.0 mV          0.0 mV            0.0 mV            Excitatory reversal potential
    ``E_in``           -85.0 mV        -85.0 mV          -85.0 mV          Inhibitory reversal potential
    ``E_L``            -70.0 mV        -70.0 mV          -70.0 mV          Leak reversal potential
    ``tau_syn_ex``     0.5 ms          0.5 ms            0.5 ms            Excitatory synapse time constant
    ``tau_syn_in``     2.0 ms          2.0 ms            2.0 ms            Inhibitory synapse time constant
    ``I_e``            0.0 pA          0.0 pA            0.0 pA            Constant external current
    ================== =============== ================= ================= =============================================

    **State Variables**

    The model maintains the following state variables:

    - ``V`` : ArrayLike, shape ``[..., 3]``
        Compartment membrane potentials in order ``(soma, proximal, distal)``.
        Units: ``mV``.
    - ``dg_ex`` : ArrayLike, shape ``[..., 3]``
        Excitatory alpha synapse auxiliary variable (dimensionless).
    - ``g_ex`` : ArrayLike, shape ``[..., 3]``
        Excitatory synaptic conductance. Units: ``nS``.
    - ``dg_in`` : ArrayLike, shape ``[..., 3]``
        Inhibitory alpha synapse auxiliary variable (dimensionless).
    - ``g_in`` : ArrayLike, shape ``[..., 3]``
        Inhibitory synaptic conductance. Units: ``nS``.
    - ``I_stim`` : ArrayLike, shape ``[..., 3]``
        One-step delayed current buffer per compartment. Units: ``pA``.
    - ``refractory_step_count`` : ArrayLike, dtype int32
        Somatic refractory countdown (steps remaining).
    - ``integration_step`` : ArrayLike
        Persistent RKF45 internal step size. Units: ``ms``.
    - ``last_spike_time`` : ArrayLike
        Last emitted spike time. Units: ``ms``.
    - ``refractory`` : ArrayLike, dtype bool (optional)
        Boolean refractory indicator (only if ``ref_var=True``).

    Notes
    -----
    **Implementation Details:**

    - **Adaptive Integration:** Uses Runge-Kutta-Fehlberg 45 (RKF45) with adaptive step size control
      to integrate ODEs. The internal step size is persistent across time steps and adjusts based on
      local error estimates (tolerance ``1e-3``).

    - **Refractory Clamping:** During refractory period, somatic voltage is clamped to ``V_reset``
      and all membrane derivatives are set to zero. This matches NEST's C++ implementation exactly.

    - **Spike Weight Routing:** Incoming spike weights are routed to specific compartment-receptor
      combinations (6 spike receptor types). Weights must be non-negative as per NEST semantics.

    - **Current Delay Buffer:** External currents applied via ``x`` parameter are stored in
      ``I_stim`` and used in the *next* time step, implementing NEST's one-step delay semantics.

    - **Surrogate Gradients:** The ``spk_fun`` enables gradient-based learning by providing
      differentiable spike approximations. The voltage is scaled to ``[0, 1]`` range before
      applying the surrogate function.

    **Deprecation Notice:**

    NEST marks ``iaf_cond_alpha_mc`` as deprecated in favor of the more general ``cm_default``
    multi-compartment model. This implementation maintains backward compatibility with legacy
    NEST code and benchmarks.

    **Failure Modes:**

    - Raises ``ValueError`` if ``V_reset >= V_th`` (reset must be below threshold).
    - Raises ``ValueError`` if ``t_ref < 0`` (negative refractory period).
    - Raises ``ValueError`` if any capacitance ``C_m <= 0`` (must be strictly positive).
    - Raises ``ValueError`` if any time constant ``tau_syn_ex`` or ``tau_syn_in <= 0``.
    - Raises ``ValueError`` if spike weights are negative (non-negative constraint).
    - Raises ``TypeError`` if compartment parameter overrides are not dictionaries.
    - Raises ``ValueError`` if unknown keys are provided in compartment parameter dictionaries.

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

    Examples
    --------
    Basic three-compartment neuron with default parameters:

    .. code-block:: python

        >>> import brainpy.state as bst
        >>> import brainunit as u
        >>> neuron = bst.iaf_cond_alpha_mc(in_size=10)
        >>> neuron.init_state()
        >>> spike = neuron.update(x=100. * u.pA)

    Custom per-compartment parameters:

    .. code-block:: python

        >>> soma_params = {'C_m': 200.0 * u.pF, 'g_L': 15.0 * u.nS}
        >>> proximal_params = {'C_m': 100.0 * u.pF}
        >>> neuron = bst.iaf_cond_alpha_mc(
        ...     in_size=5,
        ...     soma=soma_params,
        ...     proximal=proximal_params,
        ...     g_sp=3.0 * u.nS
        ... )

    Compartment-specific current injection using dictionary:

    .. code-block:: python

        >>> currents = {
        ...     'soma': 50.0 * u.pA,
        ...     'proximal': 30.0 * u.pA,
        ...     'distal': 20.0 * u.pA
        ... }
        >>> spike = neuron.update(x=currents)

    Receptor-specific spike input:

    .. code-block:: python

        >>> spike_events = [
        ...     ('soma_exc', 5.0 * u.nS),      # receptor type 1
        ...     ('proximal_inh', 3.0 * u.nS),  # receptor type 4
        ...     ('distal_exc', 2.0 * u.nS)     # receptor type 5
        ... ]
        >>> spike = neuron.update(spike_events=spike_events)

    Accessing compartment-specific membrane potentials:

    .. code-block:: python

        >>> V_soma = neuron.V.value[..., neuron.SOMA]
        >>> V_proximal = neuron.V.value[..., neuron.PROX]
        >>> V_distal = neuron.V.value[..., neuron.DIST]
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
        r"""Mapping of receptor labels to numeric receptor type IDs.

        Returns
        -------
        dict
            Dictionary mapping receptor labels (str) to receptor type integers.
            Includes both spike receptors (1-6) and current receptors (7-9).

        Examples
        --------
        .. code-block:: python

            >>> neuron = bst.iaf_cond_alpha_mc(in_size=10)
            >>> neuron.receptor_types
            {'soma_exc': 1, 'soma_inh': 2, 'proximal_exc': 3, ...}
        """
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        r"""List of recordable state variable names.

        Returns
        -------
        list of str
            Names of state variables that can be recorded during simulation.
            Includes compartment-specific voltages, conductances, and refractory status.

        Notes
        -----
        Recordable names:

        - ``'V_m.s'``, ``'V_m.p'``, ``'V_m.d'``: compartment membrane potentials
        - ``'g_ex.s'``, ``'g_ex.p'``, ``'g_ex.d'``: excitatory conductances
        - ``'g_in.s'``, ``'g_in.p'``, ``'g_in.d'``: inhibitory conductances
        - ``'t_ref_remaining'``: remaining refractory time

        Examples
        --------
        .. code-block:: python

            >>> neuron = bst.iaf_cond_alpha_mc(in_size=10)
            >>> neuron.recordables
            ['V_m.s', 'g_ex.s', 'g_in.s', 'V_m.p', ...]
        """
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
        r"""Initialize all state variables.

        Sets initial values for membrane potentials (using ``V_initializer``),
        synaptic conductances (zero), refractory counters (zero), integration
        step size (to ``dt``), and current buffer (zero).

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size for vectorized simulation. If ``None``, no batch
            dimension is added. Default: ``None``.
        **kwargs
            Additional keyword arguments (ignored, for API compatibility).

        Notes
        -----
        - Membrane potentials default to each compartment's ``E_L`` unless
          ``V_initializer`` is provided.
        - Refractory counter is initialized to 0 (not refractory).
        - RKF45 integration step size is initialized to environment ``dt``.
        - Last spike time is initialized to ``-1e7 ms`` (effectively never spiked).
        """
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
        r"""Reset all state variables to initial values.

        Resets membrane potentials, synaptic conductances, refractory counters,
        integration step size, and current buffer. Identical to ``init_state``
        but operates on existing state variables.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension size. If ``None``, uses current batch size.
            Default: ``None``.
        **kwargs
            Additional keyword arguments (ignored, for API compatibility).

        Notes
        -----
        This method is useful for resetting the neuron state between trials
        or experimental conditions without reallocating state variables.
        """
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
        r"""Compute differentiable spike output using surrogate gradient.

        Applies the surrogate gradient function to voltage scaled to the range
        ``[0, 1]`` relative to threshold and reset potentials. This enables
        gradient-based learning while approximating discrete spike behavior.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Somatic membrane potential. If ``None``, uses current ``self.V.value[..., SOMA]``.
            Must have units of voltage or be dimensionless (assumed mV).
            Default: ``None``.

        Returns
        -------
        spike : ArrayLike
            Differentiable spike output in range ``[0, 1]``. Shape matches ``V`` input.
            Values near 1 indicate high spike probability; near 0 indicates low probability.

        Notes
        -----
        The voltage scaling formula is:

        .. math::

            V_{\\text{scaled}} = \\frac{V - V_{th}}{V_{th} - V_{reset}}

        This maps ``V = V_th`` to 0 and ``V = V_reset`` to -1, allowing the surrogate
        function to produce smooth gradients around the threshold.
        """
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
        r"""Advance the neuron state by one time step.

        Integrates membrane and synaptic ODEs using adaptive RKF45, applies spike
        and current inputs, checks threshold, emits spikes, and updates refractory
        status. Follows NEST update semantics: integrate → apply spikes → check
        threshold → buffer currents for next step.

        Parameters
        ----------
        x : ArrayLike or dict, optional
            External current input. Can be:

            - Scalar or array matching ``in_size``: applied to soma only
            - Array with shape ``[..., 3]``: applied to all compartments
            - Dict with keys ``'soma'``, ``'proximal'``, ``'distal'``, or receptor labels
              (``'soma_curr'``, ``'proximal_curr'``, ``'distal_curr'``): per-compartment currents

            Units: ``pA``. Default: ``0.0 * u.pA``.
        spike_events : list of tuples or dicts, optional
            Incoming spike events. Each element can be:

            - Tuple ``(receptor_type, weight)``
            - Dict with keys ``'receptor_type'`` (or ``'receptor'``) and ``'weight'``

            Receptor types: ``'soma_exc'`` (1), ``'soma_inh'`` (2), ``'proximal_exc'`` (3),
            ``'proximal_inh'`` (4), ``'distal_exc'`` (5), ``'distal_inh'`` (6).
            Weights must be non-negative with units of conductance (``nS``).
            Default: ``None`` (no spike inputs).
        current_events : list of tuples or dicts, optional
            Incoming current events. Each element can be:

            - Tuple ``(receptor_type, current)``
            - Dict with keys ``'receptor_type'`` (or ``'receptor'``) and ``'current'`` (or ``'weight'``)

            Receptor types: ``'soma_curr'`` (7), ``'proximal_curr'`` (8), ``'distal_curr'`` (9),
            or compartment indices 0-2, or names ``'soma'``, ``'proximal'``, ``'distal'``.
            Currents have units of ``pA``.
            Default: ``None`` (no current events).

        Returns
        -------
        spike : ArrayLike
            Differentiable spike output for this time step. Shape matches ``in_size``.
            Values in range ``[0, 1]`` represent spike probability via surrogate gradient.

        Notes
        -----
        **Update Order:**

        1. Integrate ODEs from ``t`` to ``t + dt`` using RKF45 with adaptive substeps
        2. Apply incoming spike weights to alpha synapse auxiliary variables (``dg_ex``, ``dg_in``)
        3. Check somatic voltage against threshold; emit spike if ``V_soma >= V_th``
        4. Reset somatic voltage and set refractory counter on spike
        5. Store incoming currents in ``I_stim`` buffer for use in next time step

        **Refractory Behavior:**

        During refractory period (``refractory_step_count > 0``):

        - Somatic voltage is clamped to ``V_reset``
        - All membrane derivatives are set to zero
        - Synaptic conductances continue to evolve normally
        - Refractory counter decrements by 1 each step

        **Adaptive Integration:**

        RKF45 adjusts internal step size based on local error estimates. The integration
        step size is persistent across time steps and stored in ``integration_step`` state.
        Error tolerance is ``1e-3`` with minimum step size ``1e-8 ms``.

        **Failure Modes:**

        - Raises ``ValueError`` if spike weights are negative (NEST constraint).
        - May fail to converge if integration step size becomes too small (>10000 iterations).

        Examples
        --------
        Simple current injection to soma:

        .. code-block:: python

            >>> spike = neuron.update(x=100.0 * u.pA)

        Multi-compartment current injection:

        .. code-block:: python

            >>> currents = {
            ...     'soma': 50.0 * u.pA,
            ...     'proximal': 30.0 * u.pA,
            ...     'distal': 20.0 * u.pA
            ... }
            >>> spike = neuron.update(x=currents)

        Spike input to proximal dendrite:

        .. code-block:: python

            >>> spike_events = [('proximal_exc', 2.5 * u.nS)]
            >>> spike = neuron.update(spike_events=spike_events)

        Combined spike and current inputs:

        .. code-block:: python

            >>> spike_events = [
            ...     ('soma_exc', 3.0 * u.nS),
            ...     ('distal_inh', 1.5 * u.nS)
            ... ]
            >>> current_events = [('soma_curr', 75.0 * u.pA)]
            >>> spike = neuron.update(
            ...     x=50.0 * u.pA,
            ...     spike_events=spike_events,
            ...     current_events=current_events
            ... )
        """
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
