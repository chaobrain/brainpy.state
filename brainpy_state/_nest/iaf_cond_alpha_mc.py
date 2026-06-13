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

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep, cond_any

__all__ = [
    'iaf_cond_alpha_mc',
]


class iaf_cond_alpha_mc(NESTNeuron):
    r"""NEST-compatible ``iaf_cond_alpha_mc`` neuron model.

    Short Description
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
    gsl_error_tol : ArrayLike, optional
        Unitless local RKF45 error tolerance, broadcastable and strictly positive.
        Default: ``1e-3``.
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

    Parameter Mapping
    -----------------

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

    State Variables
    ---------------

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

    **Failure Modes**


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

        >>> from brainpy import state as bst
        >>> import saiunit as u
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

    _MIN_H = 1e-8 * u.ms  # ms
    _MAX_ITERS = 100000

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
        gsl_error_tol: ArrayLike = 1e-3,
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

        self.gsl_error_tol = gsl_error_tol
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

        self.integrator = AdaptiveRungeKuttaStep(
            method='RKF45',
            vf=self._vector_field,
            event_fn=self._event_fn,
            min_h=self._MIN_H,
            max_iters=self._MAX_ITERS,
            atol=self.gsl_error_tol,
            dt=brainstate.environ.get_dt()
        )

        # other variable
        ditype = brainstate.environ.ditype()
        dt = brainstate.environ.get_dt()
        self.ref_count = u.math.asarray(u.math.ceil(self.t_ref / dt), dtype=ditype)

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

    def _validate_parameters(self):
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.V_reset, self.V_th)):
            return
        if cond_any(self.V_reset >= self.V_th):
            raise ValueError('Reset potential must be smaller than threshold.')
        if cond_any(self.t_ref < 0.0 * u.ms):
            raise ValueError('Refractory time cannot be negative.')

        for comp in ('soma', 'proximal', 'distal'):
            cm = self._compartments[comp]['C_m']
            tau_ex = self._compartments[comp]['tau_syn_ex']
            tau_in = self._compartments[comp]['tau_syn_in']
            if cond_any(cm <= 0.0 * u.pF):
                raise ValueError(f'Capacitance ({comp}) must be strictly positive.')
            if cond_any(tau_ex <= 0.0 * u.ms) or cond_any(tau_in <= 0.0 * u.ms):
                raise ValueError(f'All time constants ({comp}) must be strictly positive.')

        if cond_any(self.gsl_error_tol <= 0.0):
            raise ValueError('The gsl_error_tol must be strictly positive.')

    def _initial_membrane_potential(self):
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

        v_s = braintools.init.param(init_cfg['soma'], self.varshape)
        v_p = braintools.init.param(init_cfg['proximal'], self.varshape)
        v_d = braintools.init.param(init_cfg['distal'], self.varshape)

        # braintools.init.param may return a scalar Quantity when the initializer
        # is a concrete value; broadcast explicitly so V has shape (*varshape, 3).
        v_s = u.math.broadcast_to(v_s, self.varshape)
        v_p = u.math.broadcast_to(v_p, self.varshape)
        v_d = u.math.broadcast_to(v_d, self.varshape)

        v_stack = u.math.stack([v_s, v_p, v_d], axis=-1)
        return v_stack

    def _stack_compartment_param(self, key):
        """Stack a per-compartment parameter along the last axis.

        Parameters
        ----------
        key : str
            Parameter key (e.g. ``'g_L'``, ``'C_m'``).

        Returns
        -------
        Quantity
            Stacked array with shape ``(*varshape, 3)``.
        """
        vals = [
            self._compartments[comp][key]
            for comp in ('soma', 'proximal', 'distal')
        ]
        return u.math.stack(vals, axis=-1)

    def _parse_x_input(self, x, dftype):
        """Parse x (scalar, array, or compartment dict) into a (*varshape, 3) pA array.

        Parameters
        ----------
        x : ArrayLike or dict
            External current input.  Scalar/array is applied to soma only.
            Dict keys: ``'soma'``, ``'proximal'``, ``'distal'`` (or receptor-type
            aliases ``'soma_curr'``, ``'proximal_curr'``, ``'distal_curr'``).
        dftype : dtype
            Float dtype for the zeros baseline.

        Returns
        -------
        Quantity, shape (*varshape, 3), unit pA
        """
        zeros_pA = u.math.zeros(self.varshape, dtype=dftype) * u.pA
        if isinstance(x, dict):
            i_s = zeros_pA
            i_p = zeros_pA
            i_d = zeros_pA
            for key, val in x.items():
                val_bc = u.math.broadcast_to(val, self.varshape)
                if key in ('soma', 'soma_curr', 7):
                    i_s = i_s + val_bc
                elif key in ('proximal', 'proximal_curr', 8):
                    i_p = i_p + val_bc
                elif key in ('distal', 'distal_curr', 9):
                    i_d = i_d + val_bc
                else:
                    raise ValueError(
                        f'Unknown current input key {key!r}. '
                        f'Valid keys: soma, proximal, distal, '
                        f'soma_curr, proximal_curr, distal_curr'
                    )
            return u.math.stack([i_s, i_p, i_d], axis=-1)
        else:
            # Scalar or array: apply to soma only (NEST convention).
            i_s = u.math.broadcast_to(x, self.varshape)
            return u.math.stack([i_s, zeros_pA, zeros_pA], axis=-1)

    # Spike receptor type (1-6) -> the labeled delta-input channel the model reads
    # back with ``sum_delta_inputs(label=...)`` (compartment × synapse type). The
    # Simulator's projection layer uses :meth:`delta_label_for_receptor` to route a
    # ``connect(spike_source, mc, receptor_type=k)`` deposit into this one channel.
    _SPIKE_RECEPTOR_DELTA_LABEL = {
        1: 'w_ex_s', 2: 'w_in_s',   # soma     exc / inh
        3: 'w_ex_p', 4: 'w_in_p',   # proximal exc / inh
        5: 'w_ex_d', 6: 'w_in_d',   # distal   exc / inh
    }

    def delta_label_for_receptor(self, receptor_type):
        r"""Map a spike receptor type (1-6) to its delta-input channel label.

        The Simulator routes ``connect(spike_source, mc, receptor_type=k)`` into the
        single conductance channel the model reads with ``sum_delta_inputs(label=...)``
        (see :meth:`update`). Current receptors (7-9) are rejected here: they carry an
        injected current and must be driven by a current generator.

        Parameters
        ----------
        receptor_type : int
            NEST receptor type. Spike receptors are ``1-6`` (soma/proximal/distal ×
            exc/inh; see :attr:`SPIKE_RECEPTOR_TYPES`).

        Returns
        -------
        str
            The delta-input channel label, one of ``'w_ex_s'``, ``'w_in_s'``,
            ``'w_ex_p'``, ``'w_in_p'``, ``'w_ex_d'``, ``'w_in_d'``.

        Raises
        ------
        ValueError
            If ``receptor_type`` is a current receptor (7-9) or out of range.
        """
        try:
            rt = int(receptor_type)
        except (TypeError, ValueError):
            raise ValueError(
                f'receptor_type must be an integer spike receptor 1-6, got {receptor_type!r}.'
            )
        if rt in self._SPIKE_RECEPTOR_DELTA_LABEL:
            return self._SPIKE_RECEPTOR_DELTA_LABEL[rt]
        if rt in self.CURRENT_RECEPTOR_TYPES.values():
            raise ValueError(
                f'receptor_type {rt} is a current receptor (soma_curr=7, proximal_curr=8, '
                f'distal_curr=9); drive it with a current generator '
                f'(dc_/step_/ac_/noise_generator), not a spike source.'
            )
        raise ValueError(
            f'invalid spike receptor_type {rt}; valid spike receptors are 1-6 '
            f'({sorted(self.SPIKE_RECEPTOR_TYPES.values())}).'
        )

    def current_compartment_for_receptor(self, receptor_type):
        r"""Map a current receptor type (7-9) to its compartment index.

        The Simulator routes ``connect(current_generator, mc, receptor_type=k)`` into a
        single compartment's injected current. Spike receptors (1-6) are rejected here:
        they carry a conductance weight and must be driven by a spike source.

        Parameters
        ----------
        receptor_type : int
            NEST receptor type. Current receptors are ``7`` (soma), ``8`` (proximal),
            ``9`` (distal); see :attr:`CURRENT_RECEPTOR_TYPES`.

        Returns
        -------
        int
            The compartment index (:attr:`SOMA` ``=0``, :attr:`PROX` ``=1``,
            :attr:`DIST` ``=2``).

        Raises
        ------
        ValueError
            If ``receptor_type`` is a spike receptor (1-6) or out of range.
        """
        try:
            rt = int(receptor_type)
        except (TypeError, ValueError):
            raise ValueError(
                f'receptor_type must be an integer current receptor 7-9, got {receptor_type!r}.'
            )
        mapping = {7: self.SOMA, 8: self.PROX, 9: self.DIST}
        if rt in mapping:
            return mapping[rt]
        if rt in self.SPIKE_RECEPTOR_TYPES.values():
            raise ValueError(
                f'receptor_type {rt} is a spike receptor (1-6); drive it with a spike '
                f'source (spike_generator or a neuron), not a current generator.'
            )
        raise ValueError(
            f'invalid current receptor_type {rt}; valid current receptors are 7-9 '
            f'({sorted(self.CURRENT_RECEPTOR_TYPES.values())}).'
        )

    def init_state(self, **kwargs):
        r"""Initialize all state variables.

        Sets initial values for membrane potentials (using ``V_initializer``),
        synaptic conductances (zero), refractory counters (zero), integration
        step size (to ``dt``), and current buffer (zero).

        Parameters
        ----------
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
        ditype = brainstate.environ.ditype()
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        V = self._initial_membrane_potential()
        comp_shape = self.varshape + (self.NCOMP,)

        self.V = brainstate.HiddenState(V)
        # dg_ex / dg_in are stored as plain (dimensionless) JAX arrays whose numeric
        # value represents conductance-derivative in nS/ms.  This matches NEST's flat
        # double state-vector convention and allows direct np.asarray() access.
        self.dg_ex = brainstate.ShortTermState(jnp.zeros(comp_shape, dtype=V.dtype))
        self.g_ex = brainstate.HiddenState(u.math.zeros(comp_shape, dtype=dftype) * u.nS)
        self.dg_in = brainstate.ShortTermState(jnp.zeros(comp_shape, dtype=dftype))
        self.g_in = brainstate.HiddenState(u.math.zeros(comp_shape, dtype=dftype) * u.nS)

        self.last_spike_time = brainstate.ShortTermState(u.math.full(self.varshape, -1e7 * u.ms))
        self.refractory_step_count = brainstate.ShortTermState(u.math.full(self.varshape, 0, dtype=ditype))
        self.integration_step = brainstate.ShortTermState.init(braintools.init.Constant(dt), self.varshape)
        self.I_stim = brainstate.ShortTermState(u.math.zeros(comp_shape, dtype=dftype) * u.pA)

        if self.ref_var:
            refractory = braintools.init.param(braintools.init.Constant(False), self.varshape)
            self.refractory = brainstate.ShortTermState(refractory)

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

    def _vector_field(self, state, extra):
        """Unit-aware vectorized RHS for all compartments simultaneously.

        Parameters
        ----------
        state : DotDict
            Keys: V_s, V_p, V_d, dg_ex_s, g_ex_s, dg_in_s, g_in_s,
            dg_ex_p, g_ex_p, dg_in_p, g_in_p, dg_ex_d, g_ex_d, dg_in_d, g_in_d
            -- ODE state variables.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim -- mutable auxiliary data
            carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.
        """
        is_refractory = extra.r > 0

        # Effective voltages: soma clamped during refractory; otherwise clamped
        # to V_th (NEST convention: v_eff = min(V_s, V_th) when non-refractory).
        # This matches NEST's iaf_cond_alpha_mc ODE which uses
        #   v_eff = V_reset if refractory else min(V_s, V_th)
        # The threshold clamp creates a kink in the ODE at V_s = V_th that
        # the adaptive step-size controller can detect, keeping h in sync with
        # the reference implementation.
        v_s_eff = u.math.where(is_refractory, self.V_reset, u.math.minimum(state.V_s, self.V_th))
        v_p_eff = state.V_p
        v_d_eff = state.V_d

        # Coupling currents.
        # Soma uses the clamped v_s_eff = min(V_s, V_th); proximal and distal
        # use the raw (un-clamped) state.V_s when computing their inter-compartment
        # coupling.  This exactly mirrors the per-compartment loop in NEST's
        # _dynamics_ref where each compartment has its own local v_eff and the
        # OTHER compartment's raw state vector entry is used for the coupling term.
        i_conn_s = self.g_sp * (v_s_eff - v_p_eff)
        i_conn_p = self.g_sp * (v_p_eff - state.V_s) + self.g_pd * (v_p_eff - v_d_eff)
        i_conn_d = self.g_pd * (v_d_eff - v_p_eff)

        # --- Soma ---
        i_syn_ex_s = state.g_ex_s * (v_s_eff - self.soma['E_ex'])
        i_syn_in_s = state.g_in_s * (v_s_eff - self.soma['E_in'])
        i_leak_s = self.soma['g_L'] * (v_s_eff - self.soma['E_L'])
        dV_s_raw = (-i_leak_s - i_syn_ex_s - i_syn_in_s - i_conn_s
                    + extra.i_stim[..., self.SOMA] + self.soma['I_e']) / self.soma['C_m']
        dV_s = u.math.where(is_refractory, u.math.zeros_like(dV_s_raw), dV_s_raw)

        # dg_ex_s / dg_in_s are dimensionless (value in nS/ms).
        # Their time derivative (ddg) has units 1/ms so that h[ms]*ddg = dimensionless. ✓
        # The derivative of g_ex_s (units nS) must have units nS/ms:
        #   dg_ex_s * (nS/ms) converts the dimensionless auxiliary variable to nS/ms. ✓
        ddg_ex_s = -state.dg_ex_s / self.soma['tau_syn_ex']
        dg_ex_s = state.dg_ex_s * (u.nS / u.ms) - state.g_ex_s / self.soma['tau_syn_ex']
        ddg_in_s = -state.dg_in_s / self.soma['tau_syn_in']
        dg_in_s = state.dg_in_s * (u.nS / u.ms) - state.g_in_s / self.soma['tau_syn_in']

        # --- Proximal ---
        i_syn_ex_p = state.g_ex_p * (v_p_eff - self.proximal['E_ex'])
        i_syn_in_p = state.g_in_p * (v_p_eff - self.proximal['E_in'])
        i_leak_p = self.proximal['g_L'] * (v_p_eff - self.proximal['E_L'])
        dV_p_raw = (-i_leak_p - i_syn_ex_p - i_syn_in_p - i_conn_p
                    + extra.i_stim[..., self.PROX] + self.proximal['I_e']) / self.proximal['C_m']
        # NEST freezes ALL compartment voltages during soma refractory (reference:
        # f[V_M] = 0.0 if is_refractory else ... for ALL n in {SOMA, PROX, DIST}).
        dV_p = u.math.where(is_refractory, u.math.zeros_like(dV_p_raw), dV_p_raw)

        ddg_ex_p = -state.dg_ex_p / self.proximal['tau_syn_ex']
        dg_ex_p = state.dg_ex_p * (u.nS / u.ms) - state.g_ex_p / self.proximal['tau_syn_ex']
        ddg_in_p = -state.dg_in_p / self.proximal['tau_syn_in']
        dg_in_p = state.dg_in_p * (u.nS / u.ms) - state.g_in_p / self.proximal['tau_syn_in']

        # --- Distal ---
        i_syn_ex_d = state.g_ex_d * (v_d_eff - self.distal['E_ex'])
        i_syn_in_d = state.g_in_d * (v_d_eff - self.distal['E_in'])
        i_leak_d = self.distal['g_L'] * (v_d_eff - self.distal['E_L'])
        dV_d_raw = (-i_leak_d - i_syn_ex_d - i_syn_in_d - i_conn_d
                    + extra.i_stim[..., self.DIST] + self.distal['I_e']) / self.distal['C_m']
        # Same NEST convention: distal voltage also frozen during refractory.
        dV_d = u.math.where(is_refractory, u.math.zeros_like(dV_d_raw), dV_d_raw)

        ddg_ex_d = -state.dg_ex_d / self.distal['tau_syn_ex']
        dg_ex_d = state.dg_ex_d * (u.nS / u.ms) - state.g_ex_d / self.distal['tau_syn_ex']
        ddg_in_d = -state.dg_in_d / self.distal['tau_syn_in']
        dg_in_d = state.dg_in_d * (u.nS / u.ms) - state.g_in_d / self.distal['tau_syn_in']

        return DotDict(
            V_s=dV_s, V_p=dV_p, V_d=dV_d,
            dg_ex_s=ddg_ex_s, g_ex_s=dg_ex_s, dg_in_s=ddg_in_s, g_in_s=dg_in_s,
            dg_ex_p=ddg_ex_p, g_ex_p=dg_ex_p, dg_in_p=ddg_in_p, g_in_p=dg_in_p,
            dg_ex_d=ddg_ex_d, g_ex_d=dg_ex_d, dg_in_d=ddg_in_d, g_in_d=dg_in_d,
        )

    def _event_fn(self, state, extra, accept):
        """Stability monitoring callback (called after each accepted RK substep).

        Spike detection and refractory updates are handled *outside* the
        integrator (in ``update()``) to match NEST's reference semantics:
        the refractory flag is fixed for the entire dt interval, and threshold
        crossing is checked once after the integration completes.

        Parameters
        ----------
        state : DotDict
            ODE state variables for all compartments.
        extra : DotDict
            Keys: spike_mask, r, unstable, i_stim.
        accept : array, bool
            Mask of neurons whose RK substep was accepted.

        Returns
        -------
        (state, new_extra) with only the ``unstable`` flag updated.
        """
        unstable = extra.unstable | jnp.any(
            accept & (state.V_s < -1e3 * u.mV)
        )
        new_extra = DotDict({**extra, 'unstable': unstable})
        return state, new_extra

    def update(self, x=0. * u.pA, spike_events=None, current_events=None):
        r"""Advance the neuron state by one time step.

        Integrates membrane and synaptic ODEs using adaptive RKF45, applies spike
        and current inputs, checks threshold, emits spikes, and updates refractory
        status. Follows NEST update semantics: integrate -> apply spikes -> check
        threshold -> buffer currents for next step.

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

        **Failure Modes**


        - Raises ``ValueError`` if spike weights are negative (NEST constraint).
        - May fail to converge if integration step size becomes too small.
        """
        t = brainstate.environ.get('t')
        dt = brainstate.environ.get_dt()
        dftype = brainstate.environ.dftype()
        ditype = brainstate.environ.ditype()

        # Read state variables with their natural units.
        V = self.V.value  # mV, shape (*varshape, 3)
        dg_ex = self.dg_ex.value  # nS/ms, shape (*varshape, 3)
        g_ex = self.g_ex.value  # nS, shape (*varshape, 3)
        dg_in = self.dg_in.value  # nS/ms, shape (*varshape, 3)
        g_in = self.g_in.value  # nS, shape (*varshape, 3)
        r = self.refractory_step_count.value  # int
        i_stim = self.I_stim.value  # pA, shape (*varshape, 3)
        h = self.integration_step.value  # ms

        # Parse x into per-compartment array (*varshape, 3) for next step (one-step delay).
        x_arr = self._parse_x_input(x, dftype)
        new_i_stim = self.sum_current_inputs(x_arr, self.V.value[..., self.SOMA])  # pA

        # Build ODE state as flat per-compartment DotDict
        ode_state = DotDict(
            V_s=V[..., self.SOMA], V_p=V[..., self.PROX], V_d=V[..., self.DIST],
            dg_ex_s=dg_ex[..., self.SOMA], g_ex_s=g_ex[..., self.SOMA],
            dg_in_s=dg_in[..., self.SOMA], g_in_s=g_in[..., self.SOMA],
            dg_ex_p=dg_ex[..., self.PROX], g_ex_p=g_ex[..., self.PROX],
            dg_in_p=dg_in[..., self.PROX], g_in_p=g_in[..., self.PROX],
            dg_ex_d=dg_ex[..., self.DIST], g_ex_d=g_ex[..., self.DIST],
            dg_in_d=dg_in[..., self.DIST], g_in_d=g_in[..., self.DIST],
        )
        extra = DotDict(
            spike_mask=jnp.zeros(self.varshape, dtype=jnp.bool_),
            r=r,
            unstable=jnp.array(False),
            i_stim=i_stim,
        )

        # Adaptive RKF45 integration via generic integrator.
        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        unstable = extra.unstable
        # extra.r is unchanged: _event_fn no longer modifies it, so it equals
        # the initial r read above.  Use the original r variable directly.

        # Post-loop stability check.
        brainstate.transform.jit_error_if(
            jnp.any(unstable), 'Numerical instability in iaf_cond_alpha_mc dynamics.'
        )

        # Post-integration spike detection and refractory update (NEST semantics).
        # This mirrors _reference_step / _rkf45_ref_step: refractory state is
        # fixed during integration; threshold is checked once after the full dt.
        refr_active = r > 0
        spike_mask = (~refr_active) & (ode_state.V_s >= self.V_th)

        # Reset somatic voltage: V_reset during refractory, V_reset on spike.
        V_s_post = u.math.where(refr_active | spike_mask, self.V_reset, ode_state.V_s)

        # Update refractory counter: set to ref_count on new spike, decrement
        # by 1 if already refractory, else keep at 0.
        r_new = jnp.asarray(
            u.get_mantissa(
                u.math.where(
                    spike_mask,
                    self.ref_count,
                    u.math.where(refr_active, r - 1, r),
                )
            ),
            dtype=ditype,
        )

        # Synaptic spike inputs (applied after integration).
        # Use dimensionless pscon (value in 1/ms) so that dg_ex (dimensionless) stays
        # dimensionless after: dg_ex += pscon * w_nS.
        tau_syn_ex = self._stack_compartment_param('tau_syn_ex')  # (*varshape, 3), ms
        tau_syn_in = self._stack_compartment_param('tau_syn_in')  # (*varshape, 3), ms
        pscon_ex = np.e / u.get_mantissa(tau_syn_ex / u.ms)  # (*varshape, 3), dimensionless (1/ms)
        pscon_in = np.e / u.get_mantissa(tau_syn_in / u.ms)  # (*varshape, 3), dimensionless (1/ms)

        # Initialize per-receptor spike weights (nS).
        w_ex_s = u.math.zeros_like(g_ex[..., self.SOMA])
        w_in_s = u.math.zeros_like(g_in[..., self.SOMA])
        w_ex_p = u.math.zeros_like(g_ex[..., self.PROX])
        w_in_p = u.math.zeros_like(g_in[..., self.PROX])
        w_ex_d = u.math.zeros_like(g_ex[..., self.DIST])
        w_in_d = u.math.zeros_like(g_in[..., self.DIST])

        # Process spike_events passed directly to update().
        if spike_events is not None:
            for event in spike_events:
                if isinstance(event, dict):
                    rtype = event.get('receptor_type', event.get('receptor'))
                    weight = event.get('weight')
                else:
                    rtype, weight = event
                # Resolve string receptor type to integer.
                if isinstance(rtype, str):
                    if rtype not in self.SPIKE_RECEPTOR_TYPES:
                        raise ValueError(
                            f'Unknown receptor type {rtype!r}. '
                            f'Valid spike receptors: {sorted(self.SPIKE_RECEPTOR_TYPES)}'
                        )
                    rtype = self.SPIKE_RECEPTOR_TYPES[rtype]
                # Non-negative weight constraint (NEST semantics).
                if cond_any(np.asarray(u.get_mantissa(weight)) < 0.0):
                    raise ValueError(
                        f'Spike weights must be non-negative (NEST constraint), got {weight}'
                    )
                # Route weight to the correct compartment and synapse type.
                if rtype == 1:    w_ex_s = w_ex_s + weight   # soma_exc
                elif rtype == 2:  w_in_s = w_in_s + weight   # soma_inh
                elif rtype == 3:  w_ex_p = w_ex_p + weight   # proximal_exc
                elif rtype == 4:  w_in_p = w_in_p + weight   # proximal_inh
                elif rtype == 5:  w_ex_d = w_ex_d + weight   # distal_exc
                elif rtype == 6:  w_in_d = w_in_d + weight   # distal_inh
                else:
                    raise ValueError(
                        f'Invalid spike receptor type {rtype}. Valid spike types: 1-6.'
                    )

        # Add registered delta inputs (from projection layers).
        w_ex_s = self.sum_delta_inputs(w_ex_s, label='w_ex_s')
        w_in_s = self.sum_delta_inputs(w_in_s, label='w_in_s')
        w_ex_p = self.sum_delta_inputs(w_ex_p, label='w_ex_p')
        w_in_p = self.sum_delta_inputs(w_in_p, label='w_in_p')
        w_ex_d = self.sum_delta_inputs(w_ex_d, label='w_ex_d')
        w_in_d = self.sum_delta_inputs(w_in_d, label='w_in_d')

        # Apply synaptic spike inputs.
        # w_* has units nS; pscon_* is dimensionless (1/ms); stripping nS gives the
        # dimensionless update (value in nS/ms) that matches dg_ex / dg_in storage.
        dg_ex_s = ode_state.dg_ex_s + pscon_ex[..., self.SOMA] * u.get_mantissa(w_ex_s / u.nS)
        dg_in_s = ode_state.dg_in_s + pscon_in[..., self.SOMA] * u.get_mantissa(w_in_s / u.nS)
        dg_ex_p = ode_state.dg_ex_p + pscon_ex[..., self.PROX] * u.get_mantissa(w_ex_p / u.nS)
        dg_in_p = ode_state.dg_in_p + pscon_in[..., self.PROX] * u.get_mantissa(w_in_p / u.nS)
        dg_ex_d = ode_state.dg_ex_d + pscon_ex[..., self.DIST] * u.get_mantissa(w_ex_d / u.nS)
        dg_in_d = ode_state.dg_in_d + pscon_in[..., self.DIST] * u.get_mantissa(w_in_d / u.nS)

        # Write back state - reassemble compartment dimension.
        self.V.value = u.math.stack([V_s_post, ode_state.V_p, ode_state.V_d], axis=-1)
        self.dg_ex.value = u.math.stack([dg_ex_s, dg_ex_p, dg_ex_d], axis=-1)
        self.g_ex.value = u.math.stack([ode_state.g_ex_s, ode_state.g_ex_p, ode_state.g_ex_d], axis=-1)
        self.dg_in.value = u.math.stack([dg_in_s, dg_in_p, dg_in_d], axis=-1)
        self.g_in.value = u.math.stack([ode_state.g_in_s, ode_state.g_in_p, ode_state.g_in_d], axis=-1)
        self.refractory_step_count.value = r_new
        self.integration_step.value = h
        self.I_stim.value = new_i_stim + u.math.zeros(self.varshape + (self.NCOMP,)) * u.pA
        last_spike_time = u.math.where(spike_mask, t + dt, self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_time)

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        return u.math.asarray(spike_mask, dtype=dftype)
