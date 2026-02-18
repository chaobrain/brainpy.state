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

from typing import Callable, Iterable

import brainstate
import braintools
import brainunit as bu
import jax
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike, Size

from ._base import NESTNeuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_tum_2000',
]


class iaf_tum_2000(NESTNeuron):
    r"""NEST-compatible ``iaf_tum_2000`` neuron model.

    Description
    -----------

    ``iaf_tum_2000`` is a leaky integrate-and-fire neuron with exponential
    postsynaptic currents and integrated Tsodyks-Markram short-term synaptic
    plasticity. The model extends :class:`iaf_psc_exp` by maintaining
    presynaptic resource states ``x`` (readily-releasable pool), ``y``
    (cleft/active fraction), and ``u`` (release probability), and emitting a
    per-spike ``spike_offset`` signal that encodes the jump in ``y`` at each
    spike event. This signal is used for receptor-1 coupling between
    ``iaf_tum_2000`` neurons, enabling dynamic synaptic efficacy.

    The implementation follows NEST ``models/iaf_tum_2000.{h,cpp}`` update
    ordering and event semantics exactly, including NEST-style buffered input
    handling and receptor-type routing.

    **1. Membrane and synaptic dynamics**

    Subthreshold voltage evolution follows the same equation as
    :class:`iaf_psc_exp`:

    .. math::

       \frac{dV_m}{dt} =
       -\frac{V_m - E_L}{\tau_m} +
       \frac{I_{\mathrm{syn,ex}} + I_{\mathrm{syn,in}} + I_e + I_0}{C_m},

    where ``I_0`` is the buffered current from the previous time step. Synaptic
    currents decay exponentially:

    .. math::

       \frac{dI_{\mathrm{syn,ex}}}{dt} = -\frac{I_{\mathrm{syn,ex}}}{\tau_{\mathrm{syn,ex}}},
       \qquad
       \frac{dI_{\mathrm{syn,in}}}{dt} = -\frac{I_{\mathrm{syn,in}}}{\tau_{\mathrm{syn,in}}}.

    Receptor-1 current input ``I_1`` is filtered through the excitatory kernel:

    .. math::

       I_{\mathrm{syn,ex}} \leftarrow I_{\mathrm{syn,ex}} + (1 - e^{-h/\tau_{\mathrm{syn,ex}}}) I_1,

    where ``h = dt`` is the simulation time step.

    **2. Tsodyks-Markram short-term plasticity on spike**

    When a neuron emits a spike at time ``t_spike``, the Tsodyks states are
    updated. Let ``t_last`` be the previous spike time (with NEST-compatible
    first-spike convention: ``t_last = 0`` if the internal last-spike time is
    negative, indicating no prior spike), and ``h_ts = t_spike - t_last``.

    Define propagators:

    .. math::

       P_{uu} = \begin{cases}
       0, & \tau_{\mathrm{fac}}=0 \\
       e^{-h_{ts}/\tau_{\mathrm{fac}}}, & \text{otherwise}
       \end{cases},
       \quad
       P_{yy} = e^{-h_{ts}/\tau_{\mathrm{psc}}},

    .. math::

       P_{zz} = \mathrm{expm1}(-h_{ts}/\tau_{\mathrm{rec}}) = e^{-h_{ts}/\tau_{\mathrm{rec}}} - 1,

    .. math::

       P_{xy} =
       \frac{P_{zz}\tau_{\mathrm{rec}} - (P_{yy}-1)\tau_{\mathrm{psc}}}{\tau_{\mathrm{psc}}-\tau_{\mathrm{rec}}}.

    With :math:`z = 1 - x - y` (inactive/recovered fraction), NEST performs
    state propagation in this exact order:

    .. math::

       u &\leftarrow u P_{uu}, \\
       x &\leftarrow x + P_{xy}y - P_{zz}z, \\
       y &\leftarrow y P_{yy},

    followed by utilization jump and resource transfer:

    .. math::

       u &\leftarrow u + U(1-u), \\
       \Delta y &= u x, \\
       x &\leftarrow x - \Delta y, \\
       y &\leftarrow y + \Delta y.

    ``spike_offset`` is set to :math:`\Delta y` on spike steps, zero otherwise.

    **3. NEST update ordering**

    Per time step, the model follows this precise sequence:

    1. Update membrane potential if not refractory (exact exponential propagator).
    2. Decay synaptic currents :math:`I_{\mathrm{syn,ex}}` and :math:`I_{\mathrm{syn,in}}`.
    3. Add filtered receptor-1 current to :math:`I_{\mathrm{syn,ex}}`.
    4. Add arriving spike inputs (positive weights to excitatory, non-positive to inhibitory).
    5. Perform threshold test (deterministic or escape-noise), assign refractory and reset.
    6. On emitted spike, update Tsodyks states (using the order above) and set ``spike_offset``.
    7. Buffer current inputs ``i_0`` and ``i_1`` for the next step.

    **4. Escape-noise threshold dynamics**

    Spike generation uses deterministic thresholding when :math:`\delta < 10^{-10}`:
    :math:`V_{\mathrm{rel}} \ge \theta`, where :math:`\theta = V_{th} - E_L`.

    For :math:`\delta > 0`, the model uses exponential hazard:

    .. math::

       \phi(V) = \rho \exp\left(\frac{V_{\mathrm{rel}} - \theta}{\delta}\right),

    with step spike probability :math:`p=\phi(V)\,h\,10^{-3}` (``h`` in ms,
    :math:`\phi` in ``1/s``). Stochastic decisions use ``numpy.random.random``.

    **5. Event semantics and receptor routing**

    The :meth:`update` method accepts ``spike_events`` as an iterable of event
    descriptors in one of these formats:

    - ``(receptor_type, weight)``
    - ``(receptor_type, weight, offset)``
    - ``(receptor_type, weight, offset, multiplicity)``
    - ``(receptor_type, weight, offset, multiplicity, sender_model)``
    - ``dict`` with keys ``receptor_type``/``receptor``, ``weight``, ``offset``,
      ``multiplicity``, ``sender_model``

    Receptors:

    - **Receptor 0** (DEFAULT): regular spike input, effective weight is
      ``weight * multiplicity``.
    - **Receptor 1** (TSODYKS): Tsodyks-coupled input, effective weight is
      ``weight * multiplicity * offset``, where ``offset`` is typically the
      ``spike_offset`` from the presynaptic ``iaf_tum_2000`` neuron.

    For receptor 1, the ``sender_model`` field must be ``"iaf_tum_2000"``
    (default assumption if not provided); otherwise a ``ValueError`` is raised,
    mirroring NEST's connection constraints.

    Effective weights are routed by sign: positive to excitatory channel,
    non-positive to inhibitory channel.

    **6. Stability constraints and computational implications**

    - Construction validates: ``V_reset < V_th``, ``C_m > 0``, ``tau_m > 0``,
      ``tau_syn_ex > 0``, ``tau_syn_in > 0``, ``tau_psc > 0``, ``tau_rec > 0``,
      ``tau_fac >= 0``, ``t_ref >= 0``, ``rho >= 0``, ``delta >= 0``,
      ``x + y <= 1``, ``u ∈ [0,1]``.
    - Tsodyks state propagation uses the same singularity-free logic as NEST to
      handle ``tau_psc == tau_rec`` or ``tau_fac == 0`` cases gracefully.
    - Per-call cost is :math:`O(\prod \mathrm{varshape})` with vectorized
      NumPy operations in ``float64`` for coefficient evaluation.
    - Buffered current semantics match NEST ring-buffer timing: ``x`` and
      ``x_filtered`` supplied at step ``n`` are stored and consumed at step ``n+1``.

    Parameters
    ----------
    in_size : Size
        Population shape specification. All per-neuron parameters are broadcast
        to ``self.varshape``.
    E_L : ArrayLike, optional
        Resting potential :math:`E_L` in mV; scalar or array broadcastable to
        ``self.varshape``. Default is ``-70. * bu.mV``.
    C_m : ArrayLike, optional
        Membrane capacitance :math:`C_m` in pF; broadcastable and strictly
        positive. Default is ``250. * bu.pF``.
    tau_m : ArrayLike, optional
        Membrane time constant :math:`\tau_m` in ms; broadcastable and
        strictly positive. Default is ``10. * bu.ms``.
    t_ref : ArrayLike, optional
        Absolute refractory period :math:`t_{\mathrm{ref}}` in ms;
        broadcastable and nonnegative. Converted to integer steps by
        ``ceil(t_ref / dt)``. Default is ``2. * bu.ms``.
    V_th : ArrayLike, optional
        Spike threshold :math:`V_{th}` in mV; broadcastable to
        ``self.varshape``. Default is ``-55. * bu.mV``.
    V_reset : ArrayLike, optional
        Post-spike reset potential :math:`V_{\mathrm{reset}}` in mV;
        broadcastable and must satisfy ``V_reset < V_th`` elementwise. Default
        is ``-70. * bu.mV``.
    tau_syn_ex : ArrayLike, optional
        Excitatory synaptic decay constant :math:`\tau_{\mathrm{syn,ex}}` in
        ms; broadcastable and strictly positive. Default is ``2. * bu.ms``.
    tau_syn_in : ArrayLike, optional
        Inhibitory synaptic decay constant :math:`\tau_{\mathrm{syn,in}}` in
        ms; broadcastable and strictly positive. Default is ``2. * bu.ms``.
    I_e : ArrayLike, optional
        Constant external injected current :math:`I_e` in pA; scalar or array
        broadcastable to ``self.varshape``. Default is ``0. * bu.pA``.
    rho : ArrayLike, optional
        Escape-noise base firing intensity :math:`\rho` in ``1/s``;
        broadcastable and nonnegative. Used only in stochastic mode
        (``delta > 0``). Default is ``0.01 / bu.second``.
    delta : ArrayLike, optional
        Escape-noise soft-threshold width :math:`\delta` in mV; broadcastable
        and nonnegative. ``delta == 0`` reproduces deterministic thresholding.
        Default is ``0. * bu.mV``.
    tau_fac : ArrayLike, optional
        Facilitation time constant :math:`\tau_{\mathrm{fac}}` in ms;
        broadcastable and nonnegative. ``tau_fac == 0`` disables facilitation
        (:math:`P_{uu}=0`). Default is ``1000. * bu.ms``.
    tau_psc : ArrayLike, optional
        Tsodyks postsynaptic current time constant :math:`\tau_{\mathrm{psc}}`
        in ms; broadcastable and strictly positive. Used in state propagators.
        Default is ``2. * bu.ms``.
    tau_rec : ArrayLike, optional
        Resource recovery time constant :math:`\tau_{\mathrm{rec}}` in ms;
        broadcastable and strictly positive. Default is ``400. * bu.ms``.
    U : ArrayLike, optional
        Utilization increment factor :math:`U` (dimensionless); broadcastable
        and must lie in ``[0, 1]``. Represents the per-spike increase in
        release probability. Default is ``0.5``.
    x : ArrayLike, optional
        Initial readily-releasable resource fraction (dimensionless);
        broadcastable. Must satisfy ``x + y <= 1`` and ``x >= 0``. Default is
        ``0.0``.
    y : ArrayLike, optional
        Initial cleft/active fraction (dimensionless); broadcastable. Must
        satisfy ``x + y <= 1`` and ``y >= 0``. Default is ``0.0``.
    u : ArrayLike, optional
        Initial release probability (dimensionless); broadcastable and must lie
        in ``[0, 1]``. Default is ``0.0``.
    V_initializer : Callable, optional
        Initializer for membrane state ``V`` used by :meth:`init_state`.
        Default is ``braintools.init.Constant(-70. * bu.mV)``.
    spk_fun : Callable, optional
        Surrogate spike nonlinearity used by :meth:`get_spike`. Default is
        ``braintools.surrogate.ReluGrad()``.
    spk_reset : str, optional
        Reset policy inherited from :class:`~brainpy_state._base.Neuron`.
        ``'hard'`` matches NEST reset behavior. Default is ``'hard'``.
    ref_var : bool, optional
        If ``True``, allocates ``self.refractory`` (boolean) for external
        inspection of refractory state. Default is ``False``.
    name : str or None, optional
        Optional node name.

    Parameter Mapping
    -----------------
    .. list-table::
       :header-rows: 1
       :widths: 14 26 14 16 30

       * - Parameter
         - Type / shape / unit
         - Default
         - Math symbol
         - Semantics
       * - ``in_size``
         - :class:`~brainstate.typing.Size`; scalar/tuple
         - required
         - --
         - Defines neuron population shape ``self.varshape``.
       * - ``E_L``
         - ArrayLike, broadcastable (mV)
         - ``-70. * bu.mV``
         - :math:`E_L`
         - Resting membrane potential.
       * - ``C_m``
         - ArrayLike, broadcastable (pF), ``> 0``
         - ``250. * bu.pF``
         - :math:`C_m`
         - Membrane capacitance in voltage integration.
       * - ``tau_m``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``10. * bu.ms``
         - :math:`\tau_m`
         - Membrane leak time constant.
       * - ``t_ref``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``2. * bu.ms``
         - :math:`t_{\mathrm{ref}}`
         - Absolute refractory duration.
       * - ``V_th`` and ``V_reset``
         - ArrayLike, broadcastable (mV), with ``V_reset < V_th``
         - ``-55. * bu.mV``, ``-70. * bu.mV``
         - :math:`V_{th}`, :math:`V_{\mathrm{reset}}`
         - Threshold and post-spike reset voltages.
       * - ``tau_syn_ex`` and ``tau_syn_in``
         - ArrayLike, broadcastable (ms), each ``> 0``
         - ``2. * bu.ms``
         - :math:`\tau_{\mathrm{syn,ex}}`, :math:`\tau_{\mathrm{syn,in}}`
         - Exponential PSC decay constants.
       * - ``I_e``
         - ArrayLike, broadcastable (pA)
         - ``0. * bu.pA``
         - :math:`I_e`
         - Constant current injected every step.
       * - ``rho`` and ``delta``
         - ArrayLike, broadcastable; ``rho`` in ``1/s`` and ``delta`` in mV,
           both ``>= 0``
         - ``0.01 / bu.second``, ``0. * bu.mV``
         - :math:`\rho`, :math:`\delta`
         - Escape-noise hazard parameters.
       * - ``tau_fac``
         - ArrayLike, broadcastable (ms), ``>= 0``
         - ``1000. * bu.ms``
         - :math:`\tau_{\mathrm{fac}}`
         - Facilitation decay time constant; ``0`` disables.
       * - ``tau_psc``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``2. * bu.ms``
         - :math:`\tau_{\mathrm{psc}}`
         - Tsodyks PSC time constant.
       * - ``tau_rec``
         - ArrayLike, broadcastable (ms), ``> 0``
         - ``400. * bu.ms``
         - :math:`\tau_{\mathrm{rec}}`
         - Resource recovery time constant.
       * - ``U``
         - ArrayLike, broadcastable (dimensionless), ``∈ [0,1]``
         - ``0.5``
         - :math:`U`
         - Utilization increment per spike.
       * - ``x``
         - ArrayLike, broadcastable (dimensionless), ``x+y <= 1``
         - ``0.0``
         - :math:`x`
         - Initial readily-releasable fraction.
       * - ``y``
         - ArrayLike, broadcastable (dimensionless), ``x+y <= 1``
         - ``0.0``
         - :math:`y`
         - Initial cleft/active fraction.
       * - ``u``
         - ArrayLike, broadcastable (dimensionless), ``∈ [0,1]``
         - ``0.0``
         - :math:`u`
         - Initial release probability.
       * - ``V_initializer``
         - Callable
         - ``Constant(-70. * bu.mV)``
         - --
         - Initializer for membrane state ``V``.
       * - ``spk_fun``
         - Callable
         - ``ReluGrad()``
         - --
         - Surrogate function for output spikes.
       * - ``spk_reset``
         - ``str`` (typically ``'hard'``)
         - ``'hard'``
         - --
         - Reset behavior selection in base class.
       * - ``ref_var``
         - ``bool``
         - ``False``
         - --
         - Enables explicit boolean refractory state variable.
       * - ``name``
         - ``str`` or ``None``
         - ``None``
         - --
         - Optional instance name.

    Raises
    ------
    ValueError
        Raised at construction when any validated constraint is violated:
        ``V_reset >= V_th``, nonpositive ``C_m``/``tau_m``/synaptic time
        constants/Tsodyks time constants, negative ``tau_fac``/``t_ref``/``rho``/``delta``,
        ``U`` not in ``[0,1]``, ``u`` not in ``[0,1]``, or ``x + y > 1``.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> import brainunit as bu
       >>> from brainpy_state._nest.iaf_tum_2000 import iaf_tum_2000
       >>> brainstate.environ.set(dt=0.1 * bu.ms, t=0.0 * bu.ms)
       >>> neu = iaf_tum_2000(
       ...     in_size=(2,),
       ...     I_e=250. * bu.pA,
       ...     tau_fac=500. * bu.ms,
       ...     tau_rec=400. * bu.ms,
       ...     U=0.3
       ... )
       >>> neu.init_state()
       >>> out = neu.update(x=0. * bu.pA, x_filtered=0. * bu.pA)
       >>> out.shape
       (2,)

    Notes
    -----
    - Shares the exact exponential propagator implementation with
      :class:`iaf_psc_exp` (via :meth:`iaf_psc_exp._propagator_exp`).
    - The Tsodyks update order matches NEST ``iaf_tum_2000.cpp`` exactly to
      ensure identical dynamics in network simulations.
    - Receptor-1 connections require both presynaptic and postsynaptic neurons
      to be ``iaf_tum_2000`` models, enforced via runtime validation.
    - ``spike_offset`` can be recorded and monitored for debugging or analysis
      of dynamic synaptic efficacy.
    - The model is grid-based with one-step input buffering matching NEST's
      ring-buffer semantics.
    """

    __module__ = 'brainpy.state'

    RECEPTOR_TYPES = {
        'DEFAULT': 0,
        'TSODYKS': 1,
    }

    RECORDABLES = (
        'V_m',
        'I_syn_ex',
        'I_syn_in',
        'x',
        'y',
        'u',
        'spike_offset',
    )

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * bu.mV,
        C_m: ArrayLike = 250. * bu.pF,
        tau_m: ArrayLike = 10. * bu.ms,
        t_ref: ArrayLike = 2. * bu.ms,
        V_th: ArrayLike = -55. * bu.mV,
        V_reset: ArrayLike = -70. * bu.mV,
        tau_syn_ex: ArrayLike = 2. * bu.ms,
        tau_syn_in: ArrayLike = 2. * bu.ms,
        I_e: ArrayLike = 0. * bu.pA,
        rho: ArrayLike = 0.01 / bu.second,
        delta: ArrayLike = 0. * bu.mV,
        tau_fac: ArrayLike = 1000. * bu.ms,
        tau_psc: ArrayLike = 2. * bu.ms,
        tau_rec: ArrayLike = 400. * bu.ms,
        U: ArrayLike = 0.5,
        x: ArrayLike = 0.0,
        y: ArrayLike = 0.0,
        u: ArrayLike = 0.0,
        V_initializer: Callable = braintools.init.Constant(-70. * bu.mV),
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        spk_reset: str = 'hard',
        ref_var: bool = False,
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        self.E_L = braintools.init.param(E_L, self.varshape)
        self.C_m = braintools.init.param(C_m, self.varshape)
        self.tau_m = braintools.init.param(tau_m, self.varshape)
        self.t_ref = braintools.init.param(t_ref, self.varshape)
        self.V_th = braintools.init.param(V_th, self.varshape)
        self.V_reset = braintools.init.param(V_reset, self.varshape)
        self.tau_syn_ex = braintools.init.param(tau_syn_ex, self.varshape)
        self.tau_syn_in = braintools.init.param(tau_syn_in, self.varshape)
        self.I_e = braintools.init.param(I_e, self.varshape)
        self.rho = braintools.init.param(rho, self.varshape)
        self.delta = braintools.init.param(delta, self.varshape)

        self.tau_fac = braintools.init.param(tau_fac, self.varshape)
        self.tau_psc = braintools.init.param(tau_psc, self.varshape)
        self.tau_rec = braintools.init.param(tau_rec, self.varshape)
        self.U = braintools.init.param(U, self.varshape)
        self.x_init = braintools.init.param(x, self.varshape)
        self.y_init = braintools.init.param(y, self.varshape)
        self.u_init = braintools.init.param(u, self.varshape)

        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @property
    def receptor_types(self):
        r"""Return a dictionary of available receptor type labels.

        Returns
        -------
        dict
            Mapping from receptor name (str) to receptor ID (int):
            ``{'DEFAULT': 0, 'TSODYKS': 1}``.
        """
        return dict(self.RECEPTOR_TYPES)

    @property
    def recordables(self):
        r"""Return a list of state variable names available for recording.

        Returns
        -------
        list of str
            State variable names: ``['V_m', 'I_syn_ex', 'I_syn_in', 'x', 'y',
            'u', 'spike_offset']``. Note that the membrane potential is exposed
            as ``'V_m'`` (matching NEST convention) but stored internally as
            ``self.V``.
        """
        return list(self.RECORDABLES)

    @staticmethod
    def _to_numpy(x, unit=None):
        r"""Convert brainunit quantity or array to NumPy float64 array.

        Parameters
        ----------
        x : ArrayLike
            Input value, optionally with brainunit units.
        unit : brainunit quantity or None, optional
            Expected unit for conversion. If provided, divides ``x`` by ``unit``
            before conversion. If ``None``, converts ``x`` directly. Default is
            ``None``.

        Returns
        -------
        np.ndarray
            NumPy array with dtype ``float64``, containing the numeric values of
            ``x`` (divided by ``unit`` if specified).

        Notes
        -----
        Falls back to direct conversion if unit division fails (e.g., if ``x``
        is already dimensionless).
        """
        if unit is None:
            return np.asarray(bu.math.asarray(x), dtype=np.float64)
        try:
            return np.asarray(bu.math.asarray(x / unit), dtype=np.float64)
        except Exception:
            return np.asarray(bu.math.asarray(x), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        r"""Broadcast NumPy array to the specified shape.

        Parameters
        ----------
        x_np : np.ndarray
            Input array to broadcast.
        shape : tuple of int
            Target shape.

        Returns
        -------
        np.ndarray
            Broadcasted array with shape ``shape``, sharing memory with ``x_np``
            if possible.
        """
        return np.broadcast_to(x_np, shape)

    @classmethod
    def _normalize_spike_receptor(cls, receptor):
        r"""Normalize receptor label to canonical integer ID.

        Converts string labels like ``'DEFAULT'``, ``'TSODYKS'``, ``'R0'``,
        ``'R1'``, or numeric strings/integers to the standard receptor IDs (0 or 1).

        Parameters
        ----------
        receptor : str or int
            Receptor label. Valid string labels (case-insensitive):

            - Receptor 0: ``'DEFAULT'``, ``'R0'``, ``'RECEPTOR0'``, ``'0'``
            - Receptor 1: ``'TSODYKS'``, ``'R1'``, ``'RECEPTOR1'``, ``'1'``

            Integer values must be 0 or 1.

        Returns
        -------
        int
            Canonical receptor ID: 0 or 1.

        Raises
        ------
        ValueError
            If ``receptor`` is an unrecognized string label or an integer not in
            ``{0, 1}``.
        """
        if isinstance(receptor, str):
            key = receptor.strip().upper()
            if key in ('DEFAULT', 'R0', 'RECEPTOR0', '0'):
                return 0
            if key in ('TSODYKS', 'R1', 'RECEPTOR1', '1'):
                return 1
            if key.isdigit():
                receptor = int(key)
            else:
                raise ValueError(f'Unknown receptor label: {receptor}')

        receptor = int(receptor)
        if receptor not in (0, 1):
            raise ValueError(f'Receptor type must be 0 or 1, got {receptor}.')
        return receptor

    def _validate_parameters(self):
        r"""Validate model parameters at construction time.

        Checks all parameter constraints to ensure physical consistency and
        numerical stability. Raises ``ValueError`` with a descriptive message if
        any constraint is violated.

        Raises
        ------
        ValueError
            If any of the following constraints are violated:

            - ``V_reset >= V_th``: Reset must be below threshold.
            - ``C_m <= 0``: Capacitance must be strictly positive.
            - ``tau_m <= 0``: Membrane time constant must be strictly positive.
            - ``tau_syn_ex <= 0`` or ``tau_syn_in <= 0``: Synaptic time constants
              must be strictly positive.
            - ``tau_psc <= 0`` or ``tau_rec <= 0``: Tsodyks time constants must be
              strictly positive.
            - ``tau_fac < 0``: Facilitation time constant must be nonnegative.
            - ``t_ref < 0``: Refractory time must be nonnegative.
            - ``U < 0`` or ``U > 1``: Utilization factor must be in ``[0, 1]``.
            - ``rho < 0``: Firing intensity must be nonnegative.
            - ``delta < 0``: Threshold width must be nonnegative.
            - ``x + y > 1.0``: Resource fractions must sum to at most 1.
            - ``u < 0`` or ``u > 1``: Initial release probability must be in ``[0, 1]``.
        """
        if np.any(self._to_numpy(self.V_reset, bu.mV) >= self._to_numpy(self.V_th, bu.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if np.any(self._to_numpy(self.C_m, bu.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.tau_m, bu.ms) <= 0.0):
            raise ValueError('Membrane time constant must be strictly positive.')
        if np.any(self._to_numpy(self.tau_syn_ex, bu.ms) <= 0.0) or np.any(
            self._to_numpy(self.tau_syn_in, bu.ms) <= 0.0):
            raise ValueError('Synaptic time constants must be strictly positive.')
        if np.any(self._to_numpy(self.tau_psc, bu.ms) <= 0.0) or np.any(self._to_numpy(self.tau_rec, bu.ms) <= 0.0):
            raise ValueError('Tsodyks time constants tau_psc and tau_rec must be strictly positive.')
        if np.any(self._to_numpy(self.tau_fac, bu.ms) < 0.0):
            raise ValueError("'tau_fac' must be >= 0.")
        if np.any(self._to_numpy(self.t_ref, bu.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        if np.any(self._to_numpy(self.U, None) < 0.0) or np.any(self._to_numpy(self.U, None) > 1.0):
            raise ValueError("'U' must be in [0,1].")
        if np.any(self._to_numpy(self.rho, 1 / bu.second) < 0.0):
            raise ValueError('Stochastic firing intensity rho must not be negative.')
        if np.any(self._to_numpy(self.delta, bu.mV) < 0.0):
            raise ValueError('Threshold width delta must not be negative.')

        x0 = self._to_numpy(self.x_init, None)
        y0 = self._to_numpy(self.y_init, None)
        u0 = self._to_numpy(self.u_init, None)
        if np.any(x0 + y0 > 1.0):
            raise ValueError('x + y must be <= 1.0.')
        if np.any((u0 < 0.0) | (u0 > 1.0)):
            raise ValueError("'u' must be in [0,1].")

    def init_state(self, batch_size: int = None, **kwargs):
        r"""Initialize all neuron state variables.

        Creates and allocates state variables for membrane potential, synaptic
        currents, refractory counter, Tsodyks-Markram plasticity states, and
        buffered inputs. All states are allocated as ``brainstate.HiddenState``
        or ``brainstate.ShortTermState`` with shape ``self.varshape`` (or
        ``(batch_size,) + self.varshape`` if ``batch_size`` is provided).

        Parameters
        ----------
        batch_size : int or None, optional
            If provided, prepends an additional batch dimension to all states.
            Default is ``None`` (no batch dimension).
        **kwargs
            Reserved for future extensions; currently unused.

        Notes
        -----
        State variables created:

        - ``V`` : :class:`~brainstate.HiddenState` (mV)
            Membrane potential, initialized via ``self.V_initializer``.
        - ``i_syn_ex`` : :class:`~brainstate.ShortTermState` (pA)
            Excitatory synaptic current, initialized to zero.
        - ``i_syn_in`` : :class:`~brainstate.ShortTermState` (pA)
            Inhibitory synaptic current, initialized to zero.
        - ``i_0`` : :class:`~brainstate.ShortTermState` (pA)
            Buffered receptor-0 current input, initialized to zero.
        - ``i_1`` : :class:`~brainstate.ShortTermState` (pA)
            Buffered receptor-1 current input, initialized to zero.
        - ``refractory_step_count`` : :class:`~brainstate.ShortTermState` (int32)
            Remaining refractory steps, initialized to zero.
        - ``last_spike_time`` : :class:`~brainstate.ShortTermState` (ms)
            Time of last emitted spike, initialized to ``-1e7 * bu.ms`` (no prior spike).
        - ``x`` : :class:`~brainstate.ShortTermState` (dimensionless)
            Readily-releasable resource fraction, initialized to ``self.x_init``.
        - ``y`` : :class:`~brainstate.ShortTermState` (dimensionless)
            Cleft/active fraction, initialized to ``self.y_init``.
        - ``u`` : :class:`~brainstate.ShortTermState` (dimensionless)
            Release probability, initialized to ``self.u_init``.
        - ``spike_offset`` : :class:`~brainstate.ShortTermState` (dimensionless)
            Per-spike :math:`\Delta y` signal for receptor-1 coupling, initialized to zero.
        - ``refractory`` : :class:`~brainstate.ShortTermState` (bool), optional
            Boolean refractory flag, allocated only if ``ref_var=True``.
        """
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        state_shape = V.shape
        zeros = np.zeros(state_shape, dtype=np.float64)
        ref_steps = braintools.init.param(braintools.init.Constant(0), self.varshape, batch_size)
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * bu.ms), self.varshape, batch_size)

        x0 = np.broadcast_to(self._to_numpy(self.x_init, None), state_shape).copy()
        y0 = np.broadcast_to(self._to_numpy(self.y_init, None), state_shape).copy()
        u0 = np.broadcast_to(self._to_numpy(self.u_init, None), state_shape).copy()

        self.V = brainstate.HiddenState(V)
        self.i_syn_ex = brainstate.ShortTermState(zeros * bu.pA)
        self.i_syn_in = brainstate.ShortTermState(zeros * bu.pA)
        self.i_0 = brainstate.ShortTermState(zeros * bu.pA)
        self.i_1 = brainstate.ShortTermState(zeros * bu.pA)
        self.refractory_step_count = brainstate.ShortTermState(bu.math.asarray(ref_steps, dtype=jnp.int32))
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        self.x = brainstate.ShortTermState(x0)
        self.y = brainstate.ShortTermState(y0)
        self.u = brainstate.ShortTermState(u0)
        self.spike_offset = brainstate.ShortTermState(zeros.copy())

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(bu.math.asarray(ref_steps > 0, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        r"""Compute surrogate spike output given membrane potential.

        Applies the surrogate spike function (``self.spk_fun``) to a normalized
        voltage that ranges from 0 (at reset) to 1 (at threshold). This enables
        differentiable spike computation for gradient-based learning.

        Parameters
        ----------
        V : ArrayLike or None, optional
            Membrane potential in mV; broadcastable to ``self.varshape``. If
            ``None``, uses ``self.V.value``. Default is ``None``.

        Returns
        -------
        out : dict
            Surrogate spike activation, shape matching the input ``V`` (or
            ``self.V.value``). The output is typically in ``[0, 1]`` for
            sub-threshold voltages and close to 1 for supra-threshold voltages,
            depending on the surrogate function used.

        Notes
        -----
        Voltage normalization:

        .. math::

           v_{\mathrm{scaled}} = \frac{V - V_{th}}{V_{th} - V_{\mathrm{reset}}}.

        The surrogate function ``self.spk_fun`` (default
        ``braintools.surrogate.ReluGrad()``) is then applied to
        ``v_scaled``, providing a differentiable approximation of the Heaviside
        step function.
        """
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _refractory_counts(self):
        r"""Compute refractory period duration in integer simulation steps.

        Converts the continuous-time refractory duration ``self.t_ref`` into the
        number of discrete simulation steps via ``ceil(t_ref / dt)``.

        Returns
        -------
        jnp.ndarray
            Integer array (dtype ``int32``) with shape matching ``self.varshape``,
            containing the number of refractory steps for each neuron.
        """
        dt = brainstate.environ.get_dt()
        return bu.math.asarray(bu.math.ceil(self.t_ref / dt), dtype=jnp.int32)

    def _parse_spike_events(self, spike_events: Iterable, state_shape):
        r"""Parse external spike events into excitatory and inhibitory weights.

        Processes each event descriptor in ``spike_events``, validates receptor
        types and sender models, computes effective weights including
        multiplicity and offset factors, and routes them by sign to excitatory
        or inhibitory channels.

        Parameters
        ----------
        spike_events : Iterable or None
            Collection of event descriptors (see :meth:`update` for format).
        state_shape : tuple of int
            Target shape for broadcasting weight arrays.

        Returns
        -------
        w_ex : np.ndarray
            Excitatory weights in pA, shape ``state_shape``, dtype ``float64``.
        w_in : np.ndarray
            Inhibitory weights in pA, shape ``state_shape``, dtype ``float64``.

        Raises
        ------
        ValueError
            If receptor-1 events have ``sender_model != "iaf_tum_2000"``, or if
            event format is invalid.

        Notes
        -----
        Effective weights are computed as:

        - Receptor 0: ``weight * multiplicity``
        - Receptor 1: ``weight * multiplicity * offset``

        Positive weights route to ``w_ex``, non-positive to ``w_in``.
        """
        w_ex = np.zeros(state_shape, dtype=np.float64)
        w_in = np.zeros(state_shape, dtype=np.float64)

        if spike_events is None:
            return w_ex, w_in

        for ev in spike_events:
            sender_model = 'iaf_tum_2000'
            multiplicity = 1.0
            offset = 1.0

            if isinstance(ev, dict):
                receptor = ev.get('receptor_type', ev.get('receptor', 0))
                weight = ev.get('weight', 0.0 * bu.pA)
                offset = ev.get('offset', 1.0)
                multiplicity = ev.get('multiplicity', 1.0)
                sender_model = ev.get('sender_model', 'iaf_tum_2000')
            else:
                if len(ev) == 2:
                    receptor, weight = ev
                elif len(ev) == 3:
                    receptor, weight, offset = ev
                elif len(ev) == 4:
                    receptor, weight, offset, multiplicity = ev
                elif len(ev) == 5:
                    receptor, weight, offset, multiplicity, sender_model = ev
                else:
                    raise ValueError('Spike event tuples must have length 2, 3, 4, or 5.')

            receptor_id = self._normalize_spike_receptor(receptor)
            s = np.broadcast_to(self._to_numpy(weight, bu.pA), state_shape)
            s = s * np.broadcast_to(self._to_numpy(multiplicity, None), state_shape)

            if receptor_id == 1:
                if sender_model != 'iaf_tum_2000':
                    raise ValueError(
                        'For receptor_type 1 in iaf_tum_2000, pre-synaptic neuron must also be of type iaf_tum_2000.'
                    )
                s = s * np.broadcast_to(self._to_numpy(offset, None), state_shape)

            w_ex += np.where(s > 0.0, s, 0.0)
            w_in += np.where(s > 0.0, 0.0, s)

        return w_ex, w_in

    def _parse_registered_spike_inputs(self, state_shape):
        r"""Parse registered delta inputs into excitatory and inhibitory weights.

        Processes inputs previously registered via :meth:`add_delta_input`
        (inherited from :class:`~brainpy_state._base.Dynamics`), extracts
        receptor labels from keys, and routes by sign to excitatory or
        inhibitory channels.

        Parameters
        ----------
        state_shape : tuple of int
            Target shape for broadcasting weight arrays.

        Returns
        -------
        w_ex : np.ndarray
            Excitatory weights in pA, shape ``state_shape``, dtype ``float64``.
        w_in : np.ndarray
            Inhibitory weights in pA, shape ``state_shape``, dtype ``float64``.

        Notes
        -----
        Keys in ``self.delta_inputs`` may optionally include a receptor label
        prefix (e.g., ``'TSODYKS // proj_0'``). If present, the label is
        extracted and normalized via :meth:`_normalize_spike_receptor`;
        otherwise defaults to receptor 0.

        Values are either callables (invoked and then removed) or direct
        ArrayLike values.
        """
        w_ex = np.zeros(state_shape, dtype=np.float64)
        w_in = np.zeros(state_shape, dtype=np.float64)

        if self.delta_inputs is None:
            return w_ex, w_in

        for key in tuple(self.delta_inputs.keys()):
            val = self.delta_inputs[key]
            if callable(val):
                val = val()
            else:
                self.delta_inputs.pop(key)

            label = None
            if ' // ' in key:
                label, _ = key.split(' // ', maxsplit=1)
            receptor = 0 if label is None else self._normalize_spike_receptor(label)

            s = np.broadcast_to(self._to_numpy(val, bu.pA), state_shape)
            if receptor == 0:
                w_ex += np.where(s > 0.0, s, 0.0)
                w_in += np.where(s > 0.0, 0.0, s)
            else:
                w_ex += np.where(s > 0.0, s, 0.0)
                w_in += np.where(s > 0.0, 0.0, s)

        return w_ex, w_in

    def update(self, x=0. * bu.pA, x_filtered=0. * bu.pA, spike_events=None):
        r"""Advance the neuron state by one simulation step.

        Performs a complete integration step following NEST ``iaf_tum_2000``
        update order: membrane propagation (if not refractory), synaptic current
        decay, filtered-current injection, spike input addition, threshold test,
        Tsodyks state update on spike emission, and input buffering for the next
        step.

        Parameters
        ----------
        x : ArrayLike, optional
            Current input in pA for receptor-0 (standard current port). Scalar
            or array broadcastable to ``self.varshape``. The value is buffered
            and applied in the next step (NEST ring-buffer semantics). Default
            is ``0. * bu.pA``.
        x_filtered : ArrayLike, optional
            Current input in pA for receptor-1. It is buffered to ``self.i_1``
            and injected through excitatory exponential filtering at the next
            update step via ``(1 - P11_ex) * i_1``. Scalar or array
            broadcastable to ``self.varshape``. Default is ``0. * bu.pA``.
        spike_events : Iterable or None, optional
            Collection of spike event descriptors for direct spike input. Each
            event can be:

            - ``(receptor_type, weight)``
            - ``(receptor_type, weight, offset)``
            - ``(receptor_type, weight, offset, multiplicity)``
            - ``(receptor_type, weight, offset, multiplicity, sender_model)``
            - ``dict`` with keys ``receptor_type``/``receptor`` (int or str),
              ``weight`` (ArrayLike in pA), ``offset`` (float, default ``1.0``),
              ``multiplicity`` (float, default ``1.0``), ``sender_model`` (str,
              default ``"iaf_tum_2000"``).

            **Receptor types:**

            - ``0`` (or ``'DEFAULT'``): regular spike input, effective weight
              is ``weight * multiplicity``.
            - ``1`` (or ``'TSODYKS'``): Tsodyks-coupled input, effective weight
              is ``weight * multiplicity * offset``, where ``offset`` is
              typically the ``spike_offset`` from the presynaptic neuron.

            For receptor ``1``, ``sender_model`` must be ``"iaf_tum_2000"``;
            otherwise a ``ValueError`` is raised.

            Positive effective weights route to excitatory channel, non-positive
            to inhibitory channel. Default is ``None`` (no events).

        Returns
        -------
        out : jax.Array
            Surrogate spike output returned by :meth:`get_spike`. The output is
            elementwise over the neuron state shape (and batch axis, if
            initialized). For emitted spikes, the voltage argument to
            :meth:`get_spike` is nudged above threshold by ``1e-12`` mV to
            preserve positive spike activation under hard reset.

        Raises
        ------
        ValueError
            If provided inputs cannot be broadcast to the internal state shape,
            or if receptor-1 events have ``sender_model != "iaf_tum_2000"``.

        Notes
        -----
        **Update order (following NEST ``iaf_tum_2000.cpp``):**

        1. **Membrane propagation**: If not refractory, update
           :math:`V_{\mathrm{rel}}` using exact exponential propagators (same as
           :class:`iaf_psc_exp`).
        2. **Synaptic decay**: Exponentially decay ``i_syn_ex`` and ``i_syn_in``.
        3. **Filtered current injection**: Add ``(1 - exp(-h/tau_syn_ex)) * i_1``
           to ``i_syn_ex``.
        4. **Spike input addition**: Add arriving spike inputs (from
           ``spike_events`` and registered delta inputs) to ``i_syn_ex`` and
           ``i_syn_in`` by sign.
        5. **Threshold test**: Determine spike condition (deterministic or
           escape-noise), assign refractory counter, and reset voltage.
        6. **Tsodyks update**: On emitted spike, update ``(u, x, y)`` states in
           NEST order, compute :math:`\Delta y`, and set ``spike_offset``.
        7. **Buffer inputs**: Store ``x`` and ``x_filtered`` for next step.

        **Tsodyks state update on spike:**

        When a spike is emitted, inter-spike interval ``h_ts = t_spike - t_last``
        is computed (with ``t_last = 0`` if ``last_spike_time < 0``). Propagators
        are:

        .. math::

           P_{uu} = \begin{cases}
           0, & \tau_{\mathrm{fac}}=0 \\
           e^{-h_{ts}/\tau_{\mathrm{fac}}}, & \text{otherwise}
           \end{cases}, \quad
           P_{yy} = e^{-h_{ts}/\tau_{\mathrm{psc}}}, \quad
           P_{zz} = e^{-h_{ts}/\tau_{\mathrm{rec}}} - 1,

        .. math::

           P_{xy} = \frac{P_{zz}\tau_{\mathrm{rec}} - (P_{yy}-1)\tau_{\mathrm{psc}}}{\tau_{\mathrm{psc}}-\tau_{\mathrm{rec}}}.

        Then states update as:

        .. math::

           u \leftarrow u P_{uu}, \quad
           x \leftarrow x + P_{xy}y - P_{zz}(1-x-y), \quad
           y \leftarrow y P_{yy}, \\
           u \leftarrow u + U(1-u), \quad
           \Delta y = u x, \quad
           x \leftarrow x - \Delta y, \quad
           y \leftarrow y + \Delta y.

        ``spike_offset`` is set to :math:`\Delta y` on spike, zero otherwise.

        **Performance:** Per-step computational cost is
        :math:`O(\prod \mathrm{varshape})` with vectorized NumPy operations in
        ``float64`` for coefficient computation and state updates.
        """
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(bu.math.asarray(dt_q / bu.ms))
        t_ms = float(bu.math.asarray(t / bu.ms))

        state_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, bu.mV), state_shape)
        V_rel = self._broadcast_to_state(self._to_numpy(self.V.value, bu.mV), state_shape) - E_L
        C_m = self._broadcast_to_state(self._to_numpy(self.C_m, bu.pF), state_shape)
        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, bu.ms), state_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, bu.ms), state_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, bu.ms), state_shape)
        I_e = self._broadcast_to_state(self._to_numpy(self.I_e, bu.pA), state_shape)
        theta = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, bu.mV), state_shape)
        V_reset_rel = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, bu.mV), state_shape)
        rho = self._broadcast_to_state(self._to_numpy(self.rho, 1 / bu.second), state_shape)
        delta = self._broadcast_to_state(self._to_numpy(self.delta, bu.mV), state_shape)

        tau_fac = self._broadcast_to_state(self._to_numpy(self.tau_fac, bu.ms), state_shape)
        tau_psc = self._broadcast_to_state(self._to_numpy(self.tau_psc, bu.ms), state_shape)
        tau_rec = self._broadcast_to_state(self._to_numpy(self.tau_rec, bu.ms), state_shape)
        U = self._broadcast_to_state(self._to_numpy(self.U, None), state_shape)

        i_0 = self._broadcast_to_state(self._to_numpy(self.i_0.value, bu.pA), state_shape)
        i_1 = self._broadcast_to_state(self._to_numpy(self.i_1.value, bu.pA), state_shape)
        i_syn_ex = self._broadcast_to_state(self._to_numpy(self.i_syn_ex.value, bu.pA), state_shape)
        i_syn_in = self._broadcast_to_state(self._to_numpy(self.i_syn_in.value, bu.pA), state_shape)
        r = self._broadcast_to_state(
            np.asarray(bu.math.asarray(self.refractory_step_count.value), dtype=np.int32),
            state_shape,
        )

        x_state = self._broadcast_to_state(self._to_numpy(self.x.value, None), state_shape)
        y_state = self._broadcast_to_state(self._to_numpy(self.y.value, None), state_shape)
        u_state = self._broadcast_to_state(self._to_numpy(self.u.value, None), state_shape)
        last_spike_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, bu.ms), state_shape)

        P11_ex = np.exp(-h / tau_ex)
        P11_in = np.exp(-h / tau_in)
        P22 = np.exp(-h / tau_m)
        P21_ex = iaf_psc_exp._propagator_exp(tau_ex, tau_m, C_m, h)
        P21_in = iaf_psc_exp._propagator_exp(tau_in, tau_m, C_m, h)
        P20 = tau_m / C_m * (1.0 - P22)

        ev_ex, ev_in = self._parse_spike_events(spike_events, state_shape)
        reg_ex, reg_in = self._parse_registered_spike_inputs(state_shape)
        w_ex = ev_ex + reg_ex
        w_in = ev_in + reg_in

        i_0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), bu.pA),
                                            state_shape)
        i_1_next = self._broadcast_to_state(self._to_numpy(x_filtered, bu.pA), state_shape)

        not_refractory = r == 0
        V_candidate = V_rel * P22 + i_syn_ex * P21_ex + i_syn_in * P21_in + (I_e + i_0) * P20
        V_rel = np.where(not_refractory, V_candidate, V_rel)
        r = np.where(not_refractory, r, r - 1)

        i_syn_ex = i_syn_ex * P11_ex
        i_syn_in = i_syn_in * P11_in
        i_syn_ex = i_syn_ex + (1.0 - P11_ex) * i_1
        i_syn_ex = i_syn_ex + w_ex
        i_syn_in = i_syn_in + w_in

        deterministic = delta < 1e-10
        det_spike = V_rel >= theta
        phi = rho * np.exp((V_rel - theta) / np.where(deterministic, 1.0, delta))
        stoch_spike = np.random.random(size=state_shape) < phi * h * 1e-3
        spike_cond = np.where(deterministic, det_spike, stoch_spike)

        refr_counts = self._broadcast_to_state(
            np.asarray(bu.math.asarray(self._refractory_counts()), dtype=np.int32),
            state_shape,
        )
        r = np.where(spike_cond, refr_counts, r)
        V_before_reset = V_rel
        V_rel = np.where(spike_cond, V_reset_rel, V_rel)

        t_last = np.where(last_spike_prev < 0.0, 0.0, last_spike_prev)
        t_spike = t_ms + h
        h_tsodyks = t_spike - t_last

        tau_fac_safe = np.where(tau_fac == 0.0, 1.0, tau_fac)
        Puu = np.where(tau_fac == 0.0, 0.0, np.exp(-h_tsodyks / tau_fac_safe))
        Pyy = np.exp(-h_tsodyks / tau_psc)
        Pzz = np.expm1(-h_tsodyks / tau_rec)
        Pxy = (Pzz * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)

        z_state = 1.0 - x_state - y_state
        u_prop = u_state * Puu
        x_prop = x_state + Pxy * y_state - Pzz * z_state
        y_prop = y_state * Pyy

        u_jump = u_prop + U * (1.0 - u_prop)
        delta_y_tsp = u_jump * x_prop
        x_new = x_prop - delta_y_tsp
        y_new = y_prop + delta_y_tsp

        x_state = np.where(spike_cond, x_new, x_state)
        y_state = np.where(spike_cond, y_new, y_state)
        u_state = np.where(spike_cond, u_jump, u_state)
        spike_offset = np.where(spike_cond, delta_y_tsp, 0.0)
        last_spike_next = np.where(spike_cond, t_spike, last_spike_prev)

        self.V.value = (V_rel + E_L) * bu.mV
        self.i_syn_ex.value = i_syn_ex * bu.pA
        self.i_syn_in.value = i_syn_in * bu.pA
        self.i_0.value = i_0_next * bu.pA
        self.i_1.value = i_1_next * bu.pA
        self.refractory_step_count.value = jnp.asarray(r, dtype=jnp.int32)
        self.last_spike_time.value = jax.lax.stop_gradient(last_spike_next * bu.ms)

        self.x.value = x_state
        self.y.value = y_state
        self.u.value = u_state
        self.spike_offset.value = spike_offset

        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.refractory_step_count.value > 0)

        V_out = np.where(spike_cond, theta + E_L + 1e-12, V_before_reset + E_L)
        return self.get_spike(V_out * bu.mV)
