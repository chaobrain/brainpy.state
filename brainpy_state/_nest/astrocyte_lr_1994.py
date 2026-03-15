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

import brainstate
import jax.numpy as jnp
import saiunit as u
from brainstate.typing import Size
from brainstate.util import DotDict

from ._base import NESTNeuron
from ._utils import is_tracer, AdaptiveRungeKuttaStep

__all__ = [
    'astrocyte_lr_1994',
]


class astrocyte_lr_1994(NESTNeuron):
    r"""NEST-compatible astrocyte model with IP3-mediated calcium dynamics.

    Implements the Li & Rinzel (1994) [1]_ astrocytic calcium oscillation model with
    synaptic input and SIC output mechanisms from Nadkarni & Jung (2003) [3]_.
    Models calcium release from intracellular stores (ER) via IP3 receptors,
    calcium-induced calcium inhibition, SERCA pump-mediated sequestration, and
    passive leak currents. Generates slow inward currents (SIC) when cytosolic
    calcium exceeds threshold, enabling astrocyte-neuron signaling.

    Parameters
    ----------
    in_size : int or tuple of int
        Population shape specifying number of astrocytes. Can be scalar (1D) or
        tuple for multi-dimensional astrocyte arrays.
    Ca_tot : float, default: 2.0
        Total free calcium concentration across cytosol and ER, in µM. Used for
        calcium conservation: :math:`[\mathrm{Ca}]_{\mathrm{ER}} = (C_{\mathrm{tot}} - [\mathrm{Ca}]) / \rho`.
        Must be positive.
    IP3_0 : float, default: 0.16
        Baseline (resting) IP3 concentration in µM. IP3 decays exponentially to this
        value with time constant ``tau_IP3``. Must be non-negative.
    Kd_IP3_1 : float, default: 0.13
        First IP3 dissociation constant for IP3R activation (µM). Appears in the
        activation gate :math:`m_\infty` and inactivation rate :math:`\alpha_h`.
        Must be positive.
    Kd_IP3_2 : float, default: 0.9434
        Second IP3 dissociation constant for IP3R inactivation (µM). Modulates the
        IP3 dependence of :math:`\alpha_h`. Must be positive.
    Kd_act : float, default: 0.08234
        Calcium dissociation constant for IP3R activation (µM). Controls the calcium
        activation gate :math:`n_\infty`. Must be positive.
    Kd_inh : float, default: 1.049
        Calcium dissociation constant for IP3R inactivation (µM). Scales the
        inactivation rate :math:`\alpha_h`. Must be non-negative.
    Km_SERCA : float, default: 0.1
        SERCA pump half-activation constant (µM). Calcium concentration at which the
        pump operates at half-maximal rate. Must be positive.
    SIC_scale : float, default: 1.0
        Dimensionless scaling factor for SIC output amplitude. Multiplies the logarithmic
        SIC response. Must be positive.
    SIC_th : float, default: 0.19669
        Calcium threshold for SIC generation (µM). No SIC is produced when
        :math:`[\mathrm{Ca}] < \mathrm{SIC_{th}}`. Must be non-negative.
    delta_IP3 : float, default: 0.0002
        IP3 increase per unit synaptic weight (µM). Each spike with weight :math:`w`
        increases IP3 by :math:`\Delta_{\mathrm{IP3}} \times w`. Must be non-negative.
    k_IP3R : float, default: 0.0002
        IP3R inactivation rate constant (1/(µM·ms)). Scales both activation and
        inactivation rates of the IP3R inactivation gate. Must be non-negative.
    rate_IP3R : float, default: 0.006
        Maximal calcium release rate through IP3R (1/ms). Scales :math:`J_{\mathrm{channel}}`.
        Must be non-negative.
    rate_L : float, default: 0.00011
        Passive ER leak rate constant (1/ms). Scales :math:`J_{\mathrm{leak}}`.
        Must be non-negative.
    rate_SERCA : float, default: 0.0009
        Maximal SERCA pump rate (µM/ms). Maximum flux when :math:`[\mathrm{Ca}] \to \infty`.
        Must be non-negative.
    ratio_ER_cyt : float, default: 0.185
        Volume ratio of ER to cytosol (dimensionless). Denoted :math:`\rho`, scales
        ER-cytosol calcium fluxes. Must be positive.
    tau_IP3 : float, default: 7142.0
        IP3 exponential decay time constant (ms). IP3 relaxes to ``IP3_0`` with this
        time scale. Must be positive.
    gsl_error_tol : float, default: 1e-3
        Local error tolerance for the RKF45 adaptive integrator. Smaller values increase
        accuracy but require more substeps. Typical range: 1e-6 to 1e-2.
    IP3_initializer : float, optional
        Initial IP3 concentration (µM). If None, defaults to ``IP3_0``.
    Ca_initializer : float, default: 0.073
        Initial cytosolic calcium concentration (µM).
    h_IP3R_initializer : float, default: 0.793
        Initial fraction of non-inactivated IP3 receptors (dimensionless, 0–1).
    name : str, optional
        Instance name for identification.

    Parameter Mapping
    -----------------

    The following table maps constructor parameters to their mathematical symbols
    and NEST equivalents:

    ================= ============= ======================================== ======================================================
    **Parameter**     **Default**   **Math Symbol**                          **Description**
    ================= ============= ======================================== ======================================================
    ``in_size``       (required)    —                                        Population shape (number of astrocytes)
    ``Ca_tot``        2.0 µM        :math:`C_{\mathrm{tot}}`                 Total free Ca concentration (cytosol + ER basis)
    ``IP3_0``         0.16 µM       :math:`[\mathrm{IP3}]_0`                 Baseline IP3 concentration
    ``Kd_IP3_1``      0.13 µM       :math:`K_{d,\mathrm{IP3,1}}`             First IP3R dissociation constant for IP3
    ``Kd_IP3_2``      0.9434 µM     :math:`K_{d,\mathrm{IP3,2}}`             Second IP3R dissociation constant for IP3
    ``Kd_act``        0.08234 µM    :math:`K_{d,\mathrm{act}}`               IP3R Ca dissociation constant (activation)
    ``Kd_inh``        1.049 µM      :math:`K_{d,\mathrm{inh}}`               IP3R Ca dissociation constant (inhibition)
    ``Km_SERCA``      0.1 µM        :math:`K_{m,\mathrm{SERCA}}`             SERCA pump half-activation constant
    ``SIC_scale``     1.0           —                                        SIC output scaling factor (dimensionless)
    ``SIC_th``        0.19669 µM    :math:`\mathrm{SIC_{th}}`                Ca threshold for SIC generation
    ``delta_IP3``     0.0002 µM     :math:`\Delta_{\mathrm{IP3}}`            IP3 increase per unit synaptic weight
    ``k_IP3R``        0.0002        :math:`k_{\mathrm{IP3R}}`                IP3R Ca inhibition rate constant (1/(µM·ms))
    ``rate_IP3R``     0.006         :math:`r_{\mathrm{IP3R}}`                Max Ca release rate via IP3R (1/ms)
    ``rate_L``        0.00011       :math:`r_L`                              ER Ca leak rate constant (1/ms)
    ``rate_SERCA``    0.0009        :math:`r_{\mathrm{SERCA}}`               Max SERCA pump rate (µM/ms)
    ``ratio_ER_cyt``  0.185         :math:`\rho`                             ER-to-cytosol volume ratio
    ``tau_IP3``       7142.0 ms     :math:`\tau_{\mathrm{IP3}}`              IP3 exponential decay time constant
    ``gsl_error_tol`` 1e-3          —                                        RKF45 local error tolerance
    ================= ============= ======================================== ======================================================

    Raises
    ------
    ValueError
        If any of the following parameter constraints are violated:

        - ``Ca_tot`` must be positive
        - ``IP3_0`` must be non-negative
        - ``Kd_act`` must be positive
        - ``Kd_inh`` must be non-negative
        - ``Kd_IP3_1`` must be positive
        - ``Kd_IP3_2`` must be positive
        - ``Km_SERCA`` must be positive
        - ``ratio_ER_cyt`` must be positive
        - ``delta_IP3`` must be non-negative
        - ``k_IP3R`` must be non-negative
        - ``SIC_scale`` must be positive
        - ``SIC_th`` must be non-negative
        - ``rate_L`` must be non-negative
        - ``rate_IP3R`` must be non-negative
        - ``rate_SERCA`` must be non-negative
        - ``tau_IP3`` must be positive

    See Also
    --------
    aeif_cond_alpha_astro : AdEx neuron with astrocyte SIC input

    Notes
    -----
    **1. Model Overview**

    ``astrocyte_lr_1994`` simulates astrocytic calcium dynamics driven by IP3-mediated
    calcium-induced calcium release (CICR) from the endoplasmic reticulum (ER). The
    model integrates three coupled differential equations describing IP3 concentration,
    cytosolic calcium concentration, and IP3 receptor inactivation state. Synaptic
    input increases IP3, triggering calcium release; elevated calcium generates slow
    inward currents (SIC) that can modulate postsynaptic neurons.

    **2. Mathematical Model**

    The model has three state variables:

    - :math:`[\mathrm{IP3}]` : IP3 concentration in astrocytic cytosol (µM)
    - :math:`[\mathrm{Ca^{2+}}]` : Calcium concentration in astrocytic cytosol (µM)
    - :math:`h_{\mathrm{IP3R}}` : Fraction of non-inactivated IP3 receptors (0–1)

    **2.1. Calcium Dynamics**

    Cytosolic calcium evolves according to:

    .. math::

       \frac{d[\mathrm{Ca^{2+}}]}{dt} =
         J_{\mathrm{channel}} - J_{\mathrm{pump}} + J_{\mathrm{leak}}
         + J_{\mathrm{noise}}

    where:

    .. math::

       J_{\mathrm{channel}} = \rho \cdot r_{\mathrm{IP3R}} \cdot m_\infty^3
         \cdot n_\infty^3 \cdot h_{\mathrm{IP3R}}^3
         \cdot ([\mathrm{Ca^{2+}}]_{\mathrm{ER}} - [\mathrm{Ca^{2+}}])

    .. math::

       J_{\mathrm{pump}} = r_{\mathrm{SERCA}}
         \frac{[\mathrm{Ca^{2+}}]^2}{K_{m,\mathrm{SERCA}}^2 + [\mathrm{Ca^{2+}}]^2}

    .. math::

       J_{\mathrm{leak}} = \rho \cdot r_L
         \cdot ([\mathrm{Ca^{2+}}]_{\mathrm{ER}} - [\mathrm{Ca^{2+}}])

    Steady-state gating variables:

    .. math::

       m_\infty = \frac{[\mathrm{IP3}]}{[\mathrm{IP3}] + K_{d,\mathrm{IP3,1}}}

    .. math::

       n_\infty = \frac{[\mathrm{Ca^{2+}}]}{[\mathrm{Ca^{2+}}] + K_{d,\mathrm{act}}}

    ER calcium (from conservation):

    .. math::

       [\mathrm{Ca^{2+}}]_{\mathrm{ER}} =
         \frac{C_{\mathrm{tot}} - [\mathrm{Ca^{2+}}]}{\rho}

    **2.2. IP3 Dynamics**

    IP3 decays exponentially to baseline and is increased by synaptic input:

    .. math::

       \frac{d[\mathrm{IP3}]}{dt} =
         \frac{[\mathrm{IP3}]_0 - [\mathrm{IP3}]}{\tau_{\mathrm{IP3}}}
         + \Delta_{\mathrm{IP3}} \cdot J_{\mathrm{syn}}(t)

    Each incoming spike with weight :math:`w` instantaneously increases IP3 by
    :math:`\Delta_{\mathrm{IP3}} \times w`.

    **2.3. IP3 Receptor Inactivation Dynamics**

    The IP3R inactivation gate evolves as:

    .. math::

       \frac{dh_{\mathrm{IP3R}}}{dt} =
         \alpha_h (1 - h_{\mathrm{IP3R}}) - \beta_h \, h_{\mathrm{IP3R}}

    with:

    .. math::

       \alpha_h = k_{\mathrm{IP3R}} \cdot K_{d,\mathrm{inh}}
         \cdot \frac{[\mathrm{IP3}] + K_{d,\mathrm{IP3,1}}}
                    {[\mathrm{IP3}] + K_{d,\mathrm{IP3,2}}}

    .. math::

       \beta_h = k_{\mathrm{IP3R}} \cdot [\mathrm{Ca^{2+}}]

    **2.4. Slow Inward Current (SIC) Output**

    When cytosolic calcium exceeds ``SIC_th``, a slow inward current is generated:

    .. math::

       y = ([\mathrm{Ca^{2+}}] - \mathrm{SIC_{th}}) \times 1000 \text{ (µM → nM)}

    .. math::

       I_{\mathrm{SIC}} =
         \begin{cases}
           \mathrm{SIC_{scale}} \cdot \ln y & \text{if } y > 1 \\
           0 & \text{otherwise}
         \end{cases}

    **3. Numerical Integration**

    This implementation uses the **Runge–Kutta–Fehlberg (RKF45)** adaptive step-size
    method to match NEST's GSL-based solver. RKF45 computes both 4th-order and
    5th-order solutions at each substep, estimates local error, and adjusts step size
    accordingly. The algorithm:

    1. Starts with step size equal to the simulation time step ``dt``
    2. Computes 6 intermediate function evaluations (k1–k6)
    3. Compares 4th-order (:math:`y_4`) and 5th-order (:math:`y_5`) solutions
    4. If error :math:`\|y_5 - y_4\| \leq` ``gsl_error_tol``, accepts step and
       increases step size; otherwise reduces step size and retries
    5. Ensures step size remains in :math:`[h_{\min}, h_{\max}]` where
       :math:`h_{\min} = 10^{-8}` ms

    After integration, cytosolic calcium is clamped to :math:`[0, C_{\mathrm{tot}}]`
    to enforce conservation and prevent numerical artifacts.

    **4. State Variables**

    The model maintains four state arrays:

    ========== ============ ============================================================
    Variable   Type         Description
    ========== ============ ============================================================
    ``IP3``    HiddenState  IP3 concentration (µM)
    ``Ca``     HiddenState  Cytosolic Ca²⁺ concentration (µM)
    ``h_IP3R`` HiddenState  Non-inactivated IP3R fraction (0–1)
    ``SIC``    ShortTerm    SIC output (pA, computed from Ca each step)
    ========== ============ ============================================================

    **5. Computational Considerations**

    - **Adaptive integration**: Each astrocyte requires multiple RKF45 substeps per
      simulation time step. Computational cost scales with ``1 / gsl_error_tol``.
    - **Per-element loop**: Integration is performed element-wise in NumPy (not
      vectorized), matching NEST semantics but limiting performance on large arrays.
    - **Input delay**: External current ``J_ext`` is applied with a one-step delay
      to match NEST's event-driven semantics.
    - **Spike input**: Incoming spike weights instantaneously increase IP3 *after*
      the RKF45 integration step, consistent with NEST's discrete event handling.

    **6. Usage with Neurons**

    Astrocytes typically receive excitatory synaptic input from nearby neurons and
    provide SIC feedback to postsynaptic neurons. Example workflow:

    1. Connect neurons to astrocytes via projections with ``delta_IP3``-weighted synapses
    2. Accumulate spike weights at each time step
    3. Call ``astrocyte.update(spike_weights=w)`` to integrate dynamics
    4. Use ``astrocyte.SIC`` as additional input current to postsynaptic neurons

    See ``aeif_cond_alpha_astro`` for an integrated example of neuron-astrocyte coupling.

    References
    ----------
    .. [1] Li, Y. X., & Rinzel, J. (1994). Equations for InsP3
           receptor-mediated [Ca2+]i oscillations derived from a detailed
           kinetic model: a Hodgkin-Huxley like formalism.
           *Journal of Theoretical Biology*, 166(4), 461–473.
           https://doi.org/10.1006/jtbi.1994.1041

    .. [2] De Young, G. W., & Keizer, J. (1992). A single-pool inositol
           1,4,5-trisphosphate-receptor-based model for agonist-stimulated
           oscillations in Ca2+ concentration.
           *Proceedings of the National Academy of Sciences*, 89(20), 9895–9899.
           https://doi.org/10.1073/pnas.89.20.9895

    .. [3] Nadkarni, S., & Jung, P. (2003). Spontaneous oscillations of
           dressed neurons: a new mechanism for epilepsy?.
           *Physical Review Letters*, 91(26), 268101.
           https://doi.org/10.1103/PhysRevLett.91.268101

    Examples
    --------
    Basic astrocyte simulation:

    .. code-block:: python

       >>> import brainstate as bst
       >>> import saiunit as u
       >>> import brainpy_state as bps
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     astro = bps.astrocyte_lr_1994(in_size=10)
       ...     astro.init_state()
       ...     # Simulate 100 ms with constant IP3 input
       ...     for _ in range(1000):
       ...         sic = astro.update(spike_weights=0.5)

    Monitor calcium oscillations:

    .. code-block:: python

       >>> import numpy as np
       >>> import matplotlib.pyplot as plt
       >>> with bst.environ.context(dt=0.1 * u.ms):
       ...     astro = bps.astrocyte_lr_1994(in_size=1)
       ...     astro.init_state()
       ...     ca_trace = []
       ...     for _ in range(5000):  # 500 ms
       ...         astro.update(spike_weights=0.3)
       ...         ca_trace.append(float(astro.Ca.value[0]))
       >>> plt.plot(ca_trace)
       >>> plt.xlabel('Time step (0.1 ms)')
       >>> plt.ylabel('[Ca²⁺] (µM)')
       >>> plt.title('Astrocyte Calcium Oscillations')
    """

    __module__ = 'brainpy.state'

    RECORDABLES = (
        'IP3',
        'Ca',
        'h_IP3R',
        'SIC',
    )

    _MIN_H = 1e-8 * u.ms  # ms – minimum integration step
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        # Parameters (Nadkarni & Jung 2003 defaults, matching NEST)
        Ca_tot: float = 2.0,  # µM
        IP3_0: float = 0.16,  # µM
        Kd_IP3_1: float = 0.13,  # µM
        Kd_IP3_2: float = 0.9434,  # µM
        Kd_act: float = 0.08234,  # µM
        Kd_inh: float = 1.049,  # µM
        Km_SERCA: float = 0.1,  # µM
        SIC_scale: float = 1.0,  # dimensionless
        SIC_th: float = 0.19669,  # µM
        delta_IP3: float = 0.0002,  # µM
        k_IP3R: float = 0.0002,  # 1/(µM·ms)
        rate_IP3R: float = 0.006,  # 1/ms
        rate_L: float = 0.00011,  # 1/ms
        rate_SERCA: float = 0.0009,  # µM/ms
        ratio_ER_cyt: float = 0.185,  # dimensionless
        tau_IP3: float = 7142.0,  # ms
        gsl_error_tol: float = 1e-3,
        # State initializers
        IP3_initializer: float = None,
        Ca_initializer: float = 0.073,  # µM
        h_IP3R_initializer: float = 0.793,
        name: str = None,
    ):
        super().__init__(in_size, name=name)

        # Store parameters (unitless, matching NEST internal representation)
        self.Ca_tot = Ca_tot
        self.IP3_0 = IP3_0
        self.Kd_IP3_1 = Kd_IP3_1
        self.Kd_IP3_2 = Kd_IP3_2
        self.Kd_act = Kd_act
        self.Kd_inh = Kd_inh
        self.Km_SERCA = Km_SERCA
        self.SIC_scale = SIC_scale
        self.SIC_th = SIC_th
        self.delta_IP3 = delta_IP3
        self.k_IP3R = k_IP3R
        self.rate_IP3R = rate_IP3R
        self.rate_L = rate_L
        self.rate_SERCA = rate_SERCA
        self.ratio_ER_cyt = ratio_ER_cyt
        self.tau_IP3 = tau_IP3
        self.gsl_error_tol = gsl_error_tol

        # Initial state values
        self._IP3_init = IP3_0 if IP3_initializer is None else IP3_initializer
        self._Ca_init = Ca_initializer
        self._h_IP3R_init = h_IP3R_initializer

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

    def _validate_parameters(self):
        r"""Validate model parameters against NEST constraints.

        Raises
        ------
        ValueError
            If parameter inequalities or positivity constraints are violated.
        """
        # Skip validation when parameters are JAX tracers (e.g. during jit).
        if any(is_tracer(v) for v in (self.Ca_tot, self.IP3_0, self.Kd_act, self.tau_IP3)):
            return

        if self.Ca_tot <= 0:
            raise ValueError("Ca_tot must be positive.")
        if self.IP3_0 < 0:
            raise ValueError("IP3_0 must be non-negative.")
        if self.Kd_act <= 0:
            raise ValueError("Kd_act must be positive.")
        if self.Kd_inh < 0:
            raise ValueError("Kd_inh must be non-negative.")
        if self.Kd_IP3_1 <= 0:
            raise ValueError("Kd_IP3_1 must be positive.")
        if self.Kd_IP3_2 <= 0:
            raise ValueError("Kd_IP3_2 must be positive.")
        if self.Km_SERCA <= 0:
            raise ValueError("Km_SERCA must be positive.")
        if self.ratio_ER_cyt <= 0:
            raise ValueError("ratio_ER_cyt must be positive.")
        if self.delta_IP3 < 0:
            raise ValueError("delta_IP3 must be non-negative.")
        if self.k_IP3R < 0:
            raise ValueError("k_IP3R must be non-negative.")
        if self.SIC_scale <= 0:
            raise ValueError("SIC_scale must be positive.")
        if self.SIC_th < 0:
            raise ValueError("SIC_th must be non-negative.")
        if self.rate_L < 0:
            raise ValueError("rate_L must be non-negative.")
        if self.rate_IP3R < 0:
            raise ValueError("rate_IP3R must be non-negative.")
        if self.rate_SERCA < 0:
            raise ValueError("rate_SERCA must be non-negative.")
        if self.tau_IP3 <= 0:
            raise ValueError("tau_IP3 must be positive.")

    @property
    def recordables(self):
        r"""Return list of state variable names that can be recorded.

        Returns
        -------
        list of str
            ['IP3', 'Ca', 'h_IP3R', 'SIC'] — all dynamic state variables and output.
        """
        return list(self.RECORDABLES)

    def init_state(self, **kwargs):
        r"""Initialize astrocyte state variables.

        Creates state arrays for IP3 concentration, cytosolic calcium concentration,
        IP3R inactivation gate, SIC output, external current buffer, and the
        adaptive integration step size. All states are initialized to
        constructor-specified values.

        Parameters
        ----------
        **kwargs : dict, optional
            Unused compatibility parameters accepted by the base-state API.

        Notes
        -----
        - State variables are stored as JAX arrays via ``brainstate.HiddenState`` and
          ``brainstate.ShortTermState``.
        - Initial values are taken from constructor parameters: ``IP3_initializer``
          (defaults to ``IP3_0``), ``Ca_initializer``, and ``h_IP3R_initializer``.
        - SIC output and external current buffer are initialized to zero.
        """
        dftype = brainstate.environ.dftype()
        dt = brainstate.environ.get_dt()

        ip3 = jnp.full(self.varshape, self._IP3_init, dtype=dftype)
        ca = jnp.full(self.varshape, self._Ca_init, dtype=dftype)
        h = jnp.full(self.varshape, self._h_IP3R_init, dtype=dftype)

        self.IP3 = brainstate.HiddenState(ip3)
        self.Ca = brainstate.HiddenState(ca)
        self.h_IP3R = brainstate.HiddenState(h)
        self.SIC = brainstate.ShortTermState(jnp.zeros(self.varshape, dtype=dftype))
        self.J_noise = brainstate.ShortTermState(jnp.zeros(self.varshape, dtype=dftype))
        self.integration_step = brainstate.ShortTermState(
            jnp.full(self.varshape, u.get_mantissa(dt), dtype=dftype) * u.ms
        )

    def _vector_field(self, state, extra):
        r"""Vectorized RHS for the IP3-calcium ODE system.

        Evaluates the right-hand side of the three coupled differential equations
        governing IP3 concentration, cytosolic calcium concentration, and IP3R
        inactivation state. All quantities use NEST internal units (µM, ms).

        Parameters
        ----------
        state : DotDict
            Keys: IP3, Ca, h_IP3R — ODE state variables.
        extra : DotDict
            Keys: j_noise — mutable auxiliary data carried through the integrator.

        Returns
        -------
        DotDict with same keys as ``state``, containing time derivatives.

        Notes
        -----
        - Calcium is clamped to ``[0, Ca_tot]`` before computing the RHS to ensure
          ER calcium remains positive.
        - Synaptic IP3 input is handled separately via instantaneous jumps after
          integration, not within this ODE function.
        - All steady-state gates (:math:`m_\infty`, :math:`n_\infty`) are computed
          on-the-fly without storing intermediate states.
        """
        ip3 = state.IP3
        ca = jnp.clip(state.Ca, 0.0, self.Ca_tot)
        h_ip3r = state.h_IP3R
        j_noise = extra.j_noise

        alpha_h = (self.k_IP3R * self.Kd_inh
                   * (ip3 + self.Kd_IP3_1) / (ip3 + self.Kd_IP3_2))
        beta_h = self.k_IP3R * ca

        J_pump = (self.rate_SERCA * ca ** 2
                  / (self.Km_SERCA ** 2 + ca ** 2))

        m_inf = ip3 / (ip3 + self.Kd_IP3_1)
        n_inf = ca / (ca + self.Kd_act)
        calc_ER = (self.Ca_tot - ca) / self.ratio_ER_cyt

        J_leak = self.ratio_ER_cyt * self.rate_L * (calc_ER - ca)
        J_channel = (self.ratio_ER_cyt * self.rate_IP3R
                     * m_inf ** 3 * n_inf ** 3 * h_ip3r ** 3
                     * (calc_ER - ca))

        dip3 = (self.IP3_0 - ip3) / self.tau_IP3
        dca = J_channel - J_pump + J_leak + j_noise
        dh = alpha_h * (1.0 - h_ip3r) - beta_h * h_ip3r

        # The integrator multiplies derivatives by h (which carries ms units).
        # State variables are unitless, so derivatives must be in 1/ms for
        # dimensional consistency: state + h[ms] * deriv[1/ms] = unitless.
        inv_ms = 1.0 / u.ms
        return DotDict(IP3=dip3 * inv_ms, Ca=dca * inv_ms, h_IP3R=dh * inv_ms)

    def _event_fn(self, state, extra, accept):
        r"""Post-substep calcium clamping to enforce conservation.

        After each accepted RKF45 substep, clamps cytosolic calcium to
        :math:`[0, C_{\mathrm{tot}}]` to prevent numerical drift beyond
        physical bounds.

        Parameters
        ----------
        state : DotDict
            Keys: IP3, Ca, h_IP3R — ODE state variables.
        extra : DotDict
            Keys: j_noise — auxiliary data.
        accept : array, bool
            Mask of elements whose RK substep was accepted.

        Returns
        -------
        (new_state, extra) DotDicts with clamped calcium values.
        """
        clamped_ca = jnp.where(
            accept,
            jnp.clip(state.Ca, 0.0, self.Ca_tot),
            state.Ca,
        )
        new_state = DotDict({**state, 'Ca': clamped_ca})
        return new_state, extra

    def update(self, spike_weights=0.0, J_ext=0.0):
        r"""Advance astrocyte state by one simulation time step using RKF45 integration.

        Integrates the IP3-calcium ODE system over the interval :math:`[t, t+\Delta t]`
        using adaptive Runge–Kutta–Fehlberg 4(5) substeps, applies incoming spike
        input to IP3, clamps calcium to conserve total calcium, computes SIC output,
        and stores the result. External current ``J_ext`` is delayed by one step to
        match NEST's event-driven semantics.

        Parameters
        ----------
        spike_weights : float or array_like, default: 0.0
            Total accumulated synaptic weight from incoming spikes at this time step.
            Shape must be broadcastable to ``self.varshape``. Each spike with weight
            :math:`w` increases IP3 instantaneously by :math:`\Delta_{\mathrm{IP3}} \times w`.
            Applied *after* ODE integration.
        J_ext : float or array_like, default: 0.0
            External calcium flux input in µM/ms. Shape must be broadcastable to
            ``self.varshape``. Corresponds to NEST's ``CurrentEvent`` input channel.
            Applied *in the next time step* (one-step delay).

        Returns
        -------
        sic : jax.numpy.ndarray
            Slow inward current output array with shape ``self.varshape``. Computed from
            current cytosolic calcium concentration via logarithmic threshold function.
            Can be used as additional input current to postsynaptic neurons.

        Notes
        -----
        - **Integration**: Uses RKF45 adaptive stepping with local error tolerance
          ``gsl_error_tol``. Substeps are adjusted dynamically; typical substep count
          ranges from 1 to 100 per simulation time step depending on stiffness.
        - **Calcium clamping**: After integration, cytosolic calcium is clamped to
          :math:`[0, C_{\mathrm{tot}}]` to enforce conservation and prevent numerical
          overflow in the exponential SERCA term.
        - **Event ordering**: The update sequence is:

          1. Integrate ODEs from :math:`t` to :math:`t + \Delta t` using ``J_noise``
             from the *previous* step
          2. Clamp calcium to valid range
          3. Apply spike input: :math:`[\mathrm{IP3}]_{\mathrm{new}} = [\mathrm{IP3}] + \Delta_{\mathrm{IP3}} \times \sum w`
          4. Compute SIC output from updated calcium
          5. Store ``J_ext`` for use in the *next* update call

        - **Performance**: Integration is fully vectorized via the JAX-based
          ``AdaptiveRungeKuttaStep`` integrator, enabling efficient parallel
          computation across the entire astrocyte population.
        """
        dftype = brainstate.environ.dftype()

        # Read state variables.
        ip3 = self.IP3.value
        ca = self.Ca.value
        h_ip3r = self.h_IP3R.value
        j_noise = self.J_noise.value
        h = self.integration_step.value

        # Build ODE state and extra data for the integrator.
        ode_state = DotDict(IP3=ip3, Ca=ca, h_IP3R=h_ip3r)
        extra = DotDict(j_noise=j_noise)

        # Adaptive RKF45 integration via generic integrator.
        ode_state, h, extra = self.integrator(state=ode_state, h=h, extra=extra)
        ip3, ca, h_ip3r = ode_state.IP3, ode_state.Ca, ode_state.h_IP3R

        # Final clamp of calcium to [0, Ca_tot] (matches NEST).
        ca = jnp.clip(ca, 0.0, self.Ca_tot)

        # Apply spike input: IP3 += delta_IP3 * spike_weight.
        sw = jnp.broadcast_to(jnp.asarray(spike_weights, dtype=dftype), self.varshape)
        ip3 = ip3 + self.delta_IP3 * sw

        # Compute SIC output.
        calc_thr = (ca - self.SIC_th) * 1000.0  # µM -> nM
        sic_out = jnp.where(
            calc_thr > 1.0,
            jnp.log(jnp.maximum(calc_thr, 1.0)) * self.SIC_scale,
            0.0,
        )
        sic_out = jnp.asarray(sic_out, dtype=dftype)

        # Write back state.
        self.IP3.value = ip3
        self.Ca.value = ca
        self.h_IP3R.value = h_ip3r
        self.SIC.value = sic_out
        self.integration_step.value = h

        # Store new external current for next step (one-step delay, NEST semantics).
        j_ext = jnp.broadcast_to(jnp.asarray(J_ext, dtype=dftype), self.varshape)
        self.J_noise.value = j_ext

        return sic_out
