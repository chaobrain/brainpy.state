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

import math
from typing import Callable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Dynamics

__all__ = [
    'astrocyte_lr_1994',
]


class astrocyte_lr_1994(Dynamics):
    r"""Summary
    -------
    
    NEST-compatible ``astrocyte_lr_1994`` astrocyte model.
    
    Parameters
    ----------
    
    =============== ============= ========================================= ===================================================================
    **Parameter**   **Default**   **Math equivalent**                       **Description**
    =============== ============= ========================================= ===================================================================
    ``in_size``     (required)                                              Population shape (number of astrocytes)
    ``Ca_tot``      2.0 µM        :math:`C_{\mathrm{tot}}`                  Total free astrocytic Ca concentration (cytosolic vol basis)
    ``IP3_0``       0.16 µM       :math:`[\mathrm{IP3}]_0`                  Baseline IP3 concentration
    ``Kd_IP3_1``    0.13 µM       :math:`K_{d,\mathrm{IP3,1}}`              First IP3R dissociation constant for IP3
    ``Kd_IP3_2``    0.9434 µM     :math:`K_{d,\mathrm{IP3,2}}`              Second IP3R dissociation constant for IP3
    ``Kd_act``      0.08234 µM    :math:`K_{d,\mathrm{act}}`                IP3R Ca dissociation constant (activation)
    ``Kd_inh``      1.049 µM      :math:`K_{d,\mathrm{inh}}`                IP3R Ca dissociation constant (inhibition)
    ``Km_SERCA``    0.1 µM        :math:`K_{m,\mathrm{SERCA}}`              SERCA pump half-activation constant
    ``SIC_scale``   1.0                                                     SIC output scaling factor (dimensionless)
    ``SIC_th``      0.19669 µM    :math:`\mathrm{SIC_{th}}`                 Ca threshold for SIC generation
    ``delta_IP3``   0.0002 µM     :math:`\Delta_{\mathrm{IP3}}`             IP3 increase per unit synaptic weight
    ``k_IP3R``      0.0002        :math:`k_{\mathrm{IP3R}}`                 IP3R Ca inhibition rate constant (1/(µM·ms))
    ``rate_IP3R``   0.006         :math:`r_{\mathrm{IP3R}}`                 Max Ca release rate via IP3R (1/ms)
    ``rate_L``      0.00011       :math:`r_L`                               ER Ca leak rate constant (1/ms)
    ``rate_SERCA``  0.0009        :math:`r_{\mathrm{SERCA}}`                Max SERCA pump rate (µM/ms)
    ``ratio_ER_cyt``0.185         :math:`\rho`                              ER-to-cytosol volume ratio
    ``tau_IP3``     7142.0 ms     :math:`\tau_{\mathrm{IP3}}`               IP3 exponential decay time constant
    ``gsl_error_tol``1e-3         (solver tolerance)                        RKF45 local error tolerance
    =============== ============= ========================================= ===================================================================
    
    Raises
    ------
    
    ValueError
        Raised when one of the following conditions is violated:
        - Ca_tot must be positive.
        - IP3_0 must be non-negative.
        - Kd_act must be positive.
        - Kd_inh must be non-negative.
        - Kd_IP3_1 must be positive.
        - Kd_IP3_2 must be positive.
        - Km_SERCA must be positive.
        - ratio_ER_cyt must be positive.
        - delta_IP3 must be non-negative.
        - k_IP3R must be non-negative.
        - SIC_scale must be positive.
        - SIC_th must be non-negative.
        - rate_L must be non-negative.
        - rate_IP3R must be non-negative.
        - rate_SERCA must be non-negative.
        - tau_IP3 must be positive.
    
    See Also
    --------
    
    aeif_cond_alpha_astro : AdEx neuron with astrocyte SIC input
    
    Notes
    -----
    
    Short description
    .................
    
    An astrocyte model based on Li & Rinzel (1994) with input/output from
    Nadkarni & Jung (2003).
    
    Description
    ...........
    
    ``astrocyte_lr_1994`` is a model of astrocytic calcium dynamics. The model
    was first proposed by Li & Rinzel (1994) [1]_ and is based on earlier work of
    De Young & Keizer (1992) [2]_. The input and output of the model are
    implemented according to Nadkarni & Jung (2003) [3]_.
    
    The model has three dynamic state variables:
    
    - :math:`[\mathrm{IP3}]` : inositol 1,4,5-trisphosphate concentration in
      the astrocytic cytosol (µM)
    - :math:`[\mathrm{Ca^{2+}}]` : calcium concentration in the astrocytic
      cytosol (µM)
    - :math:`h_{\mathrm{IP3R}}` : fraction of IP3 receptors on the astrocytic
      endoplasmic reticulum (ER) that are not yet inactivated by calcium
      (dimensionless, 0–1)
    
    Calcium dynamics
    ................
    
    The calcium concentration in the cytosol evolves according to:
    
    .. math::
    
       \frac{d[\mathrm{Ca^{2+}}]}{dt} =
         J_{\mathrm{channel}} - J_{\mathrm{pump}} + J_{\mathrm{leak}}
         + J_{\mathrm{noise}}
    
    where the individual flux terms are:
    
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
    
    with steady-state gating variables:
    
    .. math::
    
       m_\infty = \frac{[\mathrm{IP3}]}{[\mathrm{IP3}] + K_{d,\mathrm{IP3,1}}}
    
    .. math::
    
       n_\infty = \frac{[\mathrm{Ca^{2+}}]}{[\mathrm{Ca^{2+}}] + K_{d,\mathrm{act}}}
    
    and ER calcium concentration (conservation):
    
    .. math::
    
       [\mathrm{Ca^{2+}}]_{\mathrm{ER}} =
         \frac{C_{\mathrm{tot}} - [\mathrm{Ca^{2+}}]}{\rho}
    
    where :math:`\rho` is ``ratio_ER_cyt``.
    
    IP3 dynamics
    ............
    
    .. math::
    
       \frac{d[\mathrm{IP3}]}{dt} =
         \frac{[\mathrm{IP3}]_0 - [\mathrm{IP3}]}{\tau_{\mathrm{IP3}}}
         + \Delta_{\mathrm{IP3}} \cdot J_{\mathrm{syn}}(t)
    
    Each incoming spike instantaneously increases IP3 by
    :math:`\Delta_{\mathrm{IP3}} \times w` where :math:`w` is the synaptic
    weight.
    
    IP3R inactivation dynamics
    ..........................
    
    .. math::
    
       \frac{dh_{\mathrm{IP3R}}}{dt} =
         \alpha_h (1 - h_{\mathrm{IP3R}}) - \beta_h \, h_{\mathrm{IP3R}}
    
    with
    
    .. math::
    
       \alpha_h = k_{\mathrm{IP3R}} \cdot K_{d,\mathrm{inh}}
         \cdot \frac{[\mathrm{IP3}] + K_{d,\mathrm{IP3,1}}}
                    {[\mathrm{IP3}] + K_{d,\mathrm{IP3,2}}}
    
    .. math::
    
       \beta_h = k_{\mathrm{IP3R}} \cdot [\mathrm{Ca^{2+}}]
    
    SIC output
    ..........
    
    When the cytosolic calcium exceeds a threshold :math:`\mathrm{SIC_{th}}`,
    a slow inward current (SIC) is generated:
    
    .. math::
    
       y = ([\mathrm{Ca^{2+}}] - \mathrm{SIC_{th}}) / \mathrm{nM}
    
    .. math::
    
       I_{\mathrm{SIC}} =
         \mathrm{SIC_{scale}} \cdot H(\ln y) \cdot \ln y
    
    where :math:`H(\cdot)` is the Heaviside step function and the conversion
    factor 1000 converts µM to nM.
    
    Integration method
    ..................
    
    This implementation uses the Runge–Kutta–Fehlberg (RKF45) adaptive
    step-size method, matching NEST's GSL-based RKF45 solver.
    
    Dynamic state variables
    .......................
    
    ========== ======= ============================================================
    ``IP3``    µM      IP3 concentration in the astrocytic cytosol
    ``Ca``     µM      Ca²⁺ concentration in the astrocytic cytosol
    ``h_IP3R`` (0–1)   Fraction of non-inactivated IP3 receptors on the ER
    ========== ======= ============================================================
    
    References
    ----------
    
    .. [1] Li, Y. X., & Rinzel, J. (1994). Equations for InsP3
           receptor-mediated [Ca2+]i oscillations derived from a detailed
           kinetic model: a Hodgkin-Huxley like formalism. Journal of
           theoretical Biology, 166(4), 461–473.
           DOI: https://doi.org/10.1006/jtbi.1994.1041
    
    .. [2] De Young, G. W., & Keizer, J. (1992). A single-pool inositol
           1,4,5-trisphosphate-receptor-based model for agonist-stimulated
           oscillations in Ca2+ concentration. Proceedings of the National
           Academy of Sciences, 89(20), 9895–9899.
           DOI: https://doi.org/10.1073/pnas.89.20.9895
    
    .. [3] Nadkarni, S., & Jung, P. (2003). Spontaneous oscillations of
           dressed neurons: a new mechanism for epilepsy?. Physical review
           letters, 91(26), 268101.
           DOI: https://doi.org/10.1103/PhysRevLett.91.268101
    
    Examples
    --------
    
    .. code-block:: python
    
       >>> import brainpy
       >>> model = brainpy.state.astrocyte_lr_1994(in_size=1)
       >>> model.init_state()
    """

    __module__ = 'brainpy.state'

    RECORDABLES = (
        'IP3',
        'Ca',
        'h_IP3R',
        'SIC',
    )

    _MIN_H = 1e-8   # ms – minimum integration step
    _MAX_ITERS = 100000

    def __init__(
        self,
        in_size: Size,
        # Parameters (Nadkarni & Jung 2003 defaults, matching NEST)
        Ca_tot: float = 2.0,          # µM
        IP3_0: float = 0.16,          # µM
        Kd_IP3_1: float = 0.13,       # µM
        Kd_IP3_2: float = 0.9434,     # µM
        Kd_act: float = 0.08234,      # µM
        Kd_inh: float = 1.049,        # µM
        Km_SERCA: float = 0.1,        # µM
        SIC_scale: float = 1.0,       # dimensionless
        SIC_th: float = 0.19669,      # µM
        delta_IP3: float = 0.0002,    # µM
        k_IP3R: float = 0.0002,       # 1/(µM·ms)
        rate_IP3R: float = 0.006,     # 1/ms
        rate_L: float = 0.00011,      # 1/ms
        rate_SERCA: float = 0.0009,   # µM/ms
        ratio_ER_cyt: float = 0.185,  # dimensionless
        tau_IP3: float = 7142.0,      # ms
        gsl_error_tol: float = 1e-3,
        # State initializers
        IP3_initializer: float = None,
        Ca_initializer: float = 0.073,    # µM
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

    def _validate_parameters(self):
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
        return list(self.RECORDABLES)

    def init_state(self, batch_size: int = None, **kwargs):
        shape = self.varshape
        if batch_size is not None:
            shape = (batch_size,) + shape

        ip3 = np.full(shape, self._IP3_init, dtype=np.float64)
        ca = np.full(shape, self._Ca_init, dtype=np.float64)
        h = np.full(shape, self._h_IP3R_init, dtype=np.float64)

        self.IP3 = brainstate.HiddenState(jnp.asarray(ip3))
        self.Ca = brainstate.HiddenState(jnp.asarray(ca))
        self.h_IP3R = brainstate.HiddenState(jnp.asarray(h))
        self.SIC = brainstate.ShortTermState(jnp.zeros(shape, dtype=jnp.float64))
        self.J_noise = brainstate.ShortTermState(jnp.zeros(shape, dtype=jnp.float64))

    def reset_state(self, batch_size: int = None, **kwargs):
        shape = self.varshape
        if batch_size is not None:
            shape = (batch_size,) + shape

        self.IP3.value = jnp.full(shape, self._IP3_init, dtype=jnp.float64)
        self.Ca.value = jnp.full(shape, self._Ca_init, dtype=jnp.float64)
        self.h_IP3R.value = jnp.full(shape, self._h_IP3R_init, dtype=jnp.float64)
        self.SIC.value = jnp.zeros(shape, dtype=jnp.float64)
        self.J_noise.value = jnp.zeros(shape, dtype=jnp.float64)

    def _dynamics(self, ip3, ca, h_ip3r, J_noise):
        """Compute the RHS of the ODE system.

        All quantities are in NEST internal units (µM, ms).

        Parameters
        ----------
        ip3 : float
            IP3 concentration (µM).
        ca : float
            Cytosolic calcium concentration (µM), already clamped.
        h_ip3r : float
            Fraction of non-inactivated IP3R.
        J_noise : float
            External current input (µM/ms).

        Returns
        -------
        dip3, dca, dh : float
            Time derivatives of the three state variables.
        """
        calc = max(0.0, min(ca, self.Ca_tot))

        alpha_h = (self.k_IP3R * self.Kd_inh
                   * (ip3 + self.Kd_IP3_1) / (ip3 + self.Kd_IP3_2))
        beta_h = self.k_IP3R * calc

        J_pump = (self.rate_SERCA * calc ** 2
                  / (self.Km_SERCA ** 2 + calc ** 2))

        m_inf = ip3 / (ip3 + self.Kd_IP3_1)
        n_inf = calc / (calc + self.Kd_act)
        calc_ER = (self.Ca_tot - calc) / self.ratio_ER_cyt

        J_leak = self.ratio_ER_cyt * self.rate_L * (calc_ER - calc)
        J_channel = (self.ratio_ER_cyt * self.rate_IP3R
                     * m_inf ** 3 * n_inf ** 3 * h_ip3r ** 3
                     * (calc_ER - calc))

        dip3 = (self.IP3_0 - ip3) / self.tau_IP3
        dca = J_channel - J_pump + J_leak + J_noise
        dh = alpha_h * (1.0 - h_ip3r) - beta_h * h_ip3r

        return dip3, dca, dh

    @staticmethod
    def _compute_sic(ca, SIC_th, SIC_scale):
        """Compute SIC output from calcium concentration.

        Matches NEST: ``calc_thr = (Ca - SIC_th) * 1000.0``
        ``sic = log(calc_thr) * SIC_scale  if calc_thr > 1  else 0``
        """
        calc_thr = (ca - SIC_th) * 1000.0  # µM -> nM
        if calc_thr > 1.0:
            return math.log(calc_thr) * SIC_scale
        return 0.0

    def update(self, spike_weights=0.0, J_ext=0.0):
        """Advance the astrocyte state by one simulation time step.

        Parameters
        ----------
        spike_weights : float or array
            Total excitatory synaptic weight arriving at this step.
            Each unit of weight increases IP3 by ``delta_IP3 * weight``.
        J_ext : float or array
            External current input added directly to the calcium flux
            (µM/ms units). Corresponds to NEST's ``CurrentEvent`` input.

        Returns
        -------
        sic : array
            The SIC output value for this time step.
        """
        dt_q = brainstate.environ.get_dt()
        dt = float(np.asarray(u.math.asarray(dt_q / u.ms), dtype=np.float64))

        v_shape = self.IP3.value.shape

        ip3 = np.broadcast_to(
            np.asarray(self.IP3.value, dtype=np.float64), v_shape
        ).copy()
        ca = np.broadcast_to(
            np.asarray(self.Ca.value, dtype=np.float64), v_shape
        ).copy()
        h = np.broadcast_to(
            np.asarray(self.h_IP3R.value, dtype=np.float64), v_shape
        ).copy()
        j_noise_arr = np.broadcast_to(
            np.asarray(self.J_noise.value, dtype=np.float64), v_shape
        ).copy()
        sic_out = np.zeros(v_shape, dtype=np.float64)

        # Convert spike_weights and J_ext to arrays
        sw = np.broadcast_to(
            np.asarray(spike_weights, dtype=np.float64), v_shape
        )
        j_ext = np.broadcast_to(
            np.asarray(J_ext, dtype=np.float64), v_shape
        )

        atol = self.gsl_error_tol

        for idx in np.ndindex(v_shape):
            y = np.array([ip3[idx], ca[idx], h[idx]], dtype=np.float64)
            j_noise_local = j_noise_arr[idx]

            # RKF45 adaptive integration over (0, dt]
            t_local = 0.0
            h_step = dt  # initial integration step
            iters = 0

            while t_local < dt and iters < self._MAX_ITERS:
                iters += 1
                h_step = max(self._MIN_H, min(h_step, dt - t_local))

                def f(y_):
                    d = self._dynamics(y_[0], y_[1], y_[2], j_noise_local)
                    return np.array(d, dtype=np.float64)

                k1 = f(y)
                k2 = f(y + h_step * (1.0 / 4.0) * k1)
                k3 = f(y + h_step * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2))
                k4 = f(y + h_step * (1932.0 / 2197.0 * k1
                                     - 7200.0 / 2197.0 * k2
                                     + 7296.0 / 2197.0 * k3))
                k5 = f(y + h_step * (439.0 / 216.0 * k1
                                     - 8.0 * k2
                                     + 3680.0 / 513.0 * k3
                                     - 845.0 / 4104.0 * k4))
                k6 = f(y + h_step * (-8.0 / 27.0 * k1
                                     + 2.0 * k2
                                     - 3544.0 / 2565.0 * k3
                                     + 1859.0 / 4104.0 * k4
                                     - 11.0 / 40.0 * k5))

                # 4th-order solution
                y4 = y + h_step * (25.0 / 216.0 * k1
                                   + 1408.0 / 2565.0 * k3
                                   + 2197.0 / 4104.0 * k4
                                   - 1.0 / 5.0 * k5)
                # 5th-order solution
                y5 = y + h_step * (16.0 / 135.0 * k1
                                   + 6656.0 / 12825.0 * k3
                                   + 28561.0 / 56430.0 * k4
                                   - 9.0 / 50.0 * k5
                                   + 2.0 / 55.0 * k6)

                err = float(np.max(np.abs(y5 - y4)))

                if err <= atol or h_step <= self._MIN_H:
                    y = y5
                    t_local += h_step
                    fac = (5.0 if err == 0.0
                           else min(5.0, max(0.2, 0.9 * (atol / err) ** 0.2)))
                    h_step = max(self._MIN_H, h_step * fac)
                else:
                    fac = min(1.0, max(0.2, 0.9 * (atol / err) ** 0.25))
                    h_step = max(self._MIN_H, h_step * fac)

            # Clamp calcium to [0, Ca_tot] (matches NEST)
            y[1] = max(0.0, min(y[1], self.Ca_tot))

            # Apply spike input: IP3 += delta_IP3 * spike_weight
            y[0] += self.delta_IP3 * sw[idx]

            # Compute SIC output
            sic_out[idx] = self._compute_sic(y[1], self.SIC_th, self.SIC_scale)

            # Store updated state
            ip3[idx] = y[0]
            ca[idx] = y[1]
            h[idx] = y[2]

        # Write back state
        self.IP3.value = jnp.asarray(ip3)
        self.Ca.value = jnp.asarray(ca)
        self.h_IP3R.value = jnp.asarray(h)
        self.SIC.value = jnp.asarray(sic_out)
        # Store new external current for next step (one-step delay, NEST semantics)
        self.J_noise.value = jnp.asarray(
            np.broadcast_to(np.asarray(j_ext, dtype=np.float64), v_shape)
        )

        return jnp.asarray(sic_out)
