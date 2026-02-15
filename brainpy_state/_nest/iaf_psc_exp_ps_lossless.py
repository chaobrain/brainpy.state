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
from typing import Callable, Iterable

import numpy as np

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike, Size

from brainpy_state._base import Neuron
from .iaf_psc_exp import iaf_psc_exp

__all__ = [
    'iaf_psc_exp_ps_lossless',
]


class iaf_psc_exp_ps_lossless(Neuron):
    r"""NEST-compatible ``iaf_psc_exp_ps_lossless``.

    Precise-time exponential-PSC neuron with lossless spike detection, matching
    NEST ``iaf_psc_exp_ps_lossless``.

    Compared with :class:`iaf_psc_exp_ps`, this model adds a state-space spike
    detector (Krishnan et al., 2018) that can detect spikes hidden between
    sampled endpoints of a mini-interval.

    **Core behavior**

    - Supports off-grid input spike times via within-step offsets.
    - Splits each simulation step into mini-intervals around events.
    - Uses the same exact subthreshold propagator as ``iaf_psc_exp_ps``.
    - Detects spikes using the lossless criterion before propagation in each
      mini-interval.
    - Emits spike time with sub-step precision and applies precise refractory
      release pseudo-events.

    **Implementation scope**

    - ``tau_syn_ex == tau_syn_in`` is required in this implementation.
    - Refractory duration can be zero (consistent with NEST lossless model).
    """

    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        E_L: ArrayLike = -70. * u.mV,
        C_m: ArrayLike = 250. * u.pF,
        tau_m: ArrayLike = 10. * u.ms,
        t_ref: ArrayLike = 2. * u.ms,
        V_th: ArrayLike = -55. * u.mV,
        V_reset: ArrayLike = -70. * u.mV,
        tau_syn_ex: ArrayLike = 2. * u.ms,
        tau_syn_in: ArrayLike = 2. * u.ms,
        I_e: ArrayLike = 0. * u.pA,
        V_min: ArrayLike = None,
        V_initializer: Callable = braintools.init.Constant(-70. * u.mV),
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
        self.V_min = None if V_min is None else braintools.init.param(V_min, self.varshape)
        self.V_initializer = V_initializer
        self.ref_var = ref_var

        self._validate_parameters()

    @staticmethod
    def _to_numpy(x, unit):
        return np.asarray(u.math.asarray(x / unit), dtype=np.float64)

    @staticmethod
    def _broadcast_to_state(x_np: np.ndarray, shape):
        return np.broadcast_to(x_np, shape)

    def _validate_parameters(self):
        if np.any(self._to_numpy(self.V_reset, u.mV) >= self._to_numpy(self.V_th, u.mV)):
            raise ValueError('Reset potential must be smaller than threshold.')
        if self.V_min is not None and np.any(self._to_numpy(self.V_reset, u.mV) < self._to_numpy(self.V_min, u.mV)):
            raise ValueError('Reset potential must be greater than or equal to minimum potential.')
        if np.any(self._to_numpy(self.C_m, u.pF) <= 0.0):
            raise ValueError('Capacitance must be strictly positive.')
        if np.any(self._to_numpy(self.t_ref, u.ms) < 0.0):
            raise ValueError('Refractory time must not be negative.')
        tau_ex = self._to_numpy(self.tau_syn_ex, u.ms)
        tau_in = self._to_numpy(self.tau_syn_in, u.ms)
        tau_m = self._to_numpy(self.tau_m, u.ms)
        if np.any(tau_m <= 0.0) or np.any(tau_ex <= 0.0) or np.any(tau_in <= 0.0):
            raise ValueError('All time constants must be strictly positive.')
        if np.any(np.abs(tau_ex - tau_in) > 0.0):
            raise ValueError('tau_syn_ex == tau_syn_in is required in this implementation.')
        if np.any(np.isclose(tau_m, tau_ex)) or np.any(np.isclose(tau_m, tau_in)):
            raise ValueError('Membrane and synapse time constants must differ.')

    def init_state(self, batch_size: int = None, **kwargs):
        V = braintools.init.param(self.V_initializer, self.varshape, batch_size)
        zeros = u.math.zeros_like(u.math.asarray(V / u.mV))
        spk_time = braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape, batch_size)
        last_step = braintools.init.param(braintools.init.Constant(-1), self.varshape, batch_size)

        self.V = brainstate.HiddenState(V)
        self.I_syn_ex = brainstate.ShortTermState(zeros * u.pA)
        self.I_syn_in = brainstate.ShortTermState(zeros * u.pA)
        self.y0 = brainstate.ShortTermState(zeros * u.pA)
        self.is_refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))
        self.last_spike_step = brainstate.ShortTermState(u.math.asarray(last_step, dtype=jnp.int32))
        self.last_spike_offset = brainstate.ShortTermState(zeros * u.ms)
        self.last_spike_time = brainstate.ShortTermState(spk_time)

        if self.ref_var:
            self.refractory = brainstate.ShortTermState(np.zeros(V.shape, dtype=bool))

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def _parse_spike_events(self, spike_events: Iterable, v_shape):
        events = []
        if spike_events is None:
            return events
        for ev in spike_events:
            if isinstance(ev, dict):
                offs = ev.get('offset', 0.0 * u.ms)
                w = ev.get('weight', 0.0 * u.pA)
            else:
                offs, w = ev
            off_ms = float(u.math.asarray(offs / u.ms))
            w_np = np.asarray(u.math.asarray(w / u.pA), dtype=np.float64)
            events.append((off_ms, np.broadcast_to(w_np, v_shape)))
        return events

    @staticmethod
    def _bisect_root(f, t_hi: float):
        lo = 0.0
        hi = float(t_hi)
        f_lo = f(lo)
        f_hi = f(hi)
        if not np.isfinite(f_hi):
            return hi
        if f_lo > 0.0:
            return 0.0
        if f_hi <= 0.0:
            return hi
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            if f_mid > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def update(self, x=0. * u.pA, spike_events=None):
        t = brainstate.environ.get('t')
        dt_q = brainstate.environ.get_dt()
        h = float(u.math.asarray(dt_q / u.ms))
        t_ms = float(u.math.asarray(t / u.ms))
        step_idx = int(round(t_ms / h))
        eps = np.finfo(np.float64).eps

        v_shape = self.V.value.shape

        E_L = self._broadcast_to_state(self._to_numpy(self.E_L, u.mV), v_shape)
        y2 = self._broadcast_to_state(self._to_numpy(self.V.value, u.mV), v_shape) - E_L
        I_syn_ex = self._broadcast_to_state(self._to_numpy(self.I_syn_ex.value, u.pA), v_shape)
        I_syn_in = self._broadcast_to_state(self._to_numpy(self.I_syn_in.value, u.pA), v_shape)
        y0 = self._broadcast_to_state(self._to_numpy(self.y0.value, u.pA), v_shape)

        is_refractory = self._broadcast_to_state(np.asarray(u.math.asarray(self.is_refractory.value), dtype=bool), v_shape)
        last_spike_step = self._broadcast_to_state(np.asarray(u.math.asarray(self.last_spike_step.value), dtype=np.int32), v_shape)
        last_spike_offset = self._broadcast_to_state(self._to_numpy(self.last_spike_offset.value, u.ms), v_shape)
        last_spike_time_prev = self._broadcast_to_state(self._to_numpy(self.last_spike_time.value, u.ms), v_shape)

        tau_m = self._broadcast_to_state(self._to_numpy(self.tau_m, u.ms), v_shape)
        tau_ex = self._broadcast_to_state(self._to_numpy(self.tau_syn_ex, u.ms), v_shape)
        tau_in = self._broadcast_to_state(self._to_numpy(self.tau_syn_in, u.ms), v_shape)
        c_m = self._broadcast_to_state(self._to_numpy(self.C_m, u.pF), v_shape)
        i_e = self._broadcast_to_state(self._to_numpy(self.I_e, u.pA), v_shape)
        u_th = self._broadcast_to_state(self._to_numpy(self.V_th - self.E_L, u.mV), v_shape)
        u_reset = self._broadcast_to_state(self._to_numpy(self.V_reset - self.E_L, u.mV), v_shape)
        u_min = -np.inf * np.ones(v_shape, dtype=np.float64)
        if self.V_min is not None:
            u_min = self._broadcast_to_state(self._to_numpy(self.V_min - self.E_L, u.mV), v_shape)

        refr_steps = self._broadcast_to_state(
            np.asarray(np.ceil(self._to_numpy(self.t_ref, u.ms) / h), dtype=np.int64),
            v_shape,
        )
        if np.any(refr_steps < 0):
            raise ValueError('Refractory time must not be negative.')

        events = self._parse_spike_events(spike_events, v_shape)
        on_grid = self._broadcast_to_state(self._to_numpy(self.sum_delta_inputs(0. * u.pA), u.pA), v_shape)
        events.append((0.0, on_grid))
        events.sort(key=lambda z: z[0], reverse=True)
        for off, _ in events:
            if off < 0.0 or off > h:
                raise ValueError('All spike event offsets must satisfy 0 <= offset <= dt.')

        y0_next = self._broadcast_to_state(self._to_numpy(self.sum_current_inputs(x, self.V.value), u.pA), v_shape)

        y0_new = np.empty_like(y0)
        y1_ex_new = np.empty_like(I_syn_ex)
        y1_in_new = np.empty_like(I_syn_in)
        y2_new = np.empty_like(y2)
        refr_new = np.empty_like(is_refractory)
        last_step_new = np.empty_like(last_spike_step)
        last_offset_new = np.empty_like(last_spike_offset)
        last_time_new = np.empty_like(last_spike_time_prev)
        spike_mask = np.zeros(v_shape, dtype=bool)
        v_for_spike = np.empty_like(y2)

        for idx in np.ndindex(v_shape):
            y0_i = float(y0[idx])
            y1e_i = float(I_syn_ex[idx])
            y1i_i = float(I_syn_in[idx])
            y2_i = float(y2[idx])
            refr_i = bool(is_refractory[idx])
            last_step_i = int(last_spike_step[idx])
            last_off_i = float(last_spike_offset[idx])
            spike_time_i = float(last_spike_time_prev[idx])

            tau_m_i = float(tau_m[idx])
            tau_ex_i = float(tau_ex[idx])
            tau_in_i = float(tau_in[idx])
            c_m_i = float(c_m[idx])
            i_e_i = float(i_e[idx])
            u_th_i = float(u_th[idx])
            u_reset_i = float(u_reset[idx])
            u_min_i = float(u_min[idx])
            refr_steps_i = int(refr_steps[idx])

            did_spike = False
            before = [y0_i, y1e_i, y1i_i, y2_i]

            def set_before():
                before[0] = y0_i
                before[1] = y1e_i
                before[2] = y1i_i
                before[3] = y2_i

            def threshold_distance(dt_local):
                P20 = -tau_m_i / c_m_i * math.expm1(-dt_local / tau_m_i)
                P21e = iaf_psc_exp._propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                P21i = iaf_psc_exp._propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                y2_r = P20 * (i_e_i + before[0]) + P21e * before[1] + P21i * before[2] + before[3] * math.exp(-dt_local / tau_m_i)
                return y2_r - u_th_i

            def propagate(dt_local):
                nonlocal y1e_i, y1i_i, y2_i
                if dt_local <= 0.0:
                    return
                if not refr_i:
                    P20 = -tau_m_i / c_m_i * math.expm1(-dt_local / tau_m_i)
                    P21e = iaf_psc_exp._propagator_exp(np.asarray(tau_ex_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                    P21i = iaf_psc_exp._propagator_exp(np.asarray(tau_in_i), np.asarray(tau_m_i), np.asarray(c_m_i), dt_local)
                    y2_i = P20 * (i_e_i + y0_i) + P21e * y1e_i + P21i * y1i_i + y2_i * math.exp(-dt_local / tau_m_i)
                    y2_i = max(y2_i, u_min_i)
                y1e_i = y1e_i * math.exp(-dt_local / tau_ex_i)
                y1i_i = y1i_i * math.exp(-dt_local / tau_in_i)

            def is_spike_lossless(dt_local):
                if dt_local <= 0.0:
                    return np.nan
                I0 = before[1] + before[2]
                V0 = before[3]
                exp_tau_s = math.expm1(dt_local / tau_ex_i)
                exp_tau_m = math.expm1(dt_local / tau_m_i)
                exp_tau_m_s = math.expm1(dt_local / tau_m_i - dt_local / tau_ex_i)
                Ie_tot = before[0] + i_e_i

                a1 = tau_m_i * tau_ex_i
                a2 = tau_m_i * (tau_m_i - tau_ex_i)
                a3 = c_m_i * u_th_i * (tau_m_i - tau_ex_i)
                a4 = c_m_i * (tau_m_i - tau_ex_i)

                b1 = -tau_m_i * tau_m_i
                b2 = tau_m_i * tau_ex_i
                b3 = tau_m_i * c_m_i * u_th_i
                b4 = -c_m_i * (tau_m_i - tau_ex_i)

                c1 = tau_m_i / c_m_i
                c2 = (-tau_m_i * tau_ex_i) / (c_m_i * (tau_m_i - tau_ex_i))
                c3 = (tau_m_i * tau_m_i) / (c_m_i * (tau_m_i - tau_ex_i))
                c4 = tau_ex_i / tau_m_i
                c5 = (c_m_i * u_th_i) / tau_m_i
                c6 = 1.0 - (tau_ex_i / tau_m_i)

                f = (a1 * I0 * exp_tau_m_s + exp_tau_m * (a3 - Ie_tot * a2) + a3) / a4
                g = ((I0 + Ie_tot) * (b1 * exp_tau_m + b2 * exp_tau_s) + b3 * (exp_tau_m - exp_tau_s)) / (b4 * exp_tau_s)
                b_env = c1 * Ie_tot + c2 * I0 + c3 * (I0 ** c4) * ((c5 - Ie_tot) ** c6)

                if (V0 < g) and (V0 <= f):
                    return np.nan
                if V0 >= f:
                    return dt_local
                if V0 < b_env:
                    return np.nan
                try:
                    return (a1 / (tau_m_i * tau_ex_i)) * math.log(b1 * I0 / (a2 * Ie_tot - a1 * I0 - a4 * V0))
                except (ValueError, ZeroDivisionError):
                    return np.nan

            def emit_spike(t0, dt_local):
                nonlocal y2_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                root = self._bisect_root(threshold_distance, dt_local)
                spike_off = h - (t0 + root)
                spike_off = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = spike_off
                y2_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - spike_off
                did_spike = True

            def emit_instant_spike(spike_off):
                nonlocal y2_i, refr_i, last_step_i, last_off_i, spike_time_i, did_spike
                so = min(h, max(0.0, spike_off))
                last_step_i = step_idx + 1
                last_off_i = so
                y2_i = u_reset_i
                refr_i = True
                spike_time_i = t_ms + h - so
                did_spike = True

            if (not refr_i) and (y2_i >= u_th_i):
                emit_instant_spike(h * (1.0 - eps))

            local_events = [(off, w[idx], False) for off, w in events]
            if refr_i and (step_idx + 1 - last_step_i == refr_steps_i):
                local_events.append((last_off_i, 0.0, True))
            local_events.sort(key=lambda z: z[0], reverse=True)

            last_off = h
            if len(local_events) == 0:
                set_before()
                st = is_spike_lossless(h)
                propagate(h)
                if np.isfinite(st):
                    emit_spike(0.0, st)
            else:
                for ev_off, ev_w, end_of_refract in local_events:
                    ministep = last_off - ev_off
                    if ministep > 0.0:
                        set_before()
                        st = is_spike_lossless(ministep)
                        propagate(ministep)
                        if np.isfinite(st):
                            emit_spike(h - last_off, st)
                    if end_of_refract:
                        refr_i = False
                    else:
                        if ev_w >= 0.0:
                            y1e_i += ev_w
                        else:
                            y1i_i += ev_w
                    set_before()
                    last_off = ev_off
                if last_off > 0.0:
                    set_before()
                    st = is_spike_lossless(last_off)
                    propagate(last_off)
                    if np.isfinite(st):
                        emit_spike(h - last_off, st)

            y0_i = float(y0_next[idx])
            y0_new[idx] = y0_i
            y1_ex_new[idx] = y1e_i
            y1_in_new[idx] = y1i_i
            y2_new[idx] = y2_i
            refr_new[idx] = refr_i
            last_step_new[idx] = last_step_i
            last_offset_new[idx] = last_off_i
            last_time_new[idx] = spike_time_i
            spike_mask[idx] = did_spike
            v_for_spike[idx] = (u_th_i + 1e-12) if did_spike else min(y2_i, u_th_i - 1e-12)

        self.y0.value = y0_new * u.pA
        self.I_syn_ex.value = y1_ex_new * u.pA
        self.I_syn_in.value = y1_in_new * u.pA
        self.V.value = (y2_new + E_L) * u.mV
        self.is_refractory.value = jnp.asarray(refr_new, dtype=bool)
        self.last_spike_step.value = jnp.asarray(last_step_new, dtype=jnp.int32)
        self.last_spike_offset.value = last_offset_new * u.ms
        self.last_spike_time.value = jax.lax.stop_gradient(last_time_new * u.ms)
        if self.ref_var:
            self.refractory.value = jax.lax.stop_gradient(self.is_refractory.value)

        return self.get_spike((v_for_spike + E_L) * u.mV)
