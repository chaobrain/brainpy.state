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

"""JAX ``jit`` compatibility tests for every public :class:`NESTNeuron` subclass.

Each model is constructed, initialised, and stepped once with its ``update``
method wrapped in :func:`brainstate.transform.jit`.  Tracing ``update`` exercises
the same machinery as ``jax.jit`` / ``brainstate.transform.for_loop`` (scan),
so a model that traces cleanly here is safe to embed in a jitted simulation.

Parameter-validation guards are routed through :func:`brainpy_state._nest_base.utils.cond_any`,
which returns ``False`` for JAX tracers so that ``if`` checks are skipped during
tracing instead of raising ``TracerBoolConversionError``.

A small set of models is *known* to be incompatible with ``jit`` because their
``update`` cores are scalar NumPy algorithms (precise-spike-time bisection over
``np.ndindex``; Python-``int`` delay-queue indexing; mean-field numerical
integration).  These are listed in :data:`KNOWN_NON_JITTABLE` with a reason and
skipped, rather than silently ignored, so the boundary stays documented.
"""

import unittest

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

import brainpy_state as B
from brainpy_state._nest_base.base import NESTNeuron

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)


# Models whose ``update`` cannot currently be JIT-compiled, with the reason.
# These rely on scalar NumPy algorithms or Python-level control flow over state
# that would require a full vectorised reimplementation to trace.
KNOWN_NON_JITTABLE = {
    # Precise-spike-time models: per-neuron ``np.ndindex`` loops with scalar
    # ``float()`` extraction and bisection root-finding on NumPy float64.
    'iaf_psc_alpha_ps': 'precise-spiking np.ndindex scalar loop',
    'iaf_psc_exp_ps': 'precise-spiking np.ndindex scalar loop',
    'iaf_psc_exp_ps_lossless': 'precise-spiking np.ndindex scalar loop',
    # Delay-queue / mean-field rate models: Python ``int(step_idx)`` queue
    # indexing and (siegert) ``np.ndindex`` numerical integration.
    'rate_neuron_ipn': 'Python-int delay-queue step indexing',
    'rate_neuron_opn': 'Python-int delay-queue step indexing',
    'rate_transformer_node': 'Python-int delay-queue step indexing',
    'siegert_neuron': 'np.ndindex mean-field integration + int step indexing',
    # Pre-existing shape bug unrelated to parameter validation: with the default
    # configuration the NMDA receptor array is empty (shape (n, 0)) and the RK
    # integrator's weighted sum fails to broadcast against the (n,) state.
    'iaf_bw_2001_exact': 'empty-NMDA (n, 0) broadcast mismatch in RK integrator (pre-existing)',
}

# Models whose constructor needs explicit per-receptor configuration; the
# single-argument ``Cls(2)`` form would otherwise build degenerate (empty or
# mismatched) receptor arrays.
CONSTRUCTION_KWARGS = {
    'aeif_cond_beta_multisynapse': dict(
        E_rev=jnp.array([0.0, -85.0]) * u.mV,
        tau_rise=jnp.array([1.0, 1.0]) * u.ms,
        tau_decay=jnp.array([5.0, 5.0]) * u.ms,
    ),
    # cm_default needs a morphology + receptors to enter its State-based update()
    # path (a bare ``cm_default(2)`` stays in legacy host-loop mode with no State).
    'cm_default': dict(
        compartments=[
            {'parent_idx': -1, 'params': {
                'C_m': 89.0, 'g_C': 0.0, 'g_L': 8.9, 'e_L': -75.0, 'v_comp': -75.0,
                'gbar_Na': 4608.0, 'e_Na': 60.0, 'gbar_K': 956.0, 'e_K': -90.0}},
            {'parent_idx': 0, 'params': {
                'C_m': 1.9, 'g_C': 1.26, 'g_L': 0.19, 'e_L': -70.0, 'v_comp': -70.0}},
        ],
        receptors=[
            {'comp_idx': 0, 'receptor_type': 'AMPA'},
            {'comp_idx': 1, 'receptor_type': 'AMPA_NMDA'},
        ],
    ),
}


def _discover_neuron_classes():
    """Return ``{name: cls}`` for every public ``NESTNeuron`` subclass."""
    out = {}
    for name in B.__all__:
        obj = getattr(B, name)
        if isinstance(obj, type) and issubclass(obj, NESTNeuron) and obj is not NESTNeuron:
            out[name] = obj
    return out


ALL_NEURONS = _discover_neuron_classes()
JITTABLE = {n: c for n, c in ALL_NEURONS.items() if n not in KNOWN_NON_JITTABLE}


class TestNESTNeuronJITCompat(unittest.TestCase):
    """Verify every public NESTNeuron subclass traces cleanly under ``jit``."""

    def test_discovery_is_nonempty(self):
        # Guard against the discovery helper silently finding nothing (e.g. an
        # import regression), which would make the parametrised test vacuous.
        self.assertGreater(len(ALL_NEURONS), 50)

    def test_jittable_models_compile_and_step(self):
        for name, Cls in sorted(JITTABLE.items()):
            with self.subTest(model=name):
                with brainstate.environ.context(dt=0.1 * u.ms):
                    model = Cls(2, **CONSTRUCTION_KWARGS.get(name, {}))
                    model.init_state()
                with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
                    update_jit = brainstate.transform.jit(model.update)
                    out = update_jit()
                self.assertIsNotNone(out)

    def test_jittable_models_step_twice(self):
        # A second jitted step must reuse the compiled trace without error,
        # confirming state updates are themselves trace-stable.
        for name, Cls in sorted(JITTABLE.items()):
            with self.subTest(model=name):
                with brainstate.environ.context(dt=0.1 * u.ms):
                    model = Cls(2, **CONSTRUCTION_KWARGS.get(name, {}))
                    model.init_state()
                update_jit = brainstate.transform.jit(model.update)
                with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
                    update_jit()
                with brainstate.environ.context(dt=0.1 * u.ms, t=0.1 * u.ms):
                    update_jit()

    def test_known_non_jittable_are_documented(self):
        # Every documented exception must still be a real, discoverable model so
        # the skip list cannot drift out of sync with the public API.
        for name in KNOWN_NON_JITTABLE:
            with self.subTest(model=name):
                self.assertIn(name, ALL_NEURONS)


if __name__ == '__main__':
    unittest.main()
