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

import importlib.util
import unittest

import brainstate
import braintools
import saiunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from brainpy.state import aeif_cond_alpha_astro, sic_connection

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')


def _is_nest_available():
    return importlib.util.find_spec('nest') is not None


class TestSICConnection(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt_ms = 0.1
        self.dt = self.dt_ms * u.ms

    def _step(self, neuron, k, sic_events=None):
        with brainstate.environ.context(t=(k * self.dt_ms) * u.ms):
            neuron.update(x=0.0 * u.pA, sic_events=sic_events)

    def test_nest_default_parameters_and_properties(self):
        syn = sic_connection()
        self.assertAlmostEqual(syn.weight, 1.0, delta=0.0)
        self.assertEqual(syn.delay_steps, 1)
        self.assertTrue(syn.HAS_DELAY)
        self.assertFalse(syn.SUPPORTS_WFR)
        self.assertEqual(syn.properties['has_delay'], True)
        self.assertEqual(syn.properties['supports_wfr'], False)

        status = syn.get_status()
        self.assertAlmostEqual(status['weight'], 1.0, delta=0.0)
        self.assertEqual(status['delay_steps'], 1)
        self.assertEqual(status['delay'], 1)
        self.assertIn('size_of', status)
        self.assertEqual(status['supported_sources'], ('astrocyte_lr_1994',))
        self.assertEqual(status['supported_targets'], ('aeif_cond_alpha_astro',))

        if _is_nest_available():
            import nest

            nest.ResetKernel()
            defaults = nest.GetDefaults('sic_connection')
            self.assertAlmostEqual(syn.weight, float(defaults['weight']), delta=0.0)
            self.assertIn('delay', defaults)
            self.assertGreaterEqual(float(defaults['delay']), float(nest.resolution))

    def test_set_status_delay_validation_and_supported_pairs(self):
        syn = sic_connection()
        syn.set_status({'weight': 2.5, 'delay_steps': 7})
        self.assertAlmostEqual(syn.weight, 2.5, delta=0.0)
        self.assertEqual(syn.delay_steps, 7)

        syn.set_status(delay=4)
        self.assertEqual(syn.delay_steps, 4)

        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            syn.set_delay_steps(0)
        with self.assertRaisesRegex(ValueError, 'must be >= 1'):
            syn.set_delay(-1)
        with self.assertRaisesRegex(ValueError, 'integer-valued'):
            syn.set_status(delay=2.5)
        with self.assertRaisesRegex(ValueError, 'must be identical'):
            syn.set_status(delay=2, delay_steps=3)

        self.assertTrue(syn.supports_connection('astrocyte_lr_1994', 'aeif_cond_alpha_astro'))
        self.assertFalse(syn.supports_connection('iaf_psc_exp', 'aeif_cond_alpha_astro'))
        self.assertFalse(syn.supports_connection('astrocyte_lr_1994', 'iaf_psc_exp'))
        with self.assertRaisesRegex(ValueError, 'Unsupported sic_connection pair'):
            syn.check_connection('iaf_psc_exp', 'aeif_cond_alpha_astro')

        if _is_nest_available():
            import nest

            models = ['astrocyte_lr_1994', 'aeif_cond_alpha_astro', 'iaf_psc_exp']
            if not set(models).issubset(set(nest.Models())):
                self.skipTest('Required NEST models for sic_connection pair checks are not available.')

            for src_model in models:
                for tgt_model in models:
                    expected = syn.supports_connection(src_model, tgt_model)
                    nest.ResetKernel()
                    src = nest.Create(src_model)
                    tgt = nest.Create(tgt_model)
                    if expected:
                        nest.Connect(src, tgt, syn_spec={'synapse_model': 'sic_connection'})
                    else:
                        with self.assertRaises(nest.kernel.NESTError):
                            nest.Connect(src, tgt, syn_spec={'synapse_model': 'sic_connection'})

    def test_event_helpers_match_nest_delay_mapping(self):
        syn = sic_connection(weight=2.0, delay_steps=7)
        dftype = brainstate.environ.dftype()
        coeff = np.asarray([0.5, -1.25, 3.0, 0.0], dtype=dftype)

        sec = syn.prepare_secondary_event(coeff)
        npt.assert_allclose(sec['coeffarray'], coeff, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(sec['weight'], 2.0, delta=0.0)
        self.assertEqual(sec['delay_steps'], 7)

        event = syn.to_aeif_sic_event(coeff, min_delay_steps=3, multiplicity=4.0)
        npt.assert_allclose(event['coeffs'], coeff, atol=0.0, rtol=0.0)
        self.assertAlmostEqual(event['weight'], 8.0, delta=0.0)
        self.assertEqual(event['delay_steps'], 5)

        mapped = syn.coeffarray_to_step_events(coeff, min_delay_steps=3, multiplicity=4.0)
        self.assertEqual(len(mapped), coeff.size)
        for i, ev in enumerate(mapped):
            self.assertAlmostEqual(ev['coeffs'], float(coeff[i]), delta=0.0)
            self.assertAlmostEqual(ev['weight'], 8.0, delta=0.0)
            self.assertEqual(ev['delay_steps'], 5 + i)

        with brainstate.environ.context(dt=self.dt):
            n_coeffarray = aeif_cond_alpha_astro(
                1,
                V_th=1e6 * u.mV,
                V_peak=1e6 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                C_m=100.0 * u.pF,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                V_reset=0.0 * u.mV,
            )
            n_mapped = aeif_cond_alpha_astro(
                1,
                V_th=1e6 * u.mV,
                V_peak=1e6 * u.mV,
                Delta_T=0.0 * u.mV,
                g_L=0.0 * u.nS,
                C_m=100.0 * u.pF,
                a=0.0 * u.nS,
                b=0.0 * u.pA,
                I_e=0.0 * u.pA,
                V_initializer=braintools.init.Constant(0.0 * u.mV),
                V_reset=0.0 * u.mV,
            )
            n_coeffarray.init_state()
            n_mapped.init_state()

            n_steps = 20
            v_shape = n_coeffarray.V.value.shape  # (1,)

            # Pre-compute I_SIC schedules using Python queue simulation.
            # i_sic_sched[k] = value popped from queue at step k
            # (stored into I_sic.value at end of step k, used by ODE at step k+1).
            i_sic_sched_coeff = np.zeros((n_steps,) + v_shape, dtype=dftype)
            i_sic_sched_mapped = np.zeros((n_steps,) + v_shape, dtype=dftype)
            for k in range(n_steps):
                ev_c = event if k == 0 else None
                ev_m = mapped if k == 0 else None
                n_coeffarray._enqueue_sic_events(ev_c, v_shape)
                i_sic_sched_coeff[k] = n_coeffarray._pop_sic_current(v_shape)
                n_coeffarray._sic_step += 1
                n_mapped._enqueue_sic_events(ev_m, v_shape)
                i_sic_sched_mapped[k] = n_mapped._pop_sic_current(v_shape)
                n_mapped._sic_step += 1

            # Reset Python queues; the for_loop will inject I_sic via JAX arrays.
            n_coeffarray._sic_queue = {}
            n_coeffarray._sic_step = 0
            n_mapped._sic_queue = {}
            n_mapped._sic_step = 0

            i_sic_coeff_jax = jnp.asarray(i_sic_sched_coeff)   # (n_steps, 1)
            i_sic_mapped_jax = jnp.asarray(i_sic_sched_mapped)  # (n_steps, 1)
            t_array = jnp.arange(n_steps, dtype=dftype) * self.dt_ms  # ms

            def body(inputs):
                t_k, i_sic_coeff_k, i_sic_mapped_k = inputs
                with brainstate.environ.context(t=t_k * u.ms):
                    # update() reads I_sic.value (set at previous step) for the ODE,
                    # then sets I_sic.value = 0 (empty queue pop).
                    n_coeffarray.update(x=0.0 * u.pA, sic_events=None)
                    # Override with pre-computed value for next step's ODE.
                    n_coeffarray.I_sic.value = i_sic_coeff_k * u.pA
                    n_mapped.update(x=0.0 * u.pA, sic_events=None)
                    n_mapped.I_sic.value = i_sic_mapped_k * u.pA
                return (
                    u.get_mantissa(n_coeffarray.I_sic.value)[0],
                    u.get_mantissa(n_mapped.I_sic.value)[0],
                )

            results = brainstate.transform.for_loop(body, (t_array, i_sic_coeff_jax, i_sic_mapped_jax))
            trace_coeffarray = np.array(results[0])
            trace_mapped = np.array(results[1])

        npt.assert_allclose(trace_coeffarray, trace_mapped, atol=1e-12, rtol=0.0)

    def test_i_sic_trace_matches_nest_pipeline(self):
        if not _is_nest_available():
            self.skipTest('NEST simulator not available')

        import nest

        required_models = {'astrocyte_lr_1994', 'aeif_cond_alpha_astro'}
        if not required_models.issubset(set(nest.Models())):
            self.skipTest('NEST astrocyte/SIC models not available')

        dt_ms = self.dt_ms
        sim_ms = 30.0
        weight = 1.7

        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.resolution = dt_ms
        nest.use_wfr = False
        nest.local_num_threads = 1

        astro_defaults = nest.GetDefaults('astrocyte_lr_1994')
        sic_defaults = nest.GetDefaults('sic_connection')
        sic_delay_ms = float(sic_defaults['delay'])
        sic_th = float(astro_defaults['SIC_th'])
        sic_scale = float(astro_defaults['SIC_scale'])

        astro = nest.Create('astrocyte_lr_1994', params={'Ca_astro': 0.2})
        nrn = nest.Create('aeif_cond_alpha_astro')
        nest.Connect(astro, nrn, syn_spec={'synapse_model': 'sic_connection', 'weight': weight, 'delay': sic_delay_ms})

        mm_nrn = nest.Create('multimeter', params={'record_from': ['I_SIC'], 'interval': dt_ms})
        mm_astro = nest.Create('multimeter', params={'record_from': ['Ca_astro'], 'interval': dt_ms})
        nest.Connect(mm_nrn, nrn)
        nest.Connect(mm_astro, astro)

        nest.Simulate(sim_ms)

        nrn_events = mm_nrn.get('events')
        astro_events = mm_astro.get('events')
        dftype = brainstate.environ.dftype()
        nest_times = np.asarray(nrn_events['times'], dtype=dftype)
        nest_i_sic = np.asarray(nrn_events['I_SIC'], dtype=dftype)
        ca = np.asarray(astro_events['Ca_astro'], dtype=dftype)
        self.assertGreater(ca.size, 0)

        calc_thr = (ca - sic_th) * 1000.0
        coeff = np.zeros_like(calc_thr)
        mask = calc_thr > 1.0
        coeff[mask] = np.log(calc_thr[mask]) * sic_scale

        # Multimeter samples correspond to end-of-step states (t+dt), so we shift by one step
        # when replaying SIC events in the local step loop.
        delay_steps = int(round(sic_delay_ms / dt_ms)) + 1
        syn = sic_connection(weight=weight, delay_steps=delay_steps)

        with brainstate.environ.context(dt=self.dt):
            neuron = aeif_cond_alpha_astro(1)
            neuron.init_state()

            n_steps = ca.size
            v_shape = neuron.V.value.shape  # (1,)

            # Pre-compute I_SIC schedule using Python queue simulation.
            i_sic_sched = np.zeros((n_steps,) + v_shape, dtype=dftype)
            for k in range(n_steps):
                ev = syn.to_sic_event(coeff=coeff[k], min_delay_steps=1)
                neuron._enqueue_sic_events(ev, v_shape)
                i_sic_sched[k] = neuron._pop_sic_current(v_shape)
                neuron._sic_step += 1

            # Reset Python queue; inject I_sic via JAX array in for_loop.
            neuron._sic_queue = {}
            neuron._sic_step = 0

            i_sic_jax = jnp.asarray(i_sic_sched)  # (n_steps, 1)
            t_array = jnp.arange(n_steps, dtype=dftype) * dt_ms  # ms

            def body(inputs):
                t_k, i_sic_k = inputs
                with brainstate.environ.context(t=t_k * u.ms):
                    neuron.update(x=0.0 * u.pA, sic_events=None)
                    neuron.I_sic.value = i_sic_k * u.pA
                return u.get_mantissa(neuron.I_sic.value)[0]

            bp_i_sic = np.array(brainstate.transform.for_loop(body, (t_array, i_sic_jax)))

        bp_indices = np.rint(nest_times / dt_ms).astype(np.int64) - 1
        valid = (bp_indices >= 0) & (bp_indices < bp_i_sic.size)
        bp_indices = bp_indices[valid]

        npt.assert_allclose(
            bp_i_sic[bp_indices],
            nest_i_sic[valid],
            atol=2e-10,
            rtol=0.0,
            err_msg='I_SIC trace mismatch vs NEST with sic_connection replay',
        )


if __name__ == '__main__':
    unittest.main()
