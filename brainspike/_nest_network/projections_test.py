# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
import unittest

import jax.numpy as jnp
import brainunit as u


class TestSparseEventMatMulJit(unittest.TestCase):
    def test_int64_edges_run_under_jit(self):
        # Regression: int64 edge indices used to give an int64 indptr that
        # brainevent >= 0.2 cast inside jit, then failed to read back to host.
        import jax
        import numpy as np
        from brainspike._nest_network.projections import _SparseEventMatMul

        comm = _SparseEventMatMul(
            np.array([1, 0, 1], dtype=np.int64), np.array([0, 1, 1], dtype=np.int64),
            np.array([1., 2., 3.]), u.pA, n_pre=2, n_post=2)
        self.assertEqual(comm._indptr.dtype, jnp.int32)
        self.assertEqual(comm._indices.dtype, jnp.int32)
        out = jax.jit(comm)(jnp.array([1., 1.]))
        np.testing.assert_allclose(np.asarray(u.get_mantissa(out)), [1., 5.])
