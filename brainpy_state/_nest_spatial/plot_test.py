# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Smoke tests for the spatial plotting helpers (matplotlib-gated)."""
import unittest

import brainunit as u
import brainstate

matplotlib = None
try:
    import matplotlib
    matplotlib.use('Agg')                      # headless backend before pyplot import
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:                            # pragma: no cover - matplotlib optional
    matplotlib = None

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial.layers import grid
from brainpy_state._nest_spatial.kernels import distance, gaussian
from brainpy_state._nest_spatial.masks import circular
from brainpy_state._nest_spatial.rule import spatial_pairwise_bernoulli


@unittest.skipIf(matplotlib is None, 'matplotlib not installed')
class TestPlotHelpers(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(platform='cpu')

    def tearDown(self):
        plt.close('all')

    def _sim(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, positions=grid([5, 5], extent=[4.0, 4.0]))
        sim.connect(pop, pop, rule=spatial_pairwise_bernoulli(p=1.0, mask=circular(1.5)),
                    weight=1.0 * u.pA, delay=1.0 * u.ms)
        return sim, pop

    def test_plot_layer_2d_returns_figure_with_points(self):
        from brainpy_state._nest_spatial.plot import plot_layer
        fig = plot_layer(grid([3, 3], extent=[2.0, 2.0]))
        self.assertIsInstance(fig, Figure)
        self.assertTrue(fig.axes[0].collections)               # something was scattered

    def test_plot_layer_3d(self):
        from brainpy_state._nest_spatial.plot import plot_layer
        fig = plot_layer(grid([2, 2, 2], extent=[1.0, 1.0, 1.0]))
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.axes[0].name, '3d')

    def test_plot_targets(self):
        from brainpy_state._nest_spatial.plot import plot_targets
        sim, pop = self._sim()
        fig = plot_targets(sim, pop, pop)
        self.assertIsInstance(fig, Figure)
        self.assertGreaterEqual(len(fig.axes[0].collections), 2)   # targets + source

    def test_plot_sources(self):
        from brainpy_state._nest_spatial.plot import plot_sources
        sim, pop = self._sim()
        fig = plot_sources(sim, pop, pop)
        self.assertIsInstance(fig, Figure)
        self.assertGreaterEqual(len(fig.axes[0].collections), 2)

    def test_plot_probability_parameter_heatmap(self):
        from brainpy_state._nest_spatial.plot import plot_probability_parameter
        fig = plot_probability_parameter(gaussian(distance, std=0.3),
                                         mask=circular(0.4), extent=(-0.5, 0.5, -0.5, 0.5))
        self.assertIsInstance(fig, Figure)
        self.assertTrue(fig.axes[0].images)                    # an imshow heatmap


class TestPlotRequiresMatplotlib(unittest.TestCase):
    def test_clear_error_when_absent(self):
        # The lazy import path raises a clear ImportError when matplotlib is missing.
        import builtins
        from brainpy_state._nest_spatial import plot as _plot
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith('matplotlib'):
                raise ImportError('no matplotlib')
            return real_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(ImportError) as cm:
                _plot._import_mpl()
            self.assertIn('matplotlib', str(cm.exception))
        finally:
            builtins.__import__ = real_import


if __name__ == '__main__':
    unittest.main()
