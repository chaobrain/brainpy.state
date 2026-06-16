# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Exact parity for the spatial queries and dump helpers (goal 27).

``nearest_element`` / ``select_nodes_by_mask`` / ``dump_layer_nodes`` / ``dump_layer_connections``
are deterministic, so each is compared to its live-NEST counterpart bit-for-bit (after mapping
NEST's 1-based global ids to brainpy.state's 0-based population-local indices):

* ``nearest_element``        vs ``FindNearestElement``,
* ``select_nodes_by_mask``   vs ``SelectNodesByMask``,
* ``dump_layer_nodes``       vs ``DumpLayerNodes``      (index column offset, coords identical),
* ``dump_layer_connections`` vs ``DumpLayerConnections`` (per-edge weight/delay/displacement).
"""
import os
import tempfile
import unittest

import jax
import brainstate
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import brainunit as u

try:
    import nest
except Exception:
    nest = None

from brainpy_state import Simulator, iaf_psc_alpha
from brainpy_state._nest_spatial import grid, spatial_pairwise_bernoulli
from brainpy_state._nest_spatial.masks import circular, elliptical
from brainpy_state._nest_spatial.helpers import (
    nearest_element, select_nodes_by_mask, dump_layer_nodes, dump_layer_connections,
)
from brainpy_state._nest_validation.nest_compare import requires_nest

S, E = [5, 5], [4.0, 4.0]
LOCATIONS = [[0.9, 0.9], [-1.5, 0.3], [0.0, 0.0], [1.9, -1.9]]


def _parse(text):
    return [ln.split() for ln in text.strip().split('\n')]


class TestQueriesStructure(unittest.TestCase):
    """nearest / select / dump behave correctly on pinned layers (NEST-free)."""

    def test_nearest_pins(self):
        layer = grid([3, 1], extent=[3.0, 1.0])             # x = -1, 0, +1
        self.assertEqual(nearest_element(layer, [0.9, 0.0]), 2)
        self.assertEqual(nearest_element(layer, [[-0.9, 0.0], [0.1, 0.0]]), [0, 1])

    def test_select_pins(self):
        layer = grid([3, 3], extent=[2.0, 2.0])
        self.assertEqual(sorted(select_nodes_by_mask(layer, [0.0, 0.0], circular(0.7))),
                         [1, 3, 4, 5, 7])

    def test_dump_nodes_local_index(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, positions=grid([2, 2], extent=[2.0, 2.0]))
        path = tempfile.mktemp(suffix='.txt')
        rows = _parse(dump_layer_nodes(sim, pop, path))
        self.assertEqual([int(r[0]) for r in rows], [0, 1, 2, 3])
        os.remove(path)


@requires_nest
class TestQueriesNestParity(unittest.TestCase):
    """nearest / select / dump match live NEST after id normalisation."""

    def _bp_sim(self, mask=None):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_alpha, positions=grid(S, extent=E))
        if mask is not None:
            sim.connect(pop, pop, rule=spatial_pairwise_bernoulli(p=1.0, mask=mask),
                        weight=2.0 * u.pA, delay=1.5 * u.ms, seed=0)
        return sim, pop

    def _nest_layer(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        pos = nest.spatial.grid(shape=S, extent=E)
        a = nest.Create('iaf_psc_alpha', positions=pos)
        return a, a[0].global_id

    def test_nearest_matches_nest(self):
        layer = grid(S, extent=E)
        a, g0 = self._nest_layer()
        nest_ids = [nest.FindNearestElement(a, loc)[0].global_id - g0 for loc in LOCATIONS]
        bp_ids = [nearest_element(layer, loc) for loc in LOCATIONS]
        self.assertEqual(bp_ids, nest_ids)

    def test_select_matches_nest(self):
        layer = grid(S, extent=E)
        a, g0 = self._nest_layer()
        for anchor, mask, mt, spec in [
                ([0.0, 0.0], circular(1.2), 'circular', {'radius': 1.2}),
                ([0.3, -0.4], elliptical(2.4, 1.0, azimuth_angle=20.0), 'elliptical',
                 {'major_axis': 2.4, 'minor_axis': 1.0, 'azimuth_angle': 20.0})]:
            sel = nest.SelectNodesByMask(a, anchor, nest.CreateMask(mt, spec))
            nest_idx = sorted(int(x) - g0 for x in sel.tolist())
            bp_idx = sorted(int(x) for x in select_nodes_by_mask(layer, anchor, mask).tolist())
            self.assertEqual(bp_idx, nest_idx, msg=f'select {mt} @ {anchor}')

    def test_dump_layer_nodes_matches_nest(self):
        sim, pop = self._bp_sim()
        a, g0 = self._nest_layer()
        bp_path, nest_path = tempfile.mktemp(suffix='.txt'), tempfile.mktemp(suffix='.txt')
        bp_rows = _parse(dump_layer_nodes(sim, pop, bp_path))
        nest.DumpLayerNodes(a, nest_path)
        nest_rows = _parse(open(nest_path).read())
        self.assertEqual(len(bp_rows), len(nest_rows))
        for bp_r, nest_r in zip(bp_rows, nest_rows):
            self.assertEqual(int(bp_r[0]), int(nest_r[0]) - g0)             # index offset
            np.testing.assert_allclose([float(x) for x in bp_r[1:]],
                                       [float(x) for x in nest_r[1:]], atol=1e-9)  # coords
        os.remove(bp_path)
        os.remove(nest_path)

    def test_dump_layer_connections_matches_nest(self):
        mask = circular(1.2)
        sim, pop = self._bp_sim(mask=mask)
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        pos = nest.spatial.grid(shape=S, extent=E)
        a = nest.Create('iaf_psc_alpha', positions=pos)
        g0 = a[0].global_id
        nest.Connect(a, a, {'rule': 'pairwise_bernoulli', 'p': 1.0,
                            'mask': {'circular': {'radius': 1.2}}},
                     {'synapse_model': 'static_synapse', 'weight': 2.0, 'delay': 1.5})
        bp_path, nest_path = tempfile.mktemp(suffix='.txt'), tempfile.mktemp(suffix='.txt')
        bp_rows = _parse(dump_layer_connections(sim, pop, pop, bp_path))
        nest.DumpLayerConnections(a, a, 'static_synapse', nest_path)
        nest_rows = _parse(open(nest_path).read())

        def as_map(rows, offset):
            out = {}
            for r in rows:
                key = (int(r[0]) - offset, int(r[1]) - offset)
                out[key] = np.array([float(x) for x in r[2:]])     # weight, delay, dx, dy
            return out

        bp_map, nest_map = as_map(bp_rows, 0), as_map(nest_rows, g0)
        self.assertEqual(set(bp_map), set(nest_map))                # same edge set
        for key, bp_vals in bp_map.items():
            np.testing.assert_allclose(bp_vals, nest_map[key], atol=1e-9,
                                       err_msg=f'edge {key} weight/delay/displacement diverges')
        os.remove(bp_path)
        os.remove(nest_path)


if __name__ == '__main__':
    unittest.main()
