# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Guard the STDP parity-docs cross-links and the divergence page's doctests.

NEST-free introspection test (collected by CI's ``pytest brainpy_state/``). It
keeps the docstring -> docs cross-links from rotting and *executes* the runnable
doctests on ``docs/nest-guide/stdp-divergences.rst`` — the divergence reference
page authored in cluster 10. The page's ``brainpy.state`` examples cannot be
collected by ``--doctest-modules`` (the spec classes set
``__module__='brainpy.state'``, so ``DocTestFinder`` skips them), so they live in
the ``.rst`` and are run here via :func:`doctest.testfile`. See ``develop/NEST_PARITY_LEDGER.md``
Lessons (10-stdp-docs).

Uses the inner package name ``brainpy_state`` (per ``CLAUDE.md`` rule 9); the
page's example code uses the public ``brainpy.state``.
"""
from __future__ import annotations

import doctest
import re
import unittest
from pathlib import Path

import brainpy_state as bps

# The divergence reference page, located relative to the repo root. It is absent
# when the package is installed without the docs tree, so the page-dependent
# tests skip rather than error in that environment.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / 'docs' / 'nest-guide' / 'stdp-divergences.rst'

# The stable cross-link marker every target docstring must carry.
_DOC_LINK = ':doc:`/nest-guide/stdp-divergences`'

# Each ``stdp_*``/``clopath_``/``stdp_dopamine_`` spec -> the section anchor its
# Parity note points at. Every anchor must be defined in the page (no dangling
# ``:ref:``) and cited by the spec's docstring.
_SPEC_REF = {
    'stdp_synapse': 'stdp-tau-minus',
    'stdp_synapse_hom': 'stdp-tau-minus',
    'stdp_pl_synapse_hom': 'stdp-tau-minus',
    'stdp_triplet_synapse': 'stdp-tau-minus',
    'stdp_nn_symm_synapse': 'stdp-nn-symm',
    'stdp_nn_restr_synapse': 'stdp-nn-restr',
    'stdp_nn_pre_centered_synapse': 'stdp-nn-pre-centered',
    'stdp_facetshw_synapse_hom': 'stdp-facetshw',
    'stdp_dopamine_synapse': 'stdp-dopamine',
    'clopath_synapse': 'stdp-param-location',
}


def _specs():
    return {name: getattr(bps, name) for name in _SPEC_REF}


class CrossLinkTest(unittest.TestCase):
    def test_every_stdp_docstring_has_parity_link(self):
        """Every target spec's docstring carries the :doc: parity link."""
        missing = [name for name, cls in _specs().items()
                   if _DOC_LINK not in (cls.__doc__ or '')]
        self.assertEqual(missing, [], f"missing parity :doc: link in docstrings: {missing}")

    @unittest.skipUnless(_PAGE.exists(), 'divergence page absent (installed without docs)')
    def test_parity_refs_resolve(self):
        """Every :ref: a docstring points at is defined in the page (no dangling)."""
        text = _PAGE.read_text(encoding='utf-8')
        defined = set(re.findall(r'^\.\. _([\w-]+):', text, flags=re.MULTILINE))
        specs = _specs()
        for name, label in _SPEC_REF.items():
            with self.subTest(spec=name):
                self.assertIn(label, defined, f"dangling :ref:`{label}` (not defined in page)")
                self.assertIn(f':ref:`{label}`', specs[name].__doc__ or '',
                              f"{name} docstring does not cite :ref:`{label}`")

    @unittest.skipUnless(_PAGE.exists(), 'divergence page absent (installed without docs)')
    def test_page_doctests_pass(self):
        """The page's runnable (brainpy.state) doctests execute cleanly."""
        results = doctest.testfile(
            str(_PAGE), module_relative=False,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE, verbose=False,
        )
        self.assertEqual(results.failed, 0, f"{results.failed} doctest failure(s) on {_PAGE.name}")


if __name__ == '__main__':
    unittest.main()
