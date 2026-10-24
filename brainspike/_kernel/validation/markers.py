# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Shared test markers of the K0 gate suite.

* :data:`requires_brian2` -- skip when Brian2 is not importable and tag the test with
  the ``requires_brian2`` pytest marker (``pytest -m requires_brian2``), the Brian2
  counterpart of :func:`brainspike._nest_validation.nest_compare.requires_nest`.
  Brian2 is the ``validation`` optional extra of the package.
"""
from __future__ import annotations

import importlib.util

import pytest

__all__ = ['HAS_BRIAN2', 'requires_brian2']

#: Whether Brian2 can be imported (checked without importing it).
HAS_BRIAN2 = importlib.util.find_spec('brian2') is not None


def requires_brian2(obj):
    """Skip a test (function or class) when Brian2 is missing; attach ``requires_brian2``.

    Parameters
    ----------
    obj : callable or type
        Test function, method or class.

    Returns
    -------
    callable or type
        ``obj``, marked and conditionally skipped.

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.validation.markers import requires_brian2
        >>> @requires_brian2
        ... def test_vs_brian2():
        ...     import brian2
        >>> [m.name for m in test_vs_brian2.pytestmark]
        ['skipif', 'requires_brian2']
    """
    obj = pytest.mark.skipif(not HAS_BRIAN2, reason='brian2 not installed (validation extra)')(obj)
    return pytest.mark.requires_brian2(obj)
