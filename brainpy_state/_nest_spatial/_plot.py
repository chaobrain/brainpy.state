# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Spatial plotting helpers (NEST ``PlotLayer`` / ``PlotTargets`` / ``PlotSources`` /
``PlotProbabilityParameter``).

matplotlib is an *optional* dependency: it is imported lazily inside each function (never at
module import) so that the rest of ``brainpy.state.spatial`` works without it. Each helper returns
the :class:`matplotlib.figure.Figure` it drew on, so callers can further annotate or save it.
"""
from __future__ import annotations

import numpy as np
import brainunit as u

from brainpy_state._nest_spatial._distance import pairwise_distance
from brainpy_state._nest_spatial._helpers import target_positions
from brainpy_state._nest_spatial._layers import _LEN, _as_len

__all__ = ['plot_layer', 'plot_targets', 'plot_sources', 'plot_probability_parameter']


def _import_mpl():
    """Import :mod:`matplotlib.pyplot`, raising a clear error if matplotlib is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            'spatial plotting requires matplotlib; install it with `pip install matplotlib`.'
        ) from exc
    return plt


def _axes(plt, fig, ndim):
    """Return ``(fig, ax)``: reuse ``fig``'s first axis, else add one with the right projection."""
    if fig is None:
        fig = plt.figure()
    if fig.axes:
        ax = fig.axes[0]
    else:
        ax = fig.add_subplot(111, projection='3d' if ndim == 3 else None)
    return fig, ax


def _scatter(ax, coords, ndim, color, size):
    """Scatter an ``(n, ndim)`` magnitude array on a 2-D or 3-D axis."""
    ax.scatter(*(coords[:, i] for i in range(ndim)), c=color, s=size)


def _mag(coords):
    """Bare micrometre magnitudes of a coordinate Quantity."""
    return np.asarray(u.get_magnitude(coords.to(_LEN)))


def plot_layer(layer, fig=None, nodecolor='b', nodesize=20):
    """Scatter a layer's node positions (NEST ``PlotLayer``).

    Parameters
    ----------
    layer : Layer
        A concrete 2-D / 3-D position layer.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on; a new one is created when ``None``.
    nodecolor : color, optional
        Marker colour. Default ``'b'``.
    nodesize : float, optional
        Marker size. Default ``20``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy import state as bp
        >>> fig = bp.spatial.plot_layer(bp.spatial.grid([10, 10]))   # doctest: +SKIP
    """
    plt = _import_mpl()
    coords = _mag(layer.coords)
    fig, ax = _axes(plt, fig, layer.ndim)
    _scatter(ax, coords, layer.ndim, nodecolor, nodesize)
    return fig


def plot_targets(sim, src_node, target, fig=None,
                 src_color='red', src_size=50, tgt_color='b', tgt_size=20):
    """Highlight one source node's realized targets (NEST ``PlotTargets``).

    Parameters
    ----------
    sim : Simulator
        The simulator holding the realized connections.
    src_node : NodeView
        The source node whose targets are drawn (the first node is used if it spans several).
    target : NodeView
        The candidate-target population (created with ``positions=``).
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on.
    src_color, tgt_color : color, optional
        Marker colours for the source node and its targets.
    src_size, tgt_size : float, optional
        Marker sizes for the source node and its targets.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    plt = _import_mpl()
    tgt_coords = _mag(target_positions(sim, src_node, target)[0])
    src_coords = _mag(sim.get_position(src_node))
    ndim = src_coords.shape[1]
    fig, ax = _axes(plt, fig, ndim)
    if tgt_coords.size:
        _scatter(ax, tgt_coords, ndim, tgt_color, tgt_size)
    _scatter(ax, src_coords, ndim, src_color, src_size)
    return fig


def plot_sources(sim, source, tgt_node, fig=None,
                 src_color='b', src_size=20, tgt_color='red', tgt_size=50):
    """Highlight one target node's realized sources (NEST ``PlotSources``).

    Parameters
    ----------
    sim : Simulator
        The simulator holding the realized connections.
    source : NodeView
        The candidate-source population (created with ``positions=``).
    tgt_node : NodeView
        The target node whose sources are drawn.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on.
    src_color, tgt_color : color, optional
        Marker colours for the sources and the target node.
    src_size, tgt_size : float, optional
        Marker sizes for the sources and the target node.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    plt = _import_mpl()
    sc = sim.get_connections(source=source, target=tgt_node)
    src_idx = np.unique(np.asarray(sc.source))
    src_coords = _mag(sim.get_position(source))[src_idx]
    tgt_coords = _mag(sim.get_position(tgt_node))
    ndim = tgt_coords.shape[1]
    fig, ax = _axes(plt, fig, ndim)
    if src_coords.size:
        _scatter(ax, src_coords, ndim, src_color, src_size)
    _scatter(ax, tgt_coords, ndim, tgt_color, tgt_size)
    return fig


def plot_probability_parameter(kernel, mask=None, extent=(-0.5, 0.5, -0.5, 0.5),
                               shape=(100, 100), fig=None, cmap='Greys'):
    """Heatmap of a connection kernel ``p(d)`` over a 2-D field (NEST ``PlotProbabilityParameter``).

    The kernel is evaluated for a single source at the origin against a regular grid of target
    positions spanning ``extent``. When a ``mask`` is given the probability is zeroed outside it.

    Parameters
    ----------
    kernel : object
        A spatial kernel/expression (``_eval_pair``) or a bare callable ``p(distance)``.
    mask : object, optional
        A spatial mask whose ``contains`` zeroes the probability outside it.
    extent : tuple of float, optional
        ``(x_min, x_max, y_min, y_max)`` of the sampled field (micrometres). Default the unit box.
    shape : tuple of int, optional
        ``(nx, ny)`` sample counts. Default ``(100, 100)``.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on.
    cmap : str, optional
        Colormap name. Default ``'Greys'``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    plt = _import_mpl()
    x_min, x_max, y_min, y_max = extent
    xs = np.linspace(x_min, x_max, shape[0])
    ys = np.linspace(y_min, y_max, shape[1])
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    pre = _as_len(np.array([[0.0, 0.0]]))                  # single source at origin
    post = _as_len(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    if hasattr(kernel, '_eval_pair'):
        prob = u.get_magnitude(kernel._eval_pair(pre, post))
    else:
        prob = u.get_magnitude(kernel(pairwise_distance(pre, post)))
    prob = np.asarray(prob).reshape(grid_x.shape)
    if mask is not None:
        inside = np.asarray(mask.contains(pre, post)).reshape(grid_x.shape)
        prob = np.where(inside, prob, 0.0)
    fig, ax = _axes(plt, fig, 2)
    ax.imshow(prob, origin='lower', extent=extent, cmap=cmap, aspect='auto')
    return fig
