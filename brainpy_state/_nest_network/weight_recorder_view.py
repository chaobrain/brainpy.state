# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Thin send-view over a per-step weight trajectory — the NEST ``weight_recorder`` analogue.

NEST's ``weight_recorder`` logs one event per plastic ``send`` (a presynaptic spike
on the connection), with the weight value as of send-time (the post-``send`` weight).
The repo records the dense per-step weight trajectory as an analog State-tap
(:meth:`~brainpy_state.network.Simulator.record_weight` /
:meth:`~brainpy_state.network.SimulationResult.weight_trace`, and the parity drives'
``bp_weight_trace``). This module masks that trajectory to its **send steps**,
reproducing the recorder's ``(time, weight)`` event series with no imperative device
(weight recording reuses the analog State-tap, not a hook).

The two helpers compose: :func:`send_steps_from_pre` turns a presynaptic spike train
into the per-edge send mask, and :func:`weight_recorder_events` samples the weight
trajectory at those steps.
"""
from __future__ import annotations

import numpy as np

__all__ = ['send_steps_from_pre', 'weight_recorder_events']


def send_steps_from_pre(pre_spikes, pre_of_edge=None, *, lag=0):
    """Step indices where each edge's presynaptic neuron fired (the send mask).

    Parameters
    ----------
    pre_spikes : array_like
        ``(T,)`` single-pre spike train, or ``(T, n_pre)`` population matrix (``1``/``0``
        or boolean). A *send* is a step where the edge's presynaptic neuron fired.
    pre_of_edge : array_like of int, optional
        ``(E,)`` population-local presynaptic index per edge in CSR (sorted-by-pre)
        order, i.e. ``proj.pre_local_idx[proj._pre_idx]``. ``None`` uses a ``(T,)``
        train (or the lone column of a ``(T, 1)`` matrix) for every edge.
    lag : int, optional
        Offset added to each fire step to align with the weight trajectory. The full
        :class:`~brainpy_state.network.Simulator` reads the per-population spike holder
        one step late, so a presynaptic spike at step ``s`` shows up in
        ``weight_trace[s + lag]`` (``lag=1``); the direct-feed parity drives feed the
        projection in-step (``lag=0``, the default). Shifted steps outside ``[0, T)``
        are clipped.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        ``(n_send,)`` integer array for a single presynaptic train (``pre_of_edge`` is
        ``None``), otherwise a length-``E`` list of per-edge integer arrays (CSR order).

    Raises
    ------
    ValueError
        If ``pre_spikes`` is a multi-column ``(T, n_pre)`` matrix and ``pre_of_edge``
        is not supplied.

    See Also
    --------
    weight_recorder_events : Sample a weight trajectory at the send steps.

    Examples
    --------
    .. code-block:: python

       >>> import numpy as np
       >>> from brainpy_state._nest_network import send_steps_from_pre
       >>> pre = np.zeros(8); pre[[2, 5]] = 1
       >>> send_steps_from_pre(pre).tolist()
       [2, 5]
    """
    arr = np.asarray(pre_spikes)
    T = arr.shape[0]

    def _fire_steps(col):
        s = np.flatnonzero(np.asarray(col) > 0) + int(lag)
        return s[(s >= 0) & (s < T)]

    if arr.ndim == 1:
        return _fire_steps(arr)
    if pre_of_edge is None:
        if arr.shape[1] == 1:
            return _fire_steps(arr[:, 0])
        raise ValueError(
            'pre_of_edge is required for a multi-pre (T, n_pre) spike train; pass the '
            'per-edge presynaptic index proj.pre_local_idx[proj._pre_idx].')
    return [_fire_steps(arr[:, int(p)]) for p in np.asarray(pre_of_edge)]


def weight_recorder_events(weight_trace, send_steps):
    """Per-send weight events over a weight trajectory (the thin send-view).

    NEST's ``weight_recorder`` logs one event per plastic ``send``, value = the weight
    at send-time (our post-``send`` ``weight_trace`` value). This masks the dense
    per-step trajectory to the send steps.

    Parameters
    ----------
    weight_trace : array_like
        ``(T,)`` or ``(T, E)`` per-step weight (or delivered amplitude) trajectory,
        sampled **post-update** (the order NEST logs: after this send's pairing update).
    send_steps : numpy.ndarray or list of numpy.ndarray
        ``(n_send,)`` integer send-step indices (e.g. from :func:`send_steps_from_pre`),
        or a length-``E`` list of per-edge step arrays masking each column of a
        ``(T, E)`` trace independently (multapses recording in CSR order).

    Returns
    -------
    tuple or list
        ``(steps, weights)`` with ``steps == send_steps`` and
        ``weights == weight_trace[steps]`` (``(n_send,)`` for a 1-D trace, or
        ``(n_send, E)`` for shared steps over a 2-D trace); a list of
        ``(steps_e, weights_e)`` per edge when ``send_steps`` is a per-edge list.

    Raises
    ------
    ValueError
        If ``send_steps`` is a per-edge list but ``weight_trace`` is not 2-D, or the
        list length does not match the number of trace columns.

    See Also
    --------
    send_steps_from_pre : Build the send mask from a presynaptic spike train.

    Notes
    -----
    A weight change strictly between the last send and the run end is **absent** from
    the events (NEST's recorder logs only at sends, so it misses it too); read it from
    ``weight_trace[-1]`` / the final connection weight instead.

    Examples
    --------
    .. code-block:: python

       >>> import numpy as np
       >>> from brainpy_state._nest_network import weight_recorder_events
       >>> trace = np.array([5.0, 4.0, 4.0, 3.0])
       >>> steps, w = weight_recorder_events(trace, np.array([1, 3]))
       >>> steps.tolist(), w.tolist()
       ([1, 3], [4.0, 3.0])
    """
    wt = np.asarray(weight_trace)
    if isinstance(send_steps, (list, tuple)):
        if wt.ndim != 2:
            raise ValueError('per-edge send_steps requires a (T, E) weight_trace')
        if len(send_steps) != wt.shape[1]:
            raise ValueError(
                f'per-edge send_steps has {len(send_steps)} edges but the weight_trace '
                f'has {wt.shape[1]} columns')
        out = []
        for e, steps_e in enumerate(send_steps):
            s = np.asarray(steps_e, dtype=int)
            out.append((s, wt[s, e]))
        return out
    steps = np.asarray(send_steps, dtype=int)
    return steps, wt[steps]
