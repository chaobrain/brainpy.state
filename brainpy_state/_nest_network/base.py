# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.

import brainstate
import brainunit as u

from brainpy_state._base import Neuron
from brainpy_state._brainpy.projection import Projection
from brainpy_state._nest_base.base import NESTDevice

__all__ = ['Network']


class Network(brainstate.nn.Module):
    """brainpy.state network base class.

    Subclass and define populations, projections, and devices as
    attributes. ``update()`` walks the immediate module-tree children
    in projection-first then dynamics order.
    """
    __module__ = 'brainpy.state'

    def update(self, t=None) -> None:
        # Depth-1 traversal — matches the existing brainpy.state convention
        # documented in brainpy_state/_brainpy/projection.py:46-51. Networks
        # that need nested projection chains (Projection containing
        # Projection) currently must override update() explicitly; this is
        # tracked as an open question in the design spec §12.
        children = self.nodes(allowed_hierarchy=(1, 1))
        projections = [m for m in children.values() if isinstance(m, Projection)]
        others = [m for m in children.values() if not isinstance(m, Projection)]
        for m in projections:
            m()
        for m in others:
            m()

    @property
    def populations(self) -> dict[str, Neuron]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Neuron)}

    @property
    def projections(self) -> dict[str, Projection]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, Projection)}

    @property
    def devices(self) -> dict[str, NESTDevice]:
        return {k[-1]: m for k, m in self.nodes(allowed_hierarchy=(1, 1)).items()
                if isinstance(m, NESTDevice)}

    def simulate(self, duration, *, dt=None, monitor=None) -> dict:
        """Run the network for ``duration``.

        Wraps ``brainstate.transform.for_loop`` over ``self.update``.

        Parameters
        ----------
        duration : brainunit.Quantity
            Wall-clock time to simulate.
        dt : brainunit.Quantity, optional
            Timestep override. Defaults to ``brainstate.environ.get('dt')``.
        monitor : list[str] | dict[str, Callable] | None
            Per-step recording specification:

            - ``None`` (default) — no monitoring; the function returns ``{}``.
            - list of dotted attribute paths (e.g. ``['exc.spike']``) — stacked
              across time and returned under that key.
            - dict of ``name -> callable(net)``: callable evaluated each step,
              result stacked under ``name``.
        """
        import brainstate.transform as transform
        if dt is None:
            dt = brainstate.environ.get('dt')
        if dt is None:
            raise ValueError(
                'dt must be set via brainstate.environ.set(dt=...) or '
                'passed explicitly as simulate(..., dt=...)'
            )
        times = u.math.arange(0.0 * u.get_unit(dt), duration, dt)
        indices = u.math.arange(times.size)

        # Normalize monitor to a dict of callables.
        callables = {}
        if monitor is None:
            pass
        elif isinstance(monitor, (list, tuple)):
            for path in monitor:
                callables[path] = self._make_path_callable(path)
        elif isinstance(monitor, dict):
            for name, fn in monitor.items():
                if not callable(fn):
                    raise TypeError(
                        f'monitor dict value for {name!r} must be callable, '
                        f'got {type(fn).__name__}'
                    )
                callables[name] = fn
        else:
            raise TypeError(
                f'monitor must be list, dict, or None, got {type(monitor).__name__}'
            )

        def step(t, i):
            with brainstate.environ.context(t=t, i=i):
                self.update(t)
                if callables:
                    return {k: fn(self) for k, fn in callables.items()}
                return None

        if callables:
            stacked = transform.for_loop(step, times, indices)
            return dict(stacked)
        transform.for_loop(step, times, indices)
        return {}

    def _make_path_callable(self, path: str):
        parts = path.split('.')

        def fn(net):
            obj = net
            for p in parts:
                obj = getattr(obj, p)
            if hasattr(obj, 'value'):
                obj = obj.value
            return obj
        return fn
