# examples/nest_like/csa_example.py
"""CSA (Connection Set Algebra) demo -- documented placeholder + native equivalent.

NEST's ``csa_example`` builds connectivity through the **Connection Set Algebra** using the
``conngen`` connection rule, which requires NEST to be compiled against ``libneurosim``::

    cg = csa.cset(csa.random(0.1), 10000.0, 1.0)          # random p=0.1, weight, delay
    nest.Connect(pre, post, {"rule": "conngen", "cg": cg, "params_map": {...}})

**The CSA / conngen mechanism is intentionally not ported.** It is a NEST-specific
integration with an external library (CSA + libneurosim) for *simulator-independent*
connectivity descriptions; brainpy.state has no ``conngen`` rule and no CSA dependency.

This is not a capability gap, though: the *connectivity* a CSA expression describes is
expressed natively by brainpy.state's connection rules.

* ``csa.random(0.1)`` (this demo) is exactly :func:`~brainpy.state.pairwise_bernoulli` with
  ``p=0.1`` -- demonstrated by :func:`run` below.
* ``csa.random * (csa.gaussian(sigma, cutoff) * d)`` (the *spatial* CSA demo,
  ``csa_spatial_example``) is a distance-dependent Gaussian draw, ported natively in
  ``examples/nest_like/spatial_csa.py`` with
  ``spatial.spatial_pairwise_bernoulli(p=spatial.gaussian(spatial.distance, std=sigma),
  mask=spatial.circular(cutoff))``.

So the recommended replacement for CSA-built connectivity is the corresponding native rule;
this file documents that mapping and demonstrates the non-spatial case concretely.

See Also
--------
examples/nest_like/spatial_csa.py : the spatial CSA demo ported with the native spatial rule.

Run:  PYTHONPATH=. python examples/nest_like/csa_example.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(n=16, p=0.1, seed=0):
    """Native equivalent of ``csa.random(0.1)``: a ``pairwise_bernoulli`` connect.

    Parameters
    ----------
    n : int, optional
        Size of the pre- and postsynaptic populations. Default ``16`` (NEST's value).
    p : float, optional
        Connection probability. Default ``0.1`` (the ``csa.random(0.1)`` value).
    seed : int, optional
        Connectivity-sampling seed. Default ``0``.

    Returns
    -------
    density : float
        Realized connection density ``n_edges / n**2`` (≈ ``p`` for large ``n``).

    Examples
    --------
    .. code-block:: python

        >>> density = run(n=200, p=0.1)
        >>> bool(0.07 < density < 0.13)        # ~ p, up to sampling noise
        True
    """
    sim = bp.Simulator(dt=0.1 * u.ms)
    pre = sim.create(bp.iaf_psc_alpha, n)
    post = sim.create(bp.iaf_psc_alpha, n)
    sim.connect(pre, post, rule=bp.pairwise_bernoulli(p),
                weight=10000.0 * u.pA, delay=1.0 * u.ms, seed=seed)
    sc = sim.get_connections(source=pre, target=post)
    return len(sc) / float(n * n)


def main():
    density = run(n=200)
    print('csa_example (brainpy.state -- CSA conngen not ported; native pairwise_bernoulli)')
    print(f'  csa.random(0.1) ~= pairwise_bernoulli(0.1); realized density = {density:.3f}')
    print('  for spatial CSA connectivity, see examples/nest_like/spatial_csa.py')


if __name__ == '__main__':
    main()
