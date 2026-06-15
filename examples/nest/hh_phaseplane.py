# examples/nest/hh_phaseplane.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Phase-plane analysis of the Hodgkin–Huxley neuron — NEST-style port.

Port of NEST's ``hh_phaseplane.py``. This is a pedagogical **analysis** demo, not
a spike-train parity demo: the eight-dimensional ``hh_psc_alpha`` dynamics are
reduced to the ``V``–``n`` plane by clamping the fast gating variables ``m`` and
``h`` to their resting equilibria (``m_eq``, ``h_eq``) and setting synaptic
currents to zero. On a grid of ``(V, n)`` the analytic vector field
``(dV/dt, dn/dt)`` is evaluated directly from the model's right-hand side; the
``V``- and ``n``-nullclines are the per-``V`` loci where ``dV/dt`` resp. ``dn/dt``
vanish; and one trajectory is traced through the *full* integrator (re-clamping
``m``, ``h`` each step, exactly as NEST does). Because ``m`` is frozen at its
small resting value the regenerative Na upstroke is disabled, so this reduced
``(V, n)`` trajectory relaxes toward the resting fixed point rather than firing a
full action potential — the reduced two-dimensional system is not excitable.

Differences from NEST's ``hh_phaseplane.py``, all deliberate:

* The vector field is the **analytic RHS** ``(dV/dt, dn/dt)`` rather than a
  one-step ``Simulate`` finite difference. The RHS *is* the phase-plane vector
  field (the nullclines are its zero-loci); a single evaluation per grid point is
  exact, fast, and free of the adaptive-integrator stiffness/overflow that a
  one-step probe hits at extreme grid corners.
* The ~5000-point grid scan is **vectorized** over a population rather than
  looping one neuron over grid points.
* Nullcline extraction reduces over each fixed-``V`` **column** with proper
  interior bounds. NEST's script has two latent indexing bugs here
  (``V_matrix[:][i]`` selects a fixed-``n`` *row*, and ``index != len(n_vec)`` can
  never be false); the plot then draws only ``nullcline[0]``. This port fixes all
  three so the nullclines are correct.

Run:  python examples/nest/hh_phaseplane.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import functools

import numpy as np
import jax.numpy as jnp
import brainunit as u
from brainstate.util import DotDict

from brainpy_state import hh_psc_alpha

brainstate.environ.set(dt=0.1 * u.ms)

#: NEST ``hh_psc_alpha`` defaults (the model under analysis).
MODEL_PARAMS = dict(
    E_L=-54.402 * u.mV, C_m=100.0 * u.pF,
    g_Na=12000.0 * u.nS, g_K=3600.0 * u.nS, g_L=30.0 * u.nS,
    E_Na=50.0 * u.mV, E_K=-77.0 * u.mV, t_ref=2.0 * u.ms,
    tau_syn_ex=0.2 * u.ms, tau_syn_in=2.0 * u.ms,
    V_m_init=-65.0 * u.mV,
)

# Grid (NEST hh_phaseplane defaults).
V_MIN, V_MAX, DELTA_V = -100.0, 42.0, 2.0
N_MIN, N_MAX, DELTA_N = 0.1, 0.81, 0.01
AMPLITUDE = 100.0  # external DC current (pA)


@functools.lru_cache(maxsize=1)
def equilibrium_mh():
    """Resting equilibrium of the fast gating variables ``(m_eq, h_eq)``.

    Obtained exactly as NEST's ``hh_phaseplane`` does — by relaxing an
    unperturbed neuron for 1000 ms and reading the settled ``Act_m`` / ``Inact_h``
    (memoized; the relaxation runs once per process).

    Returns
    -------
    m_eq, h_eq : float
        ``Act_m`` and ``Inact_h`` at the resting potential.
    """
    brainstate.environ.set(dt=0.1 * u.ms)
    neuron = hh_psc_alpha(1, **MODEL_PARAMS)
    brainstate.nn.init_all_states(neuron)

    def relax(k):
        with brainstate.environ.context(t=k * 0.1 * u.ms):
            neuron.update()
        return 0

    brainstate.transform.for_loop(relax, jnp.arange(10000))
    return float(neuron.m.value[0]), float(neuron.h.value[0])


def grids():
    """Build the phase-plane scan axes.

    Returns
    -------
    V_vec : numpy.ndarray
        Membrane-potential samples (mV mantissa) from ``V_MIN`` to ``V_MAX``
        spaced by ``DELTA_V``.
    n_vec : numpy.ndarray
        K-activation samples (dimensionless) from ``N_MIN`` to ``N_MAX`` spaced
        by ``DELTA_N``.
    """
    return (np.arange(V_MIN, V_MAX, DELTA_V), np.arange(N_MIN, N_MAX, DELTA_N))


def vector_field(amplitude=AMPLITUDE):
    """Analytic phase-plane vector field over the ``(V, n)`` grid.

    Evaluates the model RHS ``(dV/dt, dn/dt)`` at every grid point with the fast
    gates clamped to their resting equilibria (``m_eq``, ``h_eq``) and synaptic
    currents zero — the dimensional reduction to the ``V``–``n`` plane.

    Parameters
    ----------
    amplitude : float, optional
        External DC current in pA. Default :data:`AMPLITUDE`.

    Returns
    -------
    V_vec, n_vec : numpy.ndarray
        The scan axes.
    dVdt : numpy.ndarray
        ``dV/dt`` in mV/ms, shape ``(n_n, n_v)`` (row = ``n``, col = ``V``).
    dndt : numpy.ndarray
        ``dn/dt`` in 1/ms, same shape.
    """
    V_vec, n_vec = grids()
    m_eq, h_eq = equilibrium_mh()
    VV, NN = np.meshgrid(V_vec, n_vec)          # (n_n, n_v)
    flatV, flatN = VV.ravel(), NN.ravel()
    N = flatV.size

    neuron = hh_psc_alpha(N, I_e=amplitude * u.pA, **MODEL_PARAMS)
    brainstate.nn.init_all_states(neuron)
    z = jnp.zeros(N)
    state = DotDict(
        V=jnp.asarray(flatV) * u.mV,
        m=jnp.full(N, m_eq), h=jnp.full(N, h_eq), n=jnp.asarray(flatN),
        dI_ex=z * (u.pA / u.ms), I_ex=z * u.pA,
        dI_in=z * (u.pA / u.ms), I_in=z * u.pA,
    )
    deriv = neuron._vector_field(state, DotDict(i_stim=z * u.pA))
    dVdt = np.asarray(u.get_mantissa(deriv.V / (u.mV / u.ms)))
    dndt = np.asarray(u.get_mantissa(deriv.n * u.ms))
    return V_vec, n_vec, dVdt.reshape(VV.shape), dndt.reshape(VV.shape)


def nullclines(V_vec, n_vec, dVdt, dndt):
    """Per-``V`` nullcline loci: ``n`` where ``dV/dt`` resp. ``dn/dt`` vanish.

    Interior minima of ``|dV/dt|`` / ``|dn/dt|`` only (endpoint columns of the
    ``n`` grid are skipped, matching NEST), so each returned point brackets a
    genuine in-range zero crossing.

    Returns
    -------
    nc_V, nc_n : numpy.ndarray
        ``(k, 2)`` arrays of ``(V, n)`` points on the V- and n-nullclines.
    """
    aV, an = np.abs(dVdt), np.abs(dndt)
    nc_V, nc_n = [], []
    for i in range(V_vec.size):
        j = int(np.nanargmin(aV[:, i]))
        if 0 < j < n_vec.size - 1:
            nc_V.append((V_vec[i], n_vec[j]))
        j = int(np.nanargmin(an[:, i]))
        if 0 < j < n_vec.size - 1:
            nc_n.append((V_vec[i], n_vec[j]))
    return np.asarray(nc_V), np.asarray(nc_n)


def ap_trajectory(amplitude=AMPLITUDE, dt=0.1, n_steps=1000, V0=-34.0, n0=0.2):
    """Trace one reduced-plane trajectory (``m``, ``h`` re-clamped each step).

    Drives the *full* ``hh_psc_alpha`` integrator (so the orbit is the true model
    dynamics), re-clamping ``m`` and ``h`` to their resting equilibria before each
    step to keep it in the reduced ``(V, n)`` plane — exactly NEST's procedure
    (the ``ap`` loop of ``hh_phaseplane.py``). With ``m`` frozen the orbit relaxes
    to the resting fixed point; it does not fire (see module docstring).

    Parameters
    ----------
    amplitude : float, optional
        External DC current in pA. Default :data:`AMPLITUDE`.
    dt : float, optional
        Step length in ms. Default ``0.1``.
    n_steps : int, optional
        Number of steps. Default ``1000``.
    V0, n0 : float, optional
        Seed state, NEST's choice. Default ``(-34.0 mV, 0.2)``.

    Returns
    -------
    numpy.ndarray
        ``(n_steps, 2)`` array of ``(V_m [mV], n)`` recorded before each step.
    """
    brainstate.environ.set(dt=dt * u.ms)
    m_eq, h_eq = equilibrium_mh()
    neuron = hh_psc_alpha(1, I_e=amplitude * u.pA, **MODEL_PARAMS)
    brainstate.nn.init_all_states(neuron)
    neuron.V.value = jnp.array([V0]) * u.mV
    neuron.n.value = jnp.array([n0])

    def step(k):
        neuron.m.value = jnp.array([m_eq])    # re-clamp the fast gates
        neuron.h.value = jnp.array([h_eq])
        V = u.get_mantissa(neuron.V.value / u.mV)[0]
        n = neuron.n.value[0]
        with brainstate.environ.context(t=k * dt * u.ms):
            neuron.update()
        return jnp.stack([V, n])

    ap = brainstate.transform.for_loop(step, jnp.arange(n_steps))
    return np.asarray(ap)


def main():
    print("hh_phaseplane (brainpy.state) — V–n phase-plane analysis")
    V_vec, n_vec, dVdt, dndt = vector_field()
    nc_V, nc_n = nullclines(V_vec, n_vec, dVdt, dndt)
    ap = ap_trajectory()
    print(f"  grid {dVdt.shape}; V-nullcline {nc_V.shape[0]} pts, "
          f"n-nullcline {nc_n.shape[0]} pts")
    print(f"  trajectory from ({ap[0, 0]:.0f} mV, {ap[0, 1]:.2f}): V "
          f"{ap[:, 0].min():.1f}..{ap[:, 0].max():.1f} mV, relaxes to "
          f"{ap[-1, 0]:.1f} mV")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        VV, NN = np.meshgrid(V_vec, n_vec)
        # Normalize arrows for a readable field (the two components differ by
        # ~10^4 in scale: dV/dt ~ 100s mV/ms, dn/dt ~ 0.01 /ms).
        U = dVdt / (np.abs(dVdt).max() + 1e-12)
        W = dndt / (np.abs(dndt).max() + 1e-12)
        plt.quiver(VV, NN, U, W, color=[0.6, 0.6, 0.6],
                   angles="xy", pivot="mid", width=0.002)
        if nc_V.size:
            plt.plot(nc_V[:, 0], nc_V[:, 1], lw=2.0, label="V-nullcline (dV/dt=0)")
        if nc_n.size:
            plt.plot(nc_n[:, 0], nc_n[:, 1], lw=2.0, label="n-nullcline (dn/dt=0)")
        plt.plot(ap[:, 0], ap[:, 1], "k", lw=1.0, label="trajectory (relaxation)")
        plt.xlim(V_vec[0], V_vec[-1]); plt.ylim(n_vec[0], n_vec[-1])
        plt.xlabel("membrane potential V (mV)"); plt.ylabel("K activation n")
        plt.title("Phase space of the Hodgkin–Huxley neuron"); plt.legend()
        plt.tight_layout()
        plt.savefig("examples/nest/hh_phaseplane.png", dpi=100)
        print("  wrote examples/nest/hh_phaseplane.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
