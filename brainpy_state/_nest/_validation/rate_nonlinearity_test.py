# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Per-φ nonlinearity parity + nonlinear steady-state — goal-15a φ correctness.

Each rate model's ``_activation`` *is* its NEST ``nonlinearities_*::input`` transfer
function. The closed forms below are transcribed from the NEST C++ sources
(``models/<model>.h``) and verified against them:

============  ===============================================================
model         φ(h)
============  ===============================================================
lin_rate      ``g·h``
gauss_rate    ``g·exp(−(h−μ)² / 2σ²)``
sigmoid_rate  ``g / (1 + exp(−β(h−θ)))``
tanh_rate     ``tanh(g·(h−θ))``
threshold     ``min(max(g·(h−θ), 0), α)``
sigmoid_gg    ``(g·h)⁴ / (0.1⁴ + (g·h)⁴)``
============  ===============================================================

Three arbiters:

* ``TestActivationClosedForm`` (NEST-free) — each ``_activation`` matches its closed
  form to machine precision over a swept input grid. Since the closed form is NEST's
  own ``input()`` (source-verified), this pins φ to NEST without a running kernel.
* ``TestPhiExtractionNestParity`` (``@requires_nest``) — drives a target with a constant
  input ``h`` (a ``λ=1, μ=0, σ=0`` cell relaxes to ``r* = φ(h)``) and matches the
  brainpy steady rate against both the closed form and live NEST, end-to-end through
  the seam-(H) coupling. Covers the deterministic-dynamics models (``gauss`` excluded:
  it reuses ``σ`` as the gain width, so it cannot be driven noise-free).
* ``TestNonlinearNetworkFixedPoint`` (NEST-free + ``@requires_nest``) — a recurrent
  ``tanh`` network relaxes to the numerically-solved nonlinear fixed point
  ``r = μ + φ(C·r)`` and matches live NEST.
"""
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import saiunit as u
import braintools

try:
    import nest
except Exception:
    nest = None

from brainpy_state import (Simulator, lin_rate_ipn, gauss_rate_ipn, sigmoid_rate_ipn,
                           tanh_rate_ipn, threshold_lin_rate_ipn, sigmoid_rate_gg_1998_ipn,
                           multimeter, one_to_one)
from brainpy_state._nest._validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest._validation.tolerance_conventions import TraceTolerance

DT = 0.1
TAU = 10.0
T = 1500.0

# --- NEST nonlinearities_*::input transcribed (models/<m>.h) ---------------------

def phi_lin(h, g=1.5):
    return g * h


def phi_gauss(h, g=1.2, mu=0.5, sigma=0.8):
    return g * np.exp(-np.square(h - mu) / (2.0 * sigma ** 2))


def phi_sigmoid(h, g=2.0, beta=1.5, theta=0.3):
    return g / (1.0 + np.exp(-beta * (h - theta)))


def phi_tanh(h, g=1.3, theta=0.2):
    return np.tanh(g * (h - theta))


def phi_threshold(h, g=1.1, theta=0.4, alpha=2.0):
    return np.minimum(np.maximum(g * (h - theta), 0.0), alpha)


def phi_sigmoid_gg(h, g=3.0):
    gh4 = (g * h) ** 4
    return gh4 / (0.1 ** 4 + gh4)


#: (label, class, φ-defining ctor kwargs, numpy reference φ, NEST model, NEST φ params).
#: ``nest_params`` is None for models excluded from the noise-free NEST sweep.
_MODELS = [
    ('lin_rate', lin_rate_ipn, dict(g=1.5), phi_lin,
     'lin_rate_ipn', dict(g=1.5)),
    ('gauss_rate', gauss_rate_ipn, dict(g=1.2, mu=0.5, sigma=0.8), phi_gauss,
     None, None),   # σ is the gain width AND the noise amplitude -> not noise-free
    ('sigmoid_rate', sigmoid_rate_ipn, dict(g=2.0, beta=1.5, theta=0.3), phi_sigmoid,
     'sigmoid_rate_ipn', dict(g=2.0, beta=1.5, theta=0.3)),
    ('tanh_rate', tanh_rate_ipn, dict(g=1.3, theta=0.2), phi_tanh,
     'tanh_rate_ipn', dict(g=1.3, theta=0.2)),
    ('threshold_lin_rate', threshold_lin_rate_ipn, dict(g=1.1, theta=0.4, alpha=2.0), phi_threshold,
     'threshold_lin_rate_ipn', dict(g=1.1, theta=0.4, alpha=2.0)),
    ('sigmoid_rate_gg_1998', sigmoid_rate_gg_1998_ipn, dict(g=3.0), phi_sigmoid_gg,
     'sigmoid_rate_gg_1998_ipn', dict(g=3.0)),
]

H_GRID = np.linspace(-2.0, 2.0, 41)


def _phi_params(phi_kwargs):
    """Deterministic cell params carrying the given φ parameters.

    ``sigma`` defaults to 0 (noise-free); gauss overrides it with its gain width via
    ``phi_kwargs``. ``mu`` (the intrinsic drive) stays at the ctor default 0 for the
    non-gauss models, so a constant-driven cell relaxes to exactly ``φ(h)``.
    """
    params = dict(tau=TAU * u.ms, lambda_=1.0, sigma=0.0, linear_summation=True,
                  rate_initializer=braintools.init.Constant(0.0),
                  noise_initializer=braintools.init.Constant(0.0))
    params.update(phi_kwargs)
    return params


def _make(cls, phi_kwargs):
    """Construct a standalone deterministic cell (for the no-Simulator φ unit test)."""
    return cls(1, **_phi_params(phi_kwargs))


class TestActivationClosedForm(unittest.TestCase):
    """Each ``_activation`` equals its NEST ``input()`` closed form (NEST-free)."""

    def test_each_phi_matches_closed_form(self):
        for label, cls, phi_kwargs, phi_ref, _, _ in _MODELS:
            with self.subTest(model=label):
                cell = _make(cls, phi_kwargs)
                got = np.asarray(u.get_mantissa(cell._activation(jnp.asarray(H_GRID)))).reshape(-1)
                want = phi_ref(H_GRID)
                np.testing.assert_allclose(
                    got, want, rtol=1e-12, atol=1e-12,
                    err_msg=f'{label}: _activation != NEST input() closed form')

    def test_threshold_saturates_and_floors(self):
        """threshold_lin clamps to ``[0, α]`` outside the linear band."""
        cell = _make(threshold_lin_rate_ipn, dict(g=1.1, theta=0.4, alpha=2.0))
        lo = float(u.get_mantissa(cell._activation(jnp.asarray(-5.0))))
        hi = float(u.get_mantissa(cell._activation(jnp.asarray(50.0))))
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 2.0, places=12)

    def test_tanh_is_odd_about_theta(self):
        """tanh φ is antisymmetric about θ: ``φ(θ+d) = −φ(θ−d)``."""
        cell = _make(tanh_rate_ipn, dict(g=1.3, theta=0.2))
        d = np.array([0.1, 0.5, 1.0, 2.0])
        up = np.asarray(u.get_mantissa(cell._activation(jnp.asarray(0.2 + d)))).reshape(-1)
        dn = np.asarray(u.get_mantissa(cell._activation(jnp.asarray(0.2 - d)))).reshape(-1)
        np.testing.assert_allclose(up, -dn, atol=1e-12)


class TestRateNeuronTemplateActivation(unittest.TestCase):
    """The ``rate_neuron`` template's φ: linear gain by default, the user callable otherwise."""

    def test_default_is_linear_gain(self):
        from brainpy_state import rate_neuron_ipn
        cell = rate_neuron_ipn(1, tau=TAU * u.ms, lambda_=1.0, g=1.7, sigma=0.0,
                               linear_summation=True,
                               rate_initializer=braintools.init.Constant(0.0),
                               noise_initializer=braintools.init.Constant(0.0))
        got = np.asarray(u.get_mantissa(cell._activation(jnp.asarray(H_GRID)))).reshape(-1)
        np.testing.assert_allclose(got, 1.7 * H_GRID, rtol=1e-12, atol=1e-12)

    def test_user_nonlinearity_is_honoured(self):
        from brainpy_state import rate_neuron_ipn
        cell = rate_neuron_ipn(1, tau=TAU * u.ms, lambda_=1.0, g=1.0, sigma=0.0,
                               input_nonlinearity=lambda self, h: u.math.tanh(2.0 * h),
                               linear_summation=True,
                               rate_initializer=braintools.init.Constant(0.0),
                               noise_initializer=braintools.init.Constant(0.0))
        got = np.asarray(u.get_mantissa(cell._activation(jnp.asarray(H_GRID)))).reshape(-1)
        np.testing.assert_allclose(got, np.tanh(2.0 * H_GRID), rtol=1e-10, atol=1e-12)


# --- NEST φ extraction (steady state of a constant-driven cell) -------------------

_SWEEP = {
    'lin_rate': np.array([-1.0, 0.0, 0.8, 1.5]),
    'sigmoid_rate': np.array([-1.0, 0.0, 0.3, 1.0, 3.0]),
    'tanh_rate': np.array([-1.0, 0.0, 0.2, 1.0, 2.0]),
    'threshold_lin_rate': np.array([-1.0, 0.4, 1.0, 3.0]),
    'sigmoid_rate_gg_1998': np.array([0.05, 0.1, 0.3, 1.0]),
}

PHI_TOL = TraceTolerance(1e-4, 1e-4, align_steps=6, label='C',
                         note='φ extracted as the steady state of a constant-driven rate cell')


def _bp_phi_steady(cls, phi_kwargs, h, *, T=T, dt=DT):
    """Steady rate of a ``cls`` target driven by a constant input ``h`` (= φ(h))."""
    sim = Simulator(dt=dt * u.ms)
    driver = sim.create(lin_rate_ipn, 1, params=dict(
        tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=float(h), g=1.0, linear_summation=True,
        rate_initializer=braintools.init.Constant(0.0),
        noise_initializer=braintools.init.Constant(0.0)))
    target = sim.create(cls, 1, params=_phi_params(phi_kwargs))
    sim.connect(driver, target, weight=1.0, rule=one_to_one, comm='dense')
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, target)
    res = sim.simulate(T * u.ms)
    return float(np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)[-1])


def _nest_phi_steady(model, phi_params, h, *, T=T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    driver = nest.Create('lin_rate_ipn', 1, params={
        'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': float(h), 'g': 1.0,
        'linear_summation': True, 'rate': 0.0})
    target = nest.Create(model, 1, params={
        'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': 0.0,
        'linear_summation': True, 'rate': 0.0, **phi_params})
    nest.Connect(driver, target,
                 syn_spec={'synapse_model': 'rate_connection_instantaneous', 'weight': 1.0})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, target)
    nest.Simulate(T)
    ev = mm.events
    order = np.argsort(np.asarray(ev['times']), kind='stable')
    return float(np.asarray(ev['rate'])[order][-1])


@requires_nest
class TestPhiExtractionNestParity(unittest.TestCase):
    """Steady ``r* = φ(h)`` matches the closed form *and* live NEST, model by model."""

    def test_phi_steady_state_matches_closed_form_and_nest(self):
        for label, cls, phi_kwargs, phi_ref, nest_model, nest_params in _MODELS:
            if nest_model is None:
                continue
            for h in _SWEEP[label]:
                with self.subTest(model=label, h=float(h)):
                    bp = _bp_phi_steady(cls, phi_kwargs, h)
                    want = float(phi_ref(np.asarray(h)))
                    self.assertAlmostEqual(bp, want, places=4,
                                           msg=f'{label}: steady {bp} != φ({h})={want}')
                    ns = _nest_phi_steady(nest_model, nest_params, h)
                    self.assertAlmostEqual(bp, ns, places=4,
                                           msg=f'{label}: brainpy {bp} != NEST {ns}')


# --- nonlinear recurrent network fixed point --------------------------------------

# A 2-neuron tanh network: r_i = mu_i + tanh(g·(sum_j C_ij r_j − θ)). C is a stable
# mixed-sign coupling; rho(gC) modest so fixed-point iteration converges.
TANH_G, TANH_THETA = 0.8, 0.0
_C = np.array([[0.0, 0.5],
               [-0.4, 0.0]])
_MU = np.array([0.6, -0.3])


def _tanh_fixed_point(C, mu, g=TANH_G, theta=TANH_THETA, iters=20000):
    """Numeric fixed point of ``r = mu + tanh(g·(C·r − θ))`` by damped iteration."""
    r = mu.astype(float).copy()
    for _ in range(iters):
        r_new = mu + np.tanh(g * (C @ r - theta))
        if np.max(np.abs(r_new - r)) < 1e-14:
            return r_new
        r = 0.5 * r + 0.5 * r_new
    return r


def _bp_tanh_network(C, mu, *, T=T, dt=DT):
    n = C.shape[0]
    sim = Simulator(dt=dt * u.ms)
    pops = [sim.create(tanh_rate_ipn, 1, params=dict(
        tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=float(mu[i]), g=TANH_G, theta=TANH_THETA,
        linear_summation=True, rate_initializer=braintools.init.Constant(0.0),
        noise_initializer=braintools.init.Constant(0.0))) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and C[i, j] != 0.0:
                sim.connect(pops[j], pops[i], weight=float(C[i, j]), rule=one_to_one, comm='dense')
    mms = [sim.create(multimeter, record_from=('rate',)) for _ in range(n)]
    for i in range(n):
        sim.connect(mms[i], pops[i])
    res = sim.simulate(T * u.ms)
    return np.array([float(np.asarray(u.get_mantissa(res.trace(mms[i], 'rate'))).reshape(-1)[-1])
                     for i in range(n)])


def _nest_tanh_network(C, mu, *, T=T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    n = C.shape[0]
    cells = [nest.Create('tanh_rate_ipn', 1, params={
        'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': float(mu[i]), 'g': TANH_G,
        'theta': TANH_THETA, 'linear_summation': True, 'rate': 0.0}) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and C[i, j] != 0.0:
                nest.Connect(cells[j], cells[i],
                             syn_spec={'synapse_model': 'rate_connection_instantaneous',
                                       'weight': float(C[i, j])})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    for c in cells:
        nest.Connect(mm, c)
    nest.Simulate(T)
    ev = mm.events
    senders, times, rate = (np.asarray(ev['senders']), np.asarray(ev['times']), np.asarray(ev['rate']))
    out = []
    for c in cells:
        m = senders == c.tolist()[0]
        out.append(rate[m][np.argsort(times[m], kind='stable')][-1])
    return np.array(out)


class TestNonlinearNetworkFixedPoint(unittest.TestCase):
    """A recurrent tanh network relaxes to the numeric nonlinear fixed point."""

    def test_reaches_numeric_fixed_point(self):
        rstar = _tanh_fixed_point(_C, _MU)
        final = _bp_tanh_network(_C, _MU)
        np.testing.assert_allclose(
            final, rstar, atol=1e-4,
            err_msg=f'tanh net relaxed {final} != numeric fixed point {rstar}')

    def test_fixed_point_is_genuinely_nonlinear(self):
        """The tanh solution differs from the linearised ``(I−gC)^{-1}(μ−gθ)`` estimate."""
        rstar = _tanh_fixed_point(_C, _MU)
        lin_est = np.linalg.solve(np.eye(2) - TANH_G * _C, _MU - TANH_G * TANH_THETA)
        self.assertGreater(float(np.max(np.abs(rstar - lin_est))), 1e-3)


@requires_nest
class TestNonlinearNetworkNestParity(unittest.TestCase):
    """The recurrent tanh network matches live NEST."""

    def test_network_steady_state_matches_nest(self):
        bp = _bp_tanh_network(_C, _MU)
        ns = _nest_tanh_network(_C, _MU)
        np.testing.assert_allclose(bp, ns, atol=1e-3,
                                   err_msg=f'tanh net brainpy {bp} != NEST {ns}')


# --- linear_summation=False (φ-emission) path, homogeneous φ -----------------------
# With linear_summation=False the *sender* emits ``φ(rate)`` (its ``phi_rate`` State)
# and the receiver integrates it directly. NEST instead applies the *receiver's* φ to
# the incoming rate in its event handler, so the two agree only for a homogeneous φ
# (the homogeneity guard enforces this). This exercises the phi_rate emission seam that
# linear_summation=True never touches. A driver (mu=M) -> target (mu=0) pair settles to
# ``r_target = w·φ(M)``.

_LS_FALSE = [
    ('sigmoid_rate', sigmoid_rate_ipn, dict(g=2.0, beta=1.5, theta=0.3), phi_sigmoid,
     'sigmoid_rate_ipn', dict(g=2.0, beta=1.5, theta=0.3)),
    ('tanh_rate', tanh_rate_ipn, dict(g=1.3, theta=0.2), phi_tanh,
     'tanh_rate_ipn', dict(g=1.3, theta=0.2)),
]
LSF_M, LSF_W = 1.0, 0.7


def _bp_ls_false_pair(cls, phi_kwargs, *, M=LSF_M, w=LSF_W, T=T, dt=DT):
    sim = Simulator(dt=dt * u.ms)
    drv = sim.create(cls, 1, params={**_phi_params(phi_kwargs), 'mu': float(M),
                                     'linear_summation': False})
    tgt = sim.create(cls, 1, params={**_phi_params(phi_kwargs), 'mu': 0.0,
                                     'linear_summation': False})
    sim.connect(drv, tgt, weight=w, rule=one_to_one, comm='dense')
    mm = sim.create(multimeter, record_from=('rate',))
    sim.connect(mm, tgt)
    res = sim.simulate(T * u.ms)
    return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)


def _nest_ls_false_pair(model, phi_params, *, M=LSF_M, w=LSF_W, T=T, dt=DT):
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus({'resolution': dt, 'use_wfr': False})
    drv = nest.Create(model, 1, params={'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': float(M),
                                        'linear_summation': False, 'rate': 0.0, **phi_params})
    tgt = nest.Create(model, 1, params={'tau': TAU, 'lambda': 1.0, 'sigma': 0.0, 'mu': 0.0,
                                        'linear_summation': False, 'rate': 0.0, **phi_params})
    nest.Connect(drv, tgt, syn_spec={'synapse_model': 'rate_connection_instantaneous', 'weight': w})
    mm = nest.Create('multimeter', params={'record_from': ['rate'], 'interval': dt})
    nest.Connect(mm, tgt)
    nest.Simulate(T)
    ev = mm.events
    order = np.argsort(np.asarray(ev['times']), kind='stable')
    return np.asarray(ev['rate'])[order]


class TestRateNeuronTemplateEquivalence(unittest.TestCase):
    """``rate_neuron`` with the default (linear) template reproduces ``lin_rate`` exactly.

    The template subclasses ``_lin_rate_base`` and overrides only φ (defaulting to the
    same linear gain ``g·h``), so a default-template cell driven through the seam-(H)
    coupling must be bit-identical to the corresponding ``lin_rate`` cell.
    """

    @staticmethod
    def _driven_trace(post_cls, *, opn):
        from brainpy_state import lin_rate_opn
        sim = Simulator(dt=DT * u.ms)
        common = dict(tau=TAU * u.ms, sigma=0.0, mu=0.3, g=1.4, linear_summation=True,
                      rate_initializer=braintools.init.Constant(0.0),
                      noise_initializer=braintools.init.Constant(0.0))
        if not opn:
            common['lambda_'] = 1.0
        driver = sim.create(lin_rate_ipn, 1, params=dict(
            tau=TAU * u.ms, lambda_=1.0, sigma=0.0, mu=1.5, g=1.0, linear_summation=True,
            rate_initializer=braintools.init.Constant(0.0),
            noise_initializer=braintools.init.Constant(0.0)))
        post = sim.create(post_cls, 1, params=common)
        sim.connect(driver, post, weight=0.5, rule=one_to_one, comm='dense')
        mm = sim.create(multimeter, record_from=('rate',))
        sim.connect(mm, post)
        res = sim.simulate(300.0 * u.ms)
        return np.asarray(u.get_mantissa(res.trace(mm, 'rate'))).reshape(-1)

    def test_ipn_template_matches_lin_rate_ipn(self):
        from brainpy_state import rate_neuron_ipn
        tmpl = self._driven_trace(rate_neuron_ipn, opn=False)
        lin = self._driven_trace(lin_rate_ipn, opn=False)
        np.testing.assert_allclose(tmpl, lin, atol=1e-12,
                                   err_msg='rate_neuron_ipn default template != lin_rate_ipn')

    def test_opn_template_matches_lin_rate_opn(self):
        from brainpy_state import rate_neuron_opn, lin_rate_opn
        tmpl = self._driven_trace(rate_neuron_opn, opn=True)
        lin = self._driven_trace(lin_rate_opn, opn=True)
        np.testing.assert_allclose(tmpl, lin, atol=1e-12,
                                   err_msg='rate_neuron_opn default template != lin_rate_opn')


class TestLinearSummationFalseEmission(unittest.TestCase):
    """The ``linear_summation=False`` φ-emission seam relaxes to ``w·φ(M)`` (NEST-free)."""

    def test_homogeneous_phi_false_summation_fixed_point(self):
        for label, cls, phi_kwargs, phi_ref, _, _ in _LS_FALSE:
            with self.subTest(model=label):
                want = float(LSF_W * phi_ref(np.asarray(LSF_M)))
                got = float(_bp_ls_false_pair(cls, phi_kwargs)[-1])
                self.assertAlmostEqual(got, want, places=4,
                                       msg=f'{label}: ls=False steady {got} != w·φ(M)={want}')


@requires_nest
class TestLinearSummationFalseNestParity(unittest.TestCase):
    """Homogeneous ``linear_summation=False`` coupling matches live NEST, model by model."""

    def test_phi_emission_trajectory_matches_nest(self):
        for label, cls, phi_kwargs, _, nest_model, nest_params in _LS_FALSE:
            with self.subTest(model=label):
                bp = _bp_ls_false_pair(cls, phi_kwargs)
                ns = _nest_ls_false_pair(nest_model, nest_params)
                k = min(len(bp), len(ns))
                compare_trace(ns[:k], bp[:k], tol=PHI_TOL,
                              metric=f'{label}:ls_false').assert_()


if __name__ == '__main__':
    unittest.main()
