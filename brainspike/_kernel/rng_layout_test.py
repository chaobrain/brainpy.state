# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-6 (D-g) acceptance tests: name-keyed streams and gid-blocked dynamics noise.

* insertion-order invariance: adding / reordering unrelated objects leaves every other
  object's connectivity, weights, initial state and noise exactly unchanged;
* extension invariance: a population grown beyond its id block draws the same noise
  for the neurons it already had (Poisson generators, ``poisson_generator_ps``);
* names: renamed objects get new streams; auto names ``<model>_<k>`` per class.
"""
import brainunit as u
import numpy as np
import pytest

from brainspike._kernel.connectivity import fixed_indegree, one_to_one, pairwise_bernoulli
from brainspike._kernel.models import iaf_psc_alpha, poisson_generator, poisson_generator_ps
from brainspike._kernel.network import Network
from brainspike._kernel.params import Normal, Uniform
from brainspike._kernel.rng import GID_BLOCK

DT = 0.1 * u.ms
T = 20.0 * u.ms


def _arr(x):
    return np.asarray(u.get_mantissa(x))


def build(extra=False, noise_first=False, n_noise=60, noise_name='noise', seed=3):
    net = Network(dt=DT, seed=seed)
    if extra:  # an unrelated sub-network added first
        X = net.add(iaf_psc_alpha, 30, name='X')
        xn = net.add(poisson_generator, 30, params=dict(rate=8000.0 * u.Hz), name='xnoise')
        net.connect(xn, X, one_to_one(), port='ex', weight=50.0 * u.pA, name='x_in')
        net.connect(X, X, fixed_indegree(3), port='ex', weight=Normal(20.0 * u.pA, 5.0 * u.pA),
                    name='XX')

    def add_noise():
        return net.add(poisson_generator, n_noise, params=dict(rate=8000.0 * u.Hz), name=noise_name)

    def add_E():
        return net.add(iaf_psc_alpha, 60, init=dict(V_m=Uniform(-70.0 * u.mV, -56.0 * u.mV)), name='E')

    if noise_first:
        noise = add_noise()
        E = add_E()
    else:
        E = add_E()
        noise = add_noise()
    net.connect(noise[:60], E, one_to_one(), port='ex', weight=Normal(80.0 * u.pA, 10.0 * u.pA),
                name='drive')
    net.connect(E, E, fixed_indegree(5), port='ex', weight=Normal(30.0 * u.pA, 5.0 * u.pA),
                delay=1.5 * u.ms, name='EE')
    net.connect(E, E, pairwise_bernoulli(0.1), port='in_', weight=-10.0 * u.pA, name='EEi')
    rE = net.observe(E, 'spike', 'V_m', name='rE')
    rN = net.observe(noise[:60], 'spike', name='rN')
    return net, rE, rN


def run(net, rE, rN):
    plan = net.compile()
    res = plan.runner().simulate(T)
    out = {
        'E_spike': _arr(res[rE]['spike']), 'E_V': _arr(res[rE]['V_m']),
        'noise': _arr(res[rN]['spike']),
    }
    for name in ('drive', 'EE', 'EEi'):
        es = plan.edges(name)
        out[name] = (np.asarray(es.pre_index), np.asarray(es.post_index), _arr(es.weight))
    return out


def assert_same(a, b, keys=None):
    for k in keys or a:
        if isinstance(a[k], tuple):
            for x, y in zip(a[k], b[k]):
                np.testing.assert_array_equal(x, y, err_msg=k)
        else:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.fixture(scope='module')
def ref():
    return run(*build())


class TestInsertionOrderInvariance:
    def test_unrelated_objects_added_first(self, ref):
        out = run(*build(extra=True))
        assert ref['noise'].sum() > 0 and ref['E_spike'].sum() > 0
        assert_same(ref, out)

    def test_reordered_populations(self, ref):
        # 'noise' before 'E': gids of both change, their streams do not
        assert_same(ref, run(*build(noise_first=True)))

    def test_renamed_object_gets_new_streams(self, ref):
        out = run(*build(noise_name='noise2'))
        assert not np.array_equal(ref['noise'], out['noise'])
        assert_same(ref, out, keys=['EE', 'EEi', 'drive'])  # connectivity is unaffected

    def test_seed_changes_everything(self, ref):
        out = run(*build(seed=4))
        assert not np.array_equal(ref['noise'], out['noise'])
        assert not np.array_equal(ref['EE'][0], out['EE'][0])


class TestBlockedNoise:
    def test_population_extended_beyond_its_block(self, ref):
        # 60 -> 2 * GID_BLOCK + 37 generators: the first 60 (block 0) draw the same counts
        out = run(*build(n_noise=2 * GID_BLOCK + 37))
        assert_same(ref, out, keys=['noise', 'E_spike', 'drive', 'EE', 'EEi'])
        # the drive's pre view is now a slice of a larger source population, so its
        # storage sums in another float order (ulp-level, not an RNG difference)
        np.testing.assert_allclose(ref['E_V'], out['E_V'], rtol=1e-6)

    def test_poisson_generator_ps_extension(self):
        def go(n):
            net = Network(dt=DT, seed=1)
            g = net.add(poisson_generator_ps, n, params=dict(rate=3000.0 * u.Hz,
                                                            dead_time=0.05 * u.ms), name='ps')
            r = net.observe(g[:40], 'spike', 'next_spike', name='r')
            res = net.compile().runner().simulate(5.0 * u.ms)
            return _arr(res[r]['spike']), _arr(res[r]['next_spike'])

        a, b = go(40), go(GID_BLOCK + 5)
        assert a[0].sum() > 0
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestAutoNames:
    def test_names_per_class(self):
        net = Network(dt=DT)
        a = net.add(iaf_psc_alpha, 2)
        net.add(iaf_psc_alpha, 2, name='iaf_psc_alpha_1')  # explicit name taken first
        b = net.add(iaf_psc_alpha, 2)
        g = net.add(poisson_generator, 2)
        p = net.connect(g, a, one_to_one(), port='ex', weight=1.0 * u.pA)
        q = net.connect(a, b, fixed_indegree(1), port='ex', weight=1.0 * u.pA)
        q2 = net.connect(b, a, fixed_indegree(1), port='ex', weight=1.0 * u.pA)
        o = net.observe(a, 'V_m')
        assert [x.name for x in net.populations] == ['iaf_psc_alpha_0', 'iaf_psc_alpha_1',
                                                     'iaf_psc_alpha_2', 'poisson_generator_0']
        assert (a.label, b.label) == ('iaf_psc_alpha_0', 'iaf_psc_alpha_2')
        assert (p.name, q.name, q2.name, o.name) == ('OneToOne_0', 'FixedIndegree_0',
                                                     'FixedIndegree_1', 'Observer_0')

    def test_duplicate_names_raise_across_kinds(self):
        net = Network(dt=DT)
        a = net.add(iaf_psc_alpha, 2, name='A')
        with pytest.raises(ValueError, match='already used'):
            net.add(iaf_psc_alpha, 2, name='A')
        with pytest.raises(ValueError, match='already used'):
            net.connect(a, a, port='ex', weight=1.0 * u.pA, name='A')
        with pytest.raises(ValueError, match='already used'):
            net.observe(a, 'V_m', name='A')
        net.add(iaf_psc_alpha, 1)
        with pytest.raises(ValueError, match='already used'):
            net.add(iaf_psc_alpha, 1, name='iaf_psc_alpha_0')

    @staticmethod
    def _unnamed(prefix_cls):
        net = Network(dt=DT, seed=2)
        if prefix_cls is not None:
            net.add(prefix_cls, 10, params=dict(rate=8000.0 * u.Hz)
                    if prefix_cls is poisson_generator else None)
        g = net.add(poisson_generator, 30, params=dict(rate=8000.0 * u.Hz))
        r = net.observe(g, 'spike', name='r')
        return _arr(net.compile().runner().simulate(5.0 * u.ms)[r]['spike'])

    def test_unnamed_order_matters_only_within_a_class(self):
        base = self._unnamed(None)
        # an unnamed object of another class added first: same auto name, same stream
        np.testing.assert_array_equal(base, self._unnamed(iaf_psc_alpha))
        # an unnamed object of the same class added first: the auto names shift (documented)
        assert not np.array_equal(base, self._unnamed(poisson_generator))


def test_connectivity_independent_of_the_post_view():
    # a target's edges depend on (projection name, population name, local id): a
    # projection onto E[10:30] realises the same in-edges as onto E for those targets
    def edges(view_slice):
        net = Network(dt=DT, seed=5)
        net.add(iaf_psc_alpha, 7, name='pad')
        E = net.add(iaf_psc_alpha, 40, name='E')
        net.connect(E, E[view_slice], fixed_indegree(4), port='ex', weight=1.0 * u.pA, name='p')
        es = net.compile().edges('p')
        return np.asarray(es.pre_index), np.asarray(es.post_index)

    pa, qa = edges(slice(None))
    pb, qb = edges(slice(10, 30))
    sel = (qa >= 10) & (qa < 30)
    np.testing.assert_array_equal(pa[sel], pb)
    np.testing.assert_array_equal(qa[sel] - 10, qb)
