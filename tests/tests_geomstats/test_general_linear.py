"""Unit tests for the General Linear group."""

import warnings

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.general_linear import GeneralLinear

RTOL = 1e-5


class TestGeneralLinear(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)
        self.n = 3
        self.n_samples = 2
        self.group = GeneralLinear(n=self.n)
        self.group_pos = GeneralLinear(self.n, positive_det=True)

        warnings.simplefilter('ignore', category=ImportWarning)

    def test_belongs_shape(self):
        mat = gs.eye(3)
        result = self.group.belongs(mat)
        self.assertAllClose(gs.shape(result), ())

        mat = gs.ones((3, 3))
        result = self.group.belongs(mat)
        self.assertAllClose(gs.shape(result), ())

    def test_belongs(self):
        mat = gs.eye(3)
        result = self.group.belongs(mat)
        expected = True
        self.assertAllClose(result, expected)

        mat = gs.ones((3, 3))
        result = self.group.belongs(mat)
        expected = False
        self.assertAllClose(result, expected)

        mat = gs.ones(3)
        result = self.group.belongs(mat)
        expected = False
        self.assertAllClose(result, expected)

    def test_belongs_vectorization_shape(self):
        mats = gs.array([gs.eye(3), gs.ones((3, 3))])
        result = self.group.belongs(mats)
        self.assertAllClose(gs.shape(result), (2,))

    def test_belongs_vectorization(self):
        mats = gs.array([gs.eye(3), gs.ones((3, 3))])
        result = self.group.belongs(mats)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_random_and_belongs(self):
        for group in [self.group, self.group_pos]:
            point = group.random_point()
            result = group.belongs(point)
            self.assertTrue(result)

    def test_random_and_belongs_vectorization(self):
        n_samples = 4
        expected = gs.array([True] * n_samples)
        for group in [self.group, self.group_pos]:
            point = group.random_point(n_samples)
            result = group.belongs(point)
            self.assertAllClose(result, expected)

    def test_compose(self):
        mat1 = gs.array([
            [1., 0.],
            [0., 2.]])
        mat2 = gs.array([
            [2., 0.],
            [0., 1.]])
        result = self.group.compose(mat1, mat2)
        expected = 2. * GeneralLinear(2).identity
        self.assertAllClose(result, expected)

    def test_inv(self):
        mat_a = gs.array([
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.]])
        imat_a = 1. / 3. * gs.array([
            [-2., -4., 3.],
            [-2., 11., -6.],
            [3., -6., 3.]])
        expected = imat_a
        result = self.group.inverse(mat_a)
        self.assertAllClose(result, expected)

    def test_inv_vectorized(self):
        mat_a = gs.array([
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]])
        mat_b = - gs.eye(3, 3)
        result = self.group.inverse(gs.array([mat_a, mat_b]))
        expected = gs.array([mat_a, mat_b])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_log_and_exp(self):
        point = 5 * gs.eye(self.n)
        group_log = self.group.log(point)

        result = self.group.exp(group_log)
        expected = point
        self.assertAllClose(result, expected)

    def test_exp_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])

        expected = gs.array([[[7.38905609, 0., 0.],
                              [0., 20.0855369, 0.],
                              [0., 0., 54.5981500]],
                             [[2.718281828, 0., 0.],
                              [0., 148.413159, 0.],
                              [0., 0., 403.42879349]]])

        expected = gs.cast(expected, gs.float64)
        point = gs.cast(point, gs.float64)

        result = self.group.exp(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_log_vectorization(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        expected = gs.array([[[0.693147180, 0., 0.],
                              [0., 1.09861228866, 0.],
                              [0., 0., 1.38629436]],
                             [[0., 0., 0.],
                              [0., 1.609437912, 0.],
                              [0., 0., 1.79175946]]])
        result = self.group.log(point)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_orbit(self):
        point = gs.array([
            [gs.exp(4.), 0.],
            [0., gs.exp(2.)]])
        sqrt = gs.array([
            [gs.exp(2.), 0.],
            [0., gs.exp(1.)]])
        identity = GeneralLinear(2).identity

        path = GeneralLinear(2).orbit(point)
        time = gs.linspace(0., 1., 3)

        result = path(time)
        expected = gs.array([identity, sqrt, point])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_orbit_vectorization(self):
        point = gs.array([
            [gs.exp(4.), 0.],
            [0., gs.exp(2.)]])
        sqrt = gs.array([
            [gs.exp(2.), 0.],
            [0., gs.exp(1.)]])
        identity = GeneralLinear(2).identity

        path = GeneralLinear(2).orbit(gs.stack([point] * 2), identity)
        time = gs.linspace(0., 1., 3)

        result = path(time)
        expected = gs.array([identity, sqrt, point])
        expected = gs.stack([expected] * 2)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_expm_and_logm_vectorization_symmetric(self):
        point = gs.array([[[2., 0., 0.],
                           [0., 3., 0.],
                           [0., 0., 4.]],
                          [[1., 0., 0.],
                           [0., 5., 0.],
                           [0., 0., 6.]]])
        result = self.group.exp(self.group.log(point))
        expected = point
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group, shape)
        for res in result:
            self.assertTrue(res)

    def test_projection_and_belongs_pos(self):
        shape = (self.n_samples, self.n, self.n)
        result = helper.test_projection_and_belongs(self.group_pos, shape)
        for res in result:
            self.assertTrue(res)
