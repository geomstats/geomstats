"""
Unit tests for Stiefel manifolds.
"""

import unittest
import warnings

import geomstats.backend as gs
import tests.helper as helper

from geomstats.stiefel import Stiefel

ATOL = 1e-6


class TestStiefelMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        """
        Tangent vectors constructed following:
        http://noodle.med.yale.edu/hdtag/notes/steifel_notes.pdf
        """
        warnings.filterwarnings('ignore')

        gs.random.seed(1234)

        self.p = 3
        self.n = 4
        self.space = Stiefel(self.n, self.p)
        self.n_samples = 10
        self.dimension = int(
            self.p * self.n - (self.p * (self.p + 1) / 2))

        self.point_a = gs.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 0.]])

        self.point_b = gs.array([
            [1. / gs.sqrt(2), 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1. / gs.sqrt(2), 0., 0.]])

        point_perp = gs.array([
            [0.],
            [0.],
            [0.],
            [1.]])

        matrix_a_1 = gs.array([
            [0., 2., -5.],
            [-2., 0., -1.],
            [5., 1., 0.]])

        matrix_b_1 = gs.array([
            [-2., 1., 4.]])

        matrix_a_2 = gs.array([
            [0., 2., -5.],
            [-2., 0., -1.],
            [5., 1., 0.]])

        matrix_b_2 = gs.array([
            [-2., 1., 4.]])

        self.tangent_vector_1 = (
            gs.matmul(self.point_a, matrix_a_1)
            + gs.matmul(point_perp, matrix_b_1))

        self.tangent_vector_2 = (
            gs.matmul(self.point_a, matrix_a_2)
            + gs.matmul(point_perp, matrix_b_2))

        self.metric = self.space.canonical_metric

    def test_belongs(self):
        point = self.space.random_uniform()
        belongs = self.space.belongs(point)

        gs.testing.assert_allclose(belongs.shape, (1, 1))

    def test_random_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = gs.array([[True]])

        gs.testing.assert_allclose(result, expected)

    def test_random_uniform(self):
        point = self.space.random_uniform()

        gs.testing.assert_allclose(point.shape, (1, self.n, self.p))

    def test_log_and_exp(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = self.point_a
        point = self.point_b

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = helper.to_matrix(point)

        gs.testing.assert_allclose(result, expected, atol=ATOL)

    def test_exp_and_belongs(self):
        base_point = self.point_a
        tangent_vec = self.tangent_vector_1

        exp = self.metric.exp(
            tangent_vec=tangent_vec,
            base_point=base_point)
        self.assertTrue(self.space.belongs(exp))

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_base_point = self.point_a
        n_base_points = gs.tile(self.point_a, (n_samples, 1, 1))

        one_tangent_vec = self.tangent_vector_1
        result = self.metric.exp(one_tangent_vec, one_base_point)
        gs.testing.assert_allclose(result.shape, (1, n, p))

        n_tangent_vecs = gs.tile(self.tangent_vector_2, (n_samples, 1, 1))

        result = self.metric.exp(n_tangent_vecs, one_base_point)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        expected = gs.zeros((n_samples, n, p))
        for i in range(n_samples):
            expected[i] = self.metric.exp(n_tangent_vecs[i], one_base_point)
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

        result = self.metric.exp(one_tangent_vec, n_base_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        expected = gs.zeros((n_samples, n, p))
        for i in range(n_samples):
            expected[i] = self.metric.exp(one_tangent_vec, n_base_points[i])
        expected = helper.to_vector(expected)
        gs.testing.assert_allclose(result, expected)

    def test_log_vectorization(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.space.random_uniform()
        one_base_point = self.space.random_uniform()
        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.log(one_point, one_base_point)
        gs.testing.assert_allclose(result.shape, (1, n, p))

        result = self.metric.log(n_points, one_base_point)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.log(one_point, n_base_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.log(n_points, n_base_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

    def test_retractation_and_lifting(self):
        """
        Test that the riemannian exponential
        and the riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = self.point_a
        point = self.point_b
        tangent_vec = self.tangent_vector_1

        lifted = self.metric.lifting(point=point, base_point=base_point)
        result = self.metric.retraction(
            tangent_vec=lifted, base_point=base_point)
        expected = helper.to_matrix(point)

        gs.testing.assert_allclose(result, expected, atol=ATOL)

        retract = self.metric.retraction(
            tangent_vec=tangent_vec, base_point=base_point)
        result = self.metric.lifting(point=retract, base_point=base_point)
        expected = helper.to_matrix(tangent_vec)

        gs.testing.assert_allclose(result, expected, atol=ATOL)

    def test_lifting_vectorization(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        one_base_point = self.point_b
        n_points = gs.tile(self.point_a, (n_samples, 1, 1))
        n_base_points = gs.tile(self.point_b, (n_samples, 1, 1))

        result = self.metric.lifting(one_point, one_base_point)
        gs.testing.assert_allclose(result.shape, (1, n, p))

        result = self.metric.lifting(n_points, one_base_point)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.lifting(one_point, n_base_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.lifting(n_points, n_base_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

    def test_retraction_vectorization(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        n_points = gs.tile(self.point_a, (n_samples, 1, 1))
        one_tangent_vec = self.tangent_vector_1
        n_tangent_vecs = gs.tile(self.tangent_vector_2, (n_samples, 1, 1))

        result = self.metric.retraction(one_tangent_vec, one_point)
        gs.testing.assert_allclose(result.shape, (1, n, p))

        result = self.metric.retraction(n_tangent_vecs, one_point)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.retraction(one_tangent_vec, n_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

        result = self.metric.retraction(n_tangent_vecs, n_points)
        gs.testing.assert_allclose(result.shape, (n_samples, n, p))

    def test_inner_product(self):
        base_point = self.point_a
        tangent_vector_1 = self.tangent_vector_1
        tangent_vector_2 = self.tangent_vector_2

        result = self.metric.inner_product(
            tangent_vector_1,
            tangent_vector_2,
            base_point=base_point)
        gs.testing.assert_allclose(result.shape, (1, 1))


if __name__ == '__main__':
        unittest.main()
