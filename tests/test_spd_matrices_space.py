"""
Unit tests for the manifold of symmetric positive definite matrices.
"""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.spd_matrices_space import SPDMatricesSpace


class TestSPDMatricesSpaceMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        gs.random.seed(1234)

        self.n = 3
        self.space = SPDMatricesSpace(n=self.n)
        self.metric = self.space.metric
        self.n_samples = 4

    @geomstats.tests.np_and_tf_only
    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = gs.array([[True]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        result = self.space.belongs(points)
        self.assertAllClose(gs.shape(result), (n_samples, 1))

    @geomstats.tests.np_and_tf_only
    def vector_from_symmetric_matrix_and_symmetric_matrix_from_vector(self):
        sym_mat_1 = gs.array([[1., 0.6, -3.],
                              [0.6, 7., 0.],
                              [-3., 0., 8.]])
        vector_1 = self.space.vector_from_symmetric_matrix(sym_mat_1)
        result_1 = self.space.symmetric_matrix_from_vector(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(gs.allclose(result_1, expected_1))

        vector_2 = gs.array([1, 2, 3, 4, 5, 6])
        sym_mat_2 = self.space.symmetric_matrix_from_vector(vector_2)
        result_2 = self.space.vector_from_symmetric_matrix(sym_mat_2)
        expected_2 = vector_2

        self.assertTrue(gs.allclose(result_2, expected_2))

    @geomstats.tests.np_and_tf_only
    def vector_and_symmetric_matrix_vectorization(self):
        n_samples = self.n_samples
        vector = gs.random.rand(n_samples, 6)
        sym_mat = self.space.symmetric_matrix_from_vector(vector)
        result = self.space.vector_from_symmetric_matrix(sym_mat)
        expected = vector

        self.assertTrue(gs.allclose(result, expected))

        sym_mat = self.space.random_uniform(n_samples)
        vector = self.space.vector_from_symmetric_matrix(sym_mat)
        result = self.space.symmetric_matrix_from_vector(vector)
        expected = sym_mat

        self.assertTrue(gs.allclose(result, expected))

    @geomstats.tests.np_and_tf_only
    def test_log_and_exp(self):
        base_point = gs.array([[5., 0., 0.],
                               [0., 7., 2.],
                               [0., 2., 8.]])
        point = gs.array([[9., 0., 0.],
                          [0., 5., 0.],
                          [0., 0., 1.]])

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = helper.to_matrix(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_and_belongs(self):
        n_samples = self.n_samples
        base_point = self.space.random_uniform(n_samples=1)
        tangent_vec = self.space.random_tangent_vec_uniform(
                                               n_samples=n_samples,
                                               base_point=base_point)
        exps = self.metric.exp(tangent_vec, base_point)
        result = self.space.belongs(exps)
        expected = gs.array([[True]] * n_samples)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_exp_vectorization(self):
        n_samples = self.n_samples
        one_base_point = self.space.random_uniform(n_samples=1)
        n_base_point = self.space.random_uniform(n_samples=n_samples)

        n_tangent_vec_same_base = self.space.random_tangent_vec_uniform(
                                                 n_samples=n_samples,
                                                 base_point=one_base_point)
        n_tangent_vec = self.space.random_tangent_vec_uniform(
                                                 n_samples=n_samples,
                                                 base_point=n_base_point)

        # Test with the 1 base_point, and several different tangent_vecs
        result = self.metric.exp(n_tangent_vec_same_base, one_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of base_points and tangent_vecs
        result = self.metric.exp(n_tangent_vec, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

    @geomstats.tests.np_and_tf_only
    def test_log_vectorization(self):
        n_samples = self.n_samples
        one_base_point = self.space.random_uniform(n_samples=1)
        n_base_point = self.space.random_uniform(n_samples=n_samples)

        one_point = self.space.random_uniform(n_samples=1)
        n_point = self.space.random_uniform(n_samples=n_samples)

        # Test with different points, one base point
        result = self.metric.log(n_point, one_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the same number of points and base points
        result = self.metric.log(n_point, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

        # Test with the one point and n base points
        result = self.metric.log(one_point, n_base_point)

        self.assertAllClose(
            gs.shape(result), (n_samples, self.space.n, self.space.n))

    @geomstats.tests.np_and_tf_only
    def test_geodesic_and_belongs(self):
        initial_point = self.space.random_uniform()
        initial_tangent_vec = self.space.random_tangent_vec_uniform(
                                                n_samples=1,
                                                base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        n_points = 10
        t = gs.linspace(start=0., stop=1., num=n_points)
        points = geodesic(t)
        result = self.space.belongs(points)
        expected = gs.array([[True]] * n_points)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_squared_dist_is_symmetric(self):
        n_samples = self.n_samples

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertAllClose(sq_dist_1_2, sq_dist_2_1)

    @geomstats.tests.np_and_tf_only
    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples, 1))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        result = self.metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples, 1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        result = self.metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (n_samples, 1))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        result = self.metric.squared_dist(point_1, point_2)

        self.assertAllClose(gs.shape(result), (1, 1))


if __name__ == '__main__':
    geomstats.tests.main()
