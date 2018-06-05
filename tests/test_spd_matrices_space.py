"""
Unit tests for the manifold of symmetric positive definite matrices.
"""

import scipy.linalg
import unittest

import geomstats.backend as gs
import geomstats.spd_matrices_space as spd_matrices_space

from geomstats.spd_matrices_space import SPDMatricesSpace


class TestSPDMatricesSpaceMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.space = SPDMatricesSpace(n=self.n)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_is_symmetric(self):
        sym_mat = gs.array([[1, 2],
                            [2, 1]])
        self.assertTrue(spd_matrices_space.is_symmetric(sym_mat))

        not_a_sym_mat = gs.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        self.assertFalse(spd_matrices_space.is_symmetric(not_a_sym_mat))

    def test_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(gs.all(spd_matrices_space.is_symmetric(points)))

    def test_make_symmetric(self):
        sym_mat = gs.array([[1, 2],
                            [2, 1]])
        result = spd_matrices_space.make_symmetric(sym_mat)
        expected = sym_mat
        self.assertTrue(gs.allclose(result, expected))

        mat = gs.array([[1, 2, 3],
                        [0, 0, 0],
                        [3, 1, 1]])
        result = spd_matrices_space.make_symmetric(mat)
        expected = gs.array([[1, 1, 3],
                            [1, 0, 0.5],
                            [3, 0.5, 1]])
        self.assertTrue(gs.allclose(result, expected))

    def test_make_symmetric_and_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        mats = gs.random.rand(n_samples, 5, 5)

        results = spd_matrices_space.make_symmetric(mats)
        self.assertTrue(gs.all(spd_matrices_space.is_symmetric(results)))

    def test_sqrtm(self):
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)

        result = spd_matrices_space.sqrtm(points)
        expected = gs.zeros((n_samples, self.n, self.n))
        for i in range(n_samples):
            expected[i] = scipy.linalg.sqrtm(points[i])

        self.assertTrue(gs.allclose(result, expected))

    def test_random_uniform_and_belongs(self):
        self.assertTrue(self.space.belongs(self.space.random_uniform()))

    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(gs.all(self.space.belongs(points)))

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

    def test_group_log_and_exp(self):
        point_1 = 5 * gs.eye(4)
        group_log_1 = spd_matrices_space.group_log(point_1)
        result_1 = spd_matrices_space.group_exp(group_log_1)
        expected_1 = point_1

        self.assertTrue(gs.allclose(result_1, expected_1))

    def test_group_log_and_exp_vectorization(self):
        n_samples = self.n_samples
        point = self.space.random_uniform(n_samples)
        group_log = spd_matrices_space.group_log(point)
        result = spd_matrices_space.group_exp(group_log)
        expected = point

        self.assertTrue(gs.allclose(result, expected))

    def test_log_and_exp(self):
        base_point_1 = gs.array([[5., 0., 0.],
                                [0., 7., 2.],
                                [0., 2., 8.]])
        point_1 = gs.array([[9., 0., 0.],
                            [0., 5., 0.],
                            [0., 0., 1.]])

        log_1 = self.metric.log(point=point_1, base_point=base_point_1)
        result_1 = self.metric.exp(tangent_vec=log_1, base_point=base_point_1)
        expected_1 = point_1

        self.assertTrue(gs.allclose(result_1, expected_1))

    def test_exp_and_belongs(self):
        n_samples = self.n_samples
        base_point = self.space.random_uniform(n_samples=1)
        tangent_vec = self.space.random_tangent_vec_uniform(
                                               n_samples=n_samples,
                                               base_point=base_point)
        results = self.metric.exp(tangent_vec, base_point)

        self.assertTrue(gs.all(self.space.belongs(results)))

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
        results = self.metric.exp(n_tangent_vec_same_base, one_base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples,
                                     self.space.n,
                                     self.space.n)))

        # Test with the same number of base_points and tangent_vecs
        results = self.metric.exp(n_tangent_vec, n_base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples,
                                     self.space.n,
                                     self.space.n)))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        one_base_point = self.space.random_uniform(n_samples=1)
        n_base_point = self.space.random_uniform(n_samples=n_samples)

        one_point = self.space.random_uniform(n_samples=1)
        n_point = self.space.random_uniform(n_samples=n_samples)

        # Test with different points, one base point
        results = self.metric.log(n_point, one_base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples,
                                     self.space.n,
                                     self.space.n)))

        # Test with the same number of points and base points
        results = self.metric.log(n_point, n_base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples,
                                     self.space.n,
                                     self.space.n)))

        # Test with the one point and n base points
        results = self.metric.log(one_point, n_base_point)

        self.assertTrue(gs.allclose(results.shape,
                                    (n_samples,
                                     self.space.n,
                                     self.space.n)))

    def test_exp_then_log_vectorization(self):
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
        exps = self.metric.exp(n_tangent_vec_same_base, one_base_point)
        results = self.metric.log(exps, one_base_point)
        expected = n_tangent_vec_same_base

        self.assertTrue(gs.allclose(results, expected))

        # Test with the same number of base_points and tangent_vecs
        exps = self.metric.exp(n_tangent_vec, n_base_point)
        results = self.metric.log(exps, n_base_point)
        expected = n_tangent_vec

        self.assertTrue(gs.allclose(results, expected))

    def test_geodesic_and_belongs(self):
        initial_point = self.space.random_uniform()
        initial_tangent_vec = self.space.random_tangent_vec_uniform(
                                                n_samples=1,
                                                base_point=initial_point)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = gs.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(gs.all(self.space.belongs(points)))

    def test_squared_dist_is_symmetric(self):
        n_samples = self.n_samples

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(gs.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(gs.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(gs.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(gs.allclose(sq_dist_1_2, sq_dist_2_1))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (n_samples, 1),
                        'sq_dist_1_2.shape = {}'.format(sq_dist_1_2.shape))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (n_samples, 1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (n_samples, 1))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (1, 1))


if __name__ == '__main__':
        unittest.main()
