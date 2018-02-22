"""Unit tests for symmetric positive definite matrices."""

import numpy as np
import unittest

import geomstats.spd_matrices_space as spd_matrices_space
from geomstats.spd_matrices_space import SPDMatricesSpace


class TestSPDMatricesSpaceMethods(unittest.TestCase):
    def setUp(self):
        self.dimension = 3
        self.space = SPDMatricesSpace(dimension=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10

    def test_is_symmetric(self):
        sym_mat = np.array([[1, 2],
                            [2, 1]])
        self.assertTrue(spd_matrices_space.is_symmetric(sym_mat))

        not_a_sym_mat = np.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        self.assertFalse(spd_matrices_space.is_symmetric(not_a_sym_mat))

    def test_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(np.all(spd_matrices_space.is_symmetric(points)))

    def test_make_symmetric(self):
        sym_mat = np.array([[1, 2],
                            [2, 1]])
        result = spd_matrices_space.make_symmetric(sym_mat)
        expected = sym_mat
        self.assertTrue(np.allclose(result, expected))

        mat = np.array([[1, 2, 3],
                        [0, 0, 0],
                        [3, 1, 1]])
        result = spd_matrices_space.make_symmetric(mat)
        expected = np.array([[1, 1, 3],
                            [1, 0, 0.5],
                            [3, 0.5, 1]])
        self.assertTrue(np.allclose(result, expected))

    def test_make_symmetric_and_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        mats = np.random.rand(n_samples, 5, 5)

        results = spd_matrices_space.make_symmetric(mats)
        self.assertTrue(np.all(spd_matrices_space.is_symmetric(results)))

    def test_random_uniform_and_belongs(self):
        self.assertTrue(self.space.belongs(self.space.random_uniform()))

    def test_random_uniform_and_belongs_vectorization(self):
        """
        Test that the random uniform method samples
        on the hypersphere space.
        """
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        self.assertTrue(np.all(self.space.belongs(points)))

    def matrix_to_vector_and_vector_to_matrix(self):
        sym_mat_1 = np.array([[1., 0.6, -3.],
                              [0.6, 7., 0.],
                              [-3., 0., 8.]])
        vector_1 = self.space.matrix_to_vector(sym_mat_1)
        result_1 = self.space.vector_to_matrix(vector_1)
        expected_1 = sym_mat_1

        self.assertTrue(np.allclose(result_1, expected_1))

        vector_2 = np.array([1, 2, 3, 4, 5, 6])
        sym_mat_2 = self.space.vector_to_matrix(vector_2)
        result_2 = self.space.matrix_to_vector(sym_mat_2)
        expected_2 = vector_2

        self.assertTrue(np.allclose(result_2, expected_2))

    def matrix_to_vector_and_vector_to_matrix_vectorization(self):
        n_samples = self.n_samples
        vector = np.random.rand(n_samples, 6)
        sym_mat = self.space.vector_to_matrix(vector)
        result = self.space.matrix_to_vector(sym_mat)
        expected = vector

        self.assertTrue(np.allclose(result, expected))

        sym_mat = self.space.random_uniform(n_samples)
        vector = self.space.matrix_to_vector(sym_mat)
        result = self.space.vector_to_matrix(vector)
        expected = sym_mat

        self.assertTrue(np.allclose(result, expected))

    def test_group_log_and_exp(self):
        point_1 = 5 * np.eye(4)
        group_log_1 = spd_matrices_space.group_log(point_1)
        result_1 = spd_matrices_space.group_exp(group_log_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_group_log_and_exp_vectorization(self):
        n_samples = self.n_samples
        point = self.space.random_uniform(n_samples)
        group_log = spd_matrices_space.group_log(point)
        result = spd_matrices_space.group_exp(group_log)
        expected = point

        self.assertTrue(np.allclose(result, expected))

    def test_log_and_exp(self):
        base_point_1 = np.array([[5., 0., 0.],
                                [0., 7., 2.],
                                [0., 2., 8.]])
        point_1 = np.array([[9., 0., 0.],
                            [0., 5., 0.],
                            [0., 0., 1.]])

        log_1 = self.metric.log(point=point_1, base_point=base_point_1)
        result_1 = self.metric.exp(tangent_vec=log_1, base_point=base_point_1)
        expected_1 = point_1

        self.assertTrue(np.allclose(result_1, expected_1))

    def test_exp_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_point = self.space.random_uniform(n_samples=1)
        results = self.metric.exp(tangent_vecs, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_points = self.space.random_uniform(n_samples=n_samples)
        results = self.metric.exp(tangent_vecs, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.space.random_uniform(n_samples=1)
        base_points = self.space.random_uniform(n_samples=n_samples)
        results = self.metric.exp(tangent_vec, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

    def test_log_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_point = self.space.random_uniform(n_samples=1)
        results = self.metric.log(tangent_vecs, base_point)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_points = self.space.random_uniform(n_samples=n_samples)
        results = self.metric.log(tangent_vecs, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.space.random_uniform(n_samples=1)
        base_points = self.space.random_uniform(n_samples=n_samples)
        results = self.metric.log(tangent_vec, base_points)

        self.assertTrue(np.allclose(results.shape,
                                    (n_samples,
                                     self.space.dimension,
                                     self.space.dimension)))

    def test_exp_and_log_vectorization(self):
        n_samples = self.n_samples
        # Test with the 1 base_point, and several different tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_point = self.space.random_uniform(n_samples=1)
        exps = self.metric.exp(tangent_vecs, base_point)
        results = self.metric.log(exps, base_point)
        expected = tangent_vecs

        self.assertTrue(np.allclose(results, expected))

        # Test with the same number of base_points and tangent_vecs
        tangent_vecs = self.space.random_uniform(n_samples=n_samples)
        base_points = self.space.random_uniform(n_samples=n_samples)
        exps = self.metric.exp(tangent_vecs, base_points)
        results = self.metric.log(exps, base_point)
        expected = tangent_vecs

        self.assertTrue(np.allclose(results, expected))

        # Test with the several base_points, and 1 tangent_vec
        tangent_vec = self.space.random_uniform(n_samples=1)
        base_points = self.space.random_uniform(n_samples=n_samples)
        exps = self.metric.exp(tangent_vec, base_points)
        results = self.metric.log(exps, base_point)
        expected = tangent_vecs

        self.assertTrue(np.allclose(results, expected))

    def test_geodesic_and_belongs(self):
        initial_point = self.space.random_uniform()
        initial_tangent_vec = np.array([[9., 0., 0.],
                                        [0., 5., 0.],
                                        [0., 0., 1.]])
        print('initial_point')
        print(initial_point.shape)
        print('initial_tangent_vec')
        print(initial_tangent_vec.shape)
        geodesic = self.metric.geodesic(
                                   initial_point=initial_point,
                                   initial_tangent_vec=initial_tangent_vec)

        t = np.linspace(start=0, stop=1, num=100)
        points = geodesic(t)
        self.assertTrue(np.all(self.space.belongs(points)))

    def test_squared_dist_is_symmetric(self):
        n_samples = self.n_samples

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=1)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=1)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1))

        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)
        sq_dist_2_1 = self.metric.squared_dist(point_2, point_1)

        self.assertTrue(np.allclose(sq_dist_1_2, sq_dist_2_1))

    def test_squared_dist_vectorization(self):
        n_samples = self.n_samples
        point_1 = self.space.random_uniform(n_samples=n_samples)
        point_2 = self.space.random_uniform(n_samples=n_samples)

        sq_dist_1_2 = self.metric.squared_dist(point_1, point_2)

        self.assertTrue(sq_dist_1_2.shape == (n_samples, 1))

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
