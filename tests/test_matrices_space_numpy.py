"""
Unit tests for the manifold of matrices.
"""

import unittest

import geomstats.backend as gs

from geomstats.matrices_space import MatricesSpace


class TestMatricesSpaceMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.space = MatricesSpace(m=self.n, n=self.n)
        self.metric = self.space.metric
        self.n_samples = 2

    def test_is_symmetric(self):
        sym_mat = gs.array([[1, 2],
                            [2, 1]])
        self.assertTrue(self.space.is_symmetric(sym_mat))

        not_a_sym_mat = gs.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        self.assertFalse(self.space.is_symmetric(not_a_sym_mat))

    def test_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        points = self.space.random_uniform(n_samples=n_samples)
        points = self.space.make_symmetric(points)
        self.assertTrue(gs.all(self.space.is_symmetric(points)))

    def test_make_symmetric(self):
        sym_mat = gs.array([[1, 2],
                            [2, 1]])
        result = self.space.make_symmetric(sym_mat)
        expected = sym_mat
        self.assertTrue(gs.allclose(result, expected))

        mat = gs.array([[1, 2, 3],
                        [0, 0, 0],
                        [3, 1, 1]])
        result = self.space.make_symmetric(mat)
        expected = gs.array([[1, 1, 3],
                            [1, 0, 0.5],
                            [3, 0.5, 1]])
        self.assertTrue(gs.allclose(result, expected))

        mat = gs.array([[1e100, 1e-100, 1e100],
                        [1e100, 1e-100, 1e100],
                        [1e-100, 1e-100, 1e100]])
        result = self.space.make_symmetric(mat)

        res = 0.5 * (1e100 + 1e-100)

        expected = gs.array([[1e100, res, res],
                             [res, 1e-100, res],
                             [res, res, 1e100]])
        self.assertTrue(gs.allclose(result, expected))

    def test_make_symmetric_and_is_symmetric_vectorization(self):
        n_samples = self.n_samples
        mats = gs.random.rand(n_samples, 5, 5)

        results = self.space.make_symmetric(mats)
        self.assertTrue(gs.all(self.space.is_symmetric(results)))

    def test_inner_product(self):
        base_point = gs.array([
            [1., 2., 3.],
            [0., 0., 0.],
            [3., 1., 1.]])

        tangent_vector_1 = gs.array([
            [1., 2., 3.],
            [0., -10., 0.],
            [30., 1., 1.]])

        tangent_vector_2 = gs.array([
            [1., 4., 3.],
            [5., 0., 0.],
            [3., 1., 1.]])

        result = self.metric.inner_product(
            tangent_vector_1,
            tangent_vector_2,
            base_point=base_point)

        expected = gs.trace(
            gs.matmul(
                tangent_vector_1.T,
                tangent_vector_2))

        gs.testing.assert_allclose(result, expected)


if __name__ == '__main__':
        unittest.main()
