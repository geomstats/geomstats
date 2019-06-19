"""
Unit tests for the manifold of matrices.
"""

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.matrices_space import MatricesSpace


class TestMatricesSpaceMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)

        self.n = 3
        self.space = MatricesSpace(m=self.n, n=self.n)
        self.metric = self.space.metric
        self.n_samples = 2

    @geomstats.tests.np_only
    def test_is_symmetric(self):
        sym_mat = gs.array([[1., 2.],
                            [2., 1.]])
        result = self.space.is_symmetric(sym_mat)
        expected = gs.array([True])
        self.assertAllClose(result, expected)

        not_a_sym_mat = gs.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        result = self.space.is_symmetric(not_a_sym_mat)
        expected = gs.array([False])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_is_symmetric_vectorization(self):
        points = gs.array([
            [[1., 2.],
             [2., 1.]],
            [[3., 4.],
             [4., 5.]]])
        result = gs.all(self.space.is_symmetric(points))
        expected = True
        self.assertAllClose(result, expected)

    def test_make_symmetric(self):
        sym_mat = gs.array([[1., 2.],
                            [2., 1.]])
        result = self.space.make_symmetric(sym_mat)
        expected = helper.to_matrix(sym_mat)
        self.assertAllClose(result, expected)

        mat = gs.array([[1., 2., 3.],
                        [0., 0., 0.],
                        [3., 1., 1.]])
        result = self.space.make_symmetric(mat)
        expected = gs.array([[[1., 1., 3.],
                              [1., 0., 0.5],
                              [3., 0.5, 1.]]])
        self.assertAllClose(result, expected)

        mat = gs.array([[[1e100, 1e-100, 1e100],
                         [1e100, 1e-100, 1e100],
                         [1e-100, 1e-100, 1e100]]])
        result = self.space.make_symmetric(mat)

        res = 0.5 * (1e100 + 1e-100)

        expected = gs.array([[[1e100, res, res],
                              [res, 1e-100, res],
                              [res, res, 1e100]]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_make_symmetric_and_is_symmetric_vectorization(self):
        points = gs.array([
            [[1., 2.],
             [3., 4.]],
            [[5., 6.],
             [4., 9.]]])

        sym_points = self.space.make_symmetric(points)
        result = gs.all(self.space.is_symmetric(sym_points))
        expected = True
        self.assertAllClose(result, expected)

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
                gs.transpose(tangent_vector_1),
                tangent_vector_2))
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
