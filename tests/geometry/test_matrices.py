"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices


class TestMatrices(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.m = 2
        self.n = 3
        self.space = Matrices(m=self.n, n=self.n)
        self.space_nonsquare = Matrices(m=self.m, n=self.n)
        self.metric = self.space.metric
        self.n_samples = 2

    @geomstats.tests.np_only
    def test_mul(self):
        a = gs.eye(3, 3, 1)
        b = gs.eye(3, 3, -1)
        c = gs.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]])
        d = gs.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
        result = self.space.mul([a, b], [b, a])
        expected = gs.array([c, d])
        self.assertAllClose(result, expected)

        result = self.space.mul(a, [a, b])
        expected = gs.array([gs.eye(3, 3, 2), c])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_bracket(self):
        x = gs.array([
            [0., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]])
        y = gs.array([
            [0., 0., 1.],
            [0., 0., 0.],
            [-1., 0., 0.]])
        z = gs.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]])
        result = self.space.bracket([x, y], [y, z])
        expected = gs.array([z, x])
        self.assertAllClose(result, expected)

        result = self.space.bracket(x, [x, y, z])
        expected = gs.array([gs.zeros((3, 3)), z, -y])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_transpose(self):
        tr = self.space.transpose
        ar = gs.array
        a = gs.eye(3, 3, 1)
        b = gs.eye(3, 3, -1)
        self.assertAllClose(tr(a), b)
        self.assertAllClose(tr(ar([a, b])), ar([b, a]))

    def test_is_symmetric(self):
        not_squared = gs.array([[1., 2.], [2., 1.], [3., 1.]])
        result = self.space.is_symmetric(not_squared)
        expected = False
        self.assertAllClose(result, expected)

        sym_mat = gs.array([[1., 2.], [2., 1.]])
        result = self.space.is_symmetric(sym_mat)
        expected = gs.array(True)
        self.assertAllClose(result, expected)

        not_a_sym_mat = gs.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        result = self.space.is_symmetric(not_a_sym_mat)
        expected = gs.array(False)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_is_skew_symmetric(self):
        skew_mat = gs.array([[0, - 2.],
                            [2., 0]])
        result = self.space.is_skew_symmetric(skew_mat)
        expected = gs.array(True)
        self.assertAllClose(result, expected)

        not_a_sym_mat = gs.array([[1., 0.6, -3.],
                                  [6., -7., 0.],
                                  [0., 7., 8.]])
        result = self.space.is_skew_symmetric(not_a_sym_mat)
        expected = gs.array(False)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_is_symmetric_vectorization(self):
        points = gs.array([
            [[1., 2.],
             [2., 1.]],
            [[3., 4.],
             [4., 5.]],
            [[1., 2.],
             [3., 4.]]])
        result = self.space.is_symmetric(points)
        expected = [True, True, False]
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_make_symmetric(self):
        sym_mat = gs.array([[1., 2.],
                            [2., 1.]])
        result = self.space.to_symmetric(sym_mat)
        expected = sym_mat
        self.assertAllClose(result, expected)

        mat = gs.array([[1., 2., 3.],
                        [0., 0., 0.],
                        [3., 1., 1.]])
        result = self.space.to_symmetric(mat)
        expected = gs.array([[1., 1., 3.],
                             [1., 0., 0.5],
                             [3., 0.5, 1.]])
        self.assertAllClose(result, expected)

        mat = gs.array([[1e100, 1e-100, 1e100],
                        [1e100, 1e-100, 1e100],
                        [1e-100, 1e-100, 1e100]])
        result = self.space.to_symmetric(mat)

        res = 0.5 * (1e100 + 1e-100)

        expected = gs.array([[1e100, res, res],
                             [res, 1e-100, res],
                             [res, res, 1e100]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_make_symmetric_and_is_symmetric_vectorization(self):
        points = gs.array([
            [[1., 2.],
             [3., 4.]],
            [[5., 6.],
             [4., 9.]]])

        sym_points = self.space.to_symmetric(points)
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

        self.assertAllClose(result, expected)

    def test_cong(self):
        base_point = gs.array([
            [1., 2., 3.],
            [0., 0., 0.],
            [3., 1., 1.]])

        tangent_vector = gs.array([
            [1., 2., 3.],
            [0., -10., 0.],
            [30., 1., 1.]])

        result = self.space.congruent(tangent_vector, base_point)
        expected = gs.matmul(
            tangent_vector, gs.transpose(base_point))
        expected = gs.matmul(base_point, expected)

        self.assertAllClose(result, expected)

    def test_belongs(self):
        base_point_square = gs.zeros((self.n, self.n))
        base_point_nonsquare = gs.zeros((self.m, self.n))

        result = self.space.belongs(base_point_square)
        expected = True
        self.assertAllClose(result, expected)
        result = self.space_nonsquare.belongs(base_point_square)
        expected = False
        self.assertAllClose(result, expected)

        result = self.space.belongs(base_point_nonsquare)
        expected = False
        self.assertAllClose(result, expected)
        result = self.space_nonsquare.belongs(base_point_nonsquare)
        expected = True
        self.assertAllClose(result, expected)

        result = self.space.belongs(gs.zeros((2, 2, 3)))
        self.assertFalse(gs.all(result))

        result = self.space.belongs(gs.zeros((2, 3, 3)))
        self.assertTrue(gs.all(result))

    def test_is_diagonal(self):
        base_point = gs.array([
            [1., 2., 3.],
            [0., 0., 0.],
            [3., 1., 1.]])
        result = self.space.is_diagonal(base_point)
        expected = False
        self.assertAllClose(result, expected)

        diagonal = gs.eye(3)
        result = self.space.is_diagonal(diagonal)
        self.assertTrue(result)

        base_point = gs.stack([base_point, diagonal])
        result = self.space.is_diagonal(base_point)
        expected = gs.array([False, True])
        self.assertAllClose(result, expected)

        base_point = gs.stack([diagonal] * 2)
        result = self.space.is_diagonal(base_point)
        self.assertTrue(gs.all(result))

        base_point = gs.reshape(gs.arange(6), (2, 3))
        result = self.space.is_diagonal(base_point)
        self.assertTrue(~result)

    def test_norm(self):
        for n_samples in [1, 2]:
            mat = self.space.random_point(n_samples)
            result = self.metric.norm(mat)
            expected = self.space.frobenius_product(mat, mat) ** .5
            self.assertAllClose(result, expected)

    def test_diagonal(self):
        mat = gs.eye(3)
        result = Matrices.diagonal(mat)
        expected = gs.ones(3)
        self.assertAllClose(result, expected)

        mat = gs.stack([mat] * 2)
        result = Matrices.diagonal(mat)
        expected = gs.ones((2, 3))
        self.assertAllClose(result, expected)
