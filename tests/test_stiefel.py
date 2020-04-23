"""Unit tests for Stiefel manifolds."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel

ATOL = 1e-6

p_xy = gs.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
r_z = gs.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
point1 = gs.array([[1., 0.], [0., 1.], [0., 0.]])


class TestStiefel(geomstats.tests.TestCase):
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
            [1. / gs.sqrt(2.), 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1. / gs.sqrt(2.), 0., 0.]])

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

    def test_belongs_shape(self):
        point = self.space.random_uniform()
        belongs = self.space.belongs(point)

        self.assertAllClose(gs.shape(belongs), ())

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point, tolerance=1e-4)
        expected = True

        self.assertAllClose(result, expected)

    def test_random_uniform_shape(self):
        result = self.space.random_uniform()

        self.assertAllClose(gs.shape(result), (self.n, self.p))

    @geomstats.tests.np_only
    def test_log_and_exp(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

        Expect their composition to give the identity function.
        """
        # Riemannian Log then Riemannian Exp
        # General case
        base_point = self.point_a
        point = self.point_b

        log = self.metric.log(point=point, base_point=base_point)
        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected, atol=ATOL)

    def test_exp_and_belongs(self):
        base_point = self.point_a
        tangent_vec = self.tangent_vector_1

        exp = self.metric.exp(
            tangent_vec=tangent_vec,
            base_point=base_point)
        result = self.space.belongs(exp)
        expected = True
        self.assertAllClose(result, expected)

    def test_exp_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_base_point = self.point_a
        one_tangent_vec = self.tangent_vector_1

        n_base_points = gs.tile(
            gs.to_ndarray(self.point_a, to_ndim=3),
            (n_samples, 1, 1))
        n_tangent_vecs = gs.tile(
            gs.to_ndarray(self.tangent_vector_2, to_ndim=3),
            (n_samples, 1, 1))

        # With single tangent vec and base point
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertAllClose(gs.shape(result), (n, p))

        # With n_samples tangent vecs and base points
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.exp(one_tangent_vec, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    @geomstats.tests.np_and_tf_only
    def test_log_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.space.random_uniform()
        one_base_point = self.space.random_uniform()

        n_points = self.space.random_uniform(n_samples=n_samples)
        n_base_points = self.space.random_uniform(n_samples=n_samples)

        # With single point and base point
        result = self.metric.log(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (n, p))

        # With multiple points and base points
        result = self.metric.log(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.log(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.log(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    @geomstats.tests.np_only
    def test_retractation_and_lifting(self):
        """
        Test that the Riemannian exponential
        and the Riemannian logarithm are inverse.

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
        expected = point

        self.assertAllClose(result, expected, atol=ATOL)

        retract = self.metric.retraction(
            tangent_vec=tangent_vec, base_point=base_point)
        result = self.metric.lifting(point=retract, base_point=base_point)
        expected = tangent_vec

        self.assertAllClose(result, expected, atol=ATOL)

    @geomstats.tests.np_only
    def test_lifting_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        one_base_point = self.point_b
        n_points = gs.tile(
            gs.to_ndarray(self.point_a, to_ndim=3),
            (n_samples, 1, 1))
        n_base_points = gs.tile(
            gs.to_ndarray(self.point_b, to_ndim=3),
            (n_samples, 1, 1))

        result = self.metric.lifting(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (n, p))

        result = self.metric.lifting(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.lifting(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.lifting(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    @geomstats.tests.np_and_tf_only
    def test_retraction_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        n_points = gs.tile(
            gs.to_ndarray(one_point, to_ndim=3),
            (n_samples, 1, 1))
        one_tangent_vec = self.tangent_vector_1
        n_tangent_vecs = gs.tile(
            gs.to_ndarray(self.tangent_vector_2, to_ndim=3),
            (n_samples, 1, 1))

        result = self.metric.retraction(one_tangent_vec, one_point)
        self.assertAllClose(gs.shape(result), (n, p))

        result = self.metric.retraction(n_tangent_vecs, one_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.retraction(one_tangent_vec, n_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.retraction(n_tangent_vecs, n_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    def test_inner_product(self):
        base_point = self.point_a
        tangent_vector_1 = self.tangent_vector_1
        tangent_vector_2 = self.tangent_vector_2

        result = self.metric.inner_product(
            tangent_vector_1,
            tangent_vector_2,
            base_point=base_point)
        self.assertAllClose(gs.shape(result), ())

    def test_to_grassmannian(self):
        point2 = gs.array([[1., -1.], [1., 1.], [0., 0.]]) / gs.sqrt(2.)
        result = self.space.to_grassmannian(point2)
        expected = p_xy
        self.assertAllClose(result, expected)

    def test_to_grassmanniann_vectorized(self):
        inf_rots = gs.array([gs.pi * r_z / n for n in [2, 3, 4]])
        rots = GeneralLinear.exp(inf_rots)
        points = Matrices.mul(rots, point1)

        result = Stiefel.to_grassmannian(points)
        expected = gs.array([p_xy, p_xy, p_xy])
        self.assertAllClose(result, expected)
