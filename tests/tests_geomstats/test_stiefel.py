"""Unit tests for Stiefel manifolds."""

import warnings

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.stiefel import Stiefel

p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
point1 = gs.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


class TestStiefel(geomstats.tests.TestCase):
    def setUp(self):
        """
        Tangent vectors constructed following:
        http://noodle.med.yale.edu/hdtag/notes/steifel_notes.pdf
        """
        warnings.filterwarnings("ignore")

        gs.random.seed(1234)

        self.p = 3
        self.n = 4
        self.space = Stiefel(self.n, self.p)
        self.n_samples = 10
        self.dimension = int(self.p * self.n - (self.p * (self.p + 1) / 2))

        self.point_a = gs.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )

        self.point_b = gs.array(
            [
                [1.0 / gs.sqrt(2.0), 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0 / gs.sqrt(2.0), 0.0, 0.0],
            ]
        )

        point_perp = gs.array([[0.0], [0.0], [0.0], [1.0]])

        matrix_a_1 = gs.array([[0.0, 2.0, -5.0], [-2.0, 0.0, -1.0], [5.0, 1.0, 0.0]])

        matrix_b_1 = gs.array([[-2.0, 1.0, 4.0]])

        matrix_a_2 = gs.array([[0.0, 2.0, -5.0], [-2.0, 0.0, -1.0], [5.0, 1.0, 0.0]])

        matrix_b_2 = gs.array([[-2.0, 1.0, 4.0]])

        self.tangent_vector_1 = gs.matmul(self.point_a, matrix_a_1) + gs.matmul(
            point_perp, matrix_b_1
        )

        self.tangent_vector_2 = gs.matmul(self.point_a, matrix_a_2) + gs.matmul(
            point_perp, matrix_b_2
        )

        self.metric = self.space.canonical_metric

    def test_belongs_shape(self):
        point = self.space.random_uniform()
        belongs = self.space.belongs(point)

        self.assertAllClose(gs.shape(belongs), ())

    def test_random_uniform_and_belongs(self):
        point = self.space.random_uniform()
        result = self.space.belongs(point)
        expected = True

        self.assertAllClose(result, expected)

    def test_random_uniform_shape(self):
        result = self.space.random_uniform()

        self.assertAllClose(gs.shape(result), (self.n, self.p))

    @geomstats.tests.np_and_autograd_only
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

        base_point = gs.cast(point, gs.float64)
        point = gs.cast(point, gs.float64)

        log = self.metric.log(point=point, base_point=base_point)
        is_tangent = self.space.is_tangent(log, base_point)
        self.assertTrue(is_tangent)

        result = self.metric.exp(tangent_vec=log, base_point=base_point)
        expected = point
        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        base_point = self.point_a
        tangent_vec = self.tangent_vector_1

        exp = self.metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = self.space.belongs(exp)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_tf_only
    def test_exp_and_log(self):
        base_point = self.space.random_uniform()
        vector = gs.random.rand(*base_point.shape)
        tangent_vec = self.space.to_tangent(vector, base_point) / 4
        point = self.metric.exp(tangent_vec, base_point)
        result = self.metric.log(point, base_point, max_iter=32)
        self.assertAllClose(result, tangent_vec)

    def test_exp_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_base_point = self.point_a
        one_tangent_vec = self.tangent_vector_1

        n_base_points = gs.tile(
            gs.to_ndarray(self.point_a, to_ndim=3), (n_samples, 1, 1)
        )
        n_tangent_vecs = gs.tile(
            gs.to_ndarray(self.tangent_vector_2, to_ndim=3), (n_samples, 1, 1)
        )

        # With single tangent vec and base point
        result = self.metric.exp(one_tangent_vec, one_base_point)
        self.assertAllClose(gs.shape(result), (n, p))

        # With n_samples tangent vecs and base points
        result = self.metric.exp(n_tangent_vecs, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.exp(n_tangent_vecs, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    @geomstats.tests.np_autograd_and_tf_only
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

    @geomstats.tests.np_autograd_and_tf_only
    def test_log_and_exp_random(self):
        base_point, point = self.space.random_uniform(2)
        log = self.metric.log(point, base_point)
        result = self.metric.exp(log, base_point)
        self.assertAllClose(result, point, rtol=1e-05, atol=1e-05)

    @geomstats.tests.np_and_autograd_only
    def test_retraction_and_lifting(self):
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
        result = self.metric.retraction(tangent_vec=lifted, base_point=base_point)
        expected = point

        self.assertAllClose(result, expected)

        retract = self.metric.retraction(tangent_vec=tangent_vec, base_point=base_point)
        result = self.metric.lifting(point=retract, base_point=base_point)
        expected = tangent_vec

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_lifting_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        one_base_point = self.point_b
        n_points = gs.tile(gs.to_ndarray(self.point_a, to_ndim=3), (n_samples, 1, 1))
        n_base_points = gs.tile(
            gs.to_ndarray(self.point_b, to_ndim=3), (n_samples, 1, 1)
        )

        result = self.metric.lifting(one_point, one_base_point)
        self.assertAllClose(gs.shape(result), (n, p))

        result = self.metric.lifting(n_points, one_base_point)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.lifting(one_point, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

        result = self.metric.lifting(n_points, n_base_points)
        self.assertAllClose(gs.shape(result), (n_samples, n, p))

    @geomstats.tests.np_autograd_and_tf_only
    def test_retraction_vectorization_shape(self):
        n_samples = self.n_samples
        n = self.n
        p = self.p

        one_point = self.point_a
        n_points = gs.tile(gs.to_ndarray(one_point, to_ndim=3), (n_samples, 1, 1))
        one_tangent_vec = self.tangent_vector_1
        n_tangent_vecs = gs.tile(
            gs.to_ndarray(self.tangent_vector_2, to_ndim=3), (n_samples, 1, 1)
        )

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
            tangent_vector_1, tangent_vector_2, base_point=base_point
        )
        self.assertAllClose(gs.shape(result), ())

    def test_to_grassmannian(self):
        point2 = gs.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]) / gs.sqrt(2.0)
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

    def test_to_tangent_is_tangent(self):
        point = self.space.random_uniform()
        vector = gs.random.rand(*point.shape) / 4
        tangent_vec = self.space.to_tangent(vector, point)
        result = self.space.is_tangent(tangent_vec, point)
        self.assertTrue(result)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n, self.p)
        result = helper.test_projection_and_belongs(self.space, shape)
        for res in result:
            self.assertTrue(res)
