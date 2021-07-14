"""Unit tests for the preshape space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeSpace


class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.k_landmarks = 4
        self.m_ambient = 3
        self.space = PreShapeSpace(self.k_landmarks, self.m_ambient)
        self.matrices = self.space.embedding_space
        self.n_samples = 10
        self.shape_metric = KendallShapeMetric(
            self.k_landmarks, self.m_ambient)

    def test_belongs(self):
        point = gs.random.rand(self.m_ambient - 1, self.k_landmarks)
        result = self.space.belongs(point)
        self.assertFalse(result)

        point = gs.random.rand(
            self.n_samples, self.m_ambient - 1, self.k_landmarks)
        result = self.space.belongs(point)
        self.assertFalse(gs.all(result))

    def test_random_point_and_belongs(self):
        """Test random uniform and belongs.

        Test that the random uniform method samples
        on the pre-shape space.
        """
        n_samples = self.n_samples
        point = self.space.random_point(n_samples)
        result = self.space.belongs(point)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    def test_random_point_shape(self):
        point = self.space.random_point()
        result = gs.shape(point)
        expected = (self.k_landmarks, self.m_ambient,)

        self.assertAllClose(result, expected)

        point = self.space.random_point(self.n_samples)
        result = gs.shape(point)
        expected = (self.n_samples, self.k_landmarks, self.m_ambient,)
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        point = Matrices.transpose(gs.array(
            [[1., 0., 0., 1.],
             [0., 1., 0., 1.],
             [0., 0., 1., 1.]]))
        proj = self.space.projection(point)
        result = self.space.belongs(proj)
        expected = True

        self.assertAllClose(expected, result)

    def test_is_centered(self):
        point = gs.ones((self.k_landmarks, self.m_ambient))
        result = self.space.is_centered(point)
        self.assertFalse(result)

        point = gs.zeros((self.k_landmarks, self.m_ambient))
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_center(self):
        point = gs.ones((self.k_landmarks, self.m_ambient))
        point = self.space.center(point)
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_centered_vectorization(self):
        point = gs.ones((self.n_samples, self.k_landmarks, self.m_ambient))
        point = self.space.center(point)
        result = gs.all(self.space.is_centered(point))
        self.assertTrue(result)

    def test_is_tangent_to_tangent(self):
        point, vector = self.matrices.random_point(2)
        point = self.space.projection(point)

        result = self.space.is_tangent(vector, point)
        self.assertFalse(result)

        tangent_vec = self.space.to_tangent(vector, point)
        result = self.space.is_tangent(tangent_vec, point)
        self.assertTrue(result)

        vec = gs.array([tangent_vec, vector])
        result = self.space.is_tangent(vec, point)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    def test_vertical_projection(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tan)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_vertical_projection_vectorization(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(transposed_point, tan)
        expected = Matrices.transpose(tmp_expected) - tmp_expected

        tmp_result = gs.matmul(transposed_point, vertical)
        result = Matrices.transpose(tmp_result) - tmp_result
        self.assertAllClose(result, expected)

    def test_horizontal_projection(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_projection_vectorized(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_and_is_tangent(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)

        horizontal = gs.stack([horizontal, vector])
        result = self.space.is_tangent(horizontal, point)
        expected = gs.array([True, False])

        self.assertAllClose(result, expected)

    def test_align(self):
        point, base_point = self.space.random_point(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(result)

    def test_align_vectorization(self):
        base_point = self.space.random_point()
        point = self.space.random_point(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

        base_point = self.space.random_point(2)
        point = self.space.random_point()
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

    def test_inner_product_shape(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        inner = self.space.ambient_metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_exp_and_belongs(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(result)

        exp = self.space.ambient_metric.exp(gs.zeros_like(point), point)
        result = gs.isclose(point, exp)
        self.assertTrue(gs.all(result))

    def test_exp_and_belongs_vectorization(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

        point = point[0]
        tan = self.space.to_tangent(vector, point)
        exp = self.space.ambient_metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

    def test_log_and_exp(self):
        point, base_point = self.space.random_point(2)
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        exp = self.space.ambient_metric.exp(log, base_point)
        self.assertAllClose(exp, point)

    def test_exp_and_log(self):
        base_point = self.space.random_point()
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        tangent_vec = self.space.to_tangent(vector, base_point)
        point = self.space.ambient_metric.exp(tangent_vec, base_point)
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        self.assertAllClose(tangent_vec, log)

    def test_log_vectorization(self):
        point = self.space.random_point(self.n_samples)
        base_point = self.space.random_point()
        log = self.space.ambient_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(gs.all(result))

        exp = self.space.ambient_metric.exp(log, base_point)
        self.assertAllClose(exp, point)

        log = self.space.ambient_metric.log(base_point, point)
        result = self.space.is_tangent(log, point)
        self.assertTrue(gs.all(result))

        exp = self.space.ambient_metric.exp(log, point)
        expected = gs.stack([base_point] * self.n_samples)
        self.assertAllClose(exp, expected)

    def test_kendall_inner_product_shape(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_point()
        tan = self.space.to_tangent(vector, point)
        inner = self.shape_metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_kendall_log_and_exp(self):
        point, base_point = self.space.random_point(2)
        expected = self.space.align(point, base_point)
        log = self.shape_metric.log(expected, base_point)
        result = self.space.is_horizontal(log, base_point)
        self.assertTrue(result)

        exp = self.shape_metric.exp(log, base_point)
        self.assertAllClose(exp, expected)

    def test_kendall_exp_and_log(self):
        base_point = self.space.random_point()
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        tangent_vec = self.space.to_tangent(vector, base_point)
        point = self.shape_metric.exp(tangent_vec, base_point)
        log = self.shape_metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        expected = self.space.horizontal_projection(tangent_vec, base_point)
        self.assertAllClose(expected, log)

    def test_dist_extreme_case(self):
        point = self.space.projection(gs.eye(self.k_landmarks, self.m_ambient))
        result = self.shape_metric.dist(point, point)
        expected = 0.
        self.assertAllClose(result, expected)

    def test_dist(self):
        point, base_point = self.space.random_point(2)
        result = self.shape_metric.dist(point, base_point)
        log = self.shape_metric.log(point, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)

    def test_dist_vectorization(self):
        point = self.space.random_point(self.n_samples)
        base_point = self.space.random_point(self.n_samples)
        aligned = self.space.align(point, base_point)
        result = self.shape_metric.dist(aligned, base_point)
        log = self.shape_metric.log(aligned, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)

    def test_curvature_is_skew_operator(self):
        space = self.space
        base_point = space.random_point(2)
        vector = gs.random.rand(
            4, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[:2], base_point)
        tangent_vec_b = space.to_tangent(vector[2:], base_point)

        result = self.shape_metric.curvature(
            tangent_vec_a, tangent_vec_a, tangent_vec_b, base_point)
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_curvature_bianchi_identity(self):
        space = self.space
        base_point = space.random_point()
        vector = gs.random.rand(
            3, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[0], base_point)
        tangent_vec_b = space.to_tangent(vector[1], base_point)
        tangent_vec_c = space.to_tangent(vector[2], base_point)

        curvature_1 = self.shape_metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point)
        curvature_2 = self.shape_metric.curvature(
            tangent_vec_b, tangent_vec_c, tangent_vec_a, base_point)
        curvature_3 = self.shape_metric.curvature(
            tangent_vec_c, tangent_vec_a, tangent_vec_b, base_point)

        result = curvature_1 + curvature_2 + curvature_3
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_integrability_tensor(self):
        space = self.space
        base_point = space.random_point()
        vector = gs.random.rand(
            2, self.k_landmarks, self.m_ambient)
        tangent_vec_a = space.to_tangent(vector[0], base_point)
        tangent_vec_b = space.to_tangent(vector[1], base_point)
        result_ab = space.integrability_tensor(
            tangent_vec_a, tangent_vec_b, base_point)

        result = space.ambient_metric.inner_product(
            tangent_vec_b, result_ab, base_point)
        expected = 0.
        self.assertAllClose(result, expected)

        horizontal_b = space.horizontal_projection(tangent_vec_b, base_point)
        horizontal_a = space.horizontal_projection(tangent_vec_a, base_point)
        result = space.integrability_tensor(
            horizontal_a, horizontal_b, base_point)
        expected = -space.integrability_tensor(
            horizontal_b, horizontal_a, base_point)
        self.assertAllClose(result, expected)

        is_vertical = space.is_vertical(result, base_point)
        self.assertTrue(is_vertical)

        vertical_b = tangent_vec_b - horizontal_b
        result = space.integrability_tensor(
            horizontal_a, vertical_b, base_point)
        is_horizontal = space.is_horizontal(result, base_point)
        self.assertTrue(is_horizontal)

    def test_kendall_directional_curvature(self):
        space = self.space
        kendall = self.shape_metric
        n_samples = 4 * self.k_landmarks * self.m_ambient
        base_point = self.space.random_point(1)

        vec_a = gs.random.rand(n_samples, self.k_landmarks, self.m_ambient)
        tg_vec_a = space.to_tangent(space.center(vec_a), base_point)
        hor_a = space.horizontal_projection(tg_vec_a, base_point)

        vec_b = gs.random.rand(n_samples, self.k_landmarks, self.m_ambient)
        tg_vec_b = space.to_tangent(space.center(vec_b), base_point)
        hor_b = space.horizontal_projection(tg_vec_b, base_point)

        tidal = kendall.directional_curvature(hor_a, hor_b, base_point)

        numerator = kendall.inner_product(tidal, hor_b, base_point)
        denominator = \
            kendall.inner_product(hor_a, hor_a, base_point) * \
            kendall.inner_product(hor_b, hor_b, base_point) - \
            kendall.inner_product(hor_a, hor_b, base_point) ** 2
        condition = ~gs.isclose(denominator, 0.)
        kappa = numerator[condition] / denominator[condition]
        kappa_direct = \
            kendall.sectional_curvature(hor_a, hor_b, base_point)[condition]
        self.assertAllClose(kappa, kappa_direct)
        result = (kappa > 1.0 - 1e-12)
        self.assertTrue(gs.all(result))

    def test_parallel_transport(self):
        space = self.space
        metric = self.shape_metric
        shape = (self.n_samples, self.k_landmarks, self.m_ambient)

        point = space.projection(gs.eye(4)[:, :3])
        tan_b = gs.random.rand(*shape)
        tan_b = space.to_tangent(tan_b, point)
        tan_b = space.horizontal_projection(tan_b, point)

        # use a vector orthonormal to tan_b
        tan_a = gs.random.rand(*shape)
        tan_a = space.to_tangent(tan_a, point)
        tan_a = space.horizontal_projection(tan_a, point)

        # orthonormalize and move to base_point
        tan_a -= gs.einsum(
            '...,...ij->...ij',
            metric.inner_product(tan_a, tan_b, point) / metric.squared_norm(
                tan_b, point), tan_b)
        tan_b = gs.einsum('...ij,...->...ij', tan_b,
                          1. / metric.norm(tan_b, point))
        tan_a = gs.einsum('...ij,...->...ij', tan_a,
                          1. / metric.norm(tan_a, point))

        transported = metric.parallel_transport(
            tan_a, tan_b, point, n_steps=100, step='rk4')
        end_point = metric.exp(tan_b, point)
        result = metric.norm(transported, end_point)
        expected = metric.norm(tan_a, point)
        self.assertAllClose(result, expected)

        is_tangent = space.is_tangent(transported, end_point)
        is_horizontal = space.is_horizontal(transported, end_point)
        self.assertTrue(gs.all(is_tangent))
        self.assertTrue(gs.all(is_horizontal))
