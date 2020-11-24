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
        self.matrices = self.space.embedding_manifold
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

    def test_random_uniform_and_belongs(self):
        """Test random uniform and belongs.

        Test that the random uniform method samples
        on the pre-shape space.
        """
        n_samples = self.n_samples
        point = self.space.random_uniform(n_samples)
        result = self.space.belongs(point)
        expected = gs.array([True] * n_samples)

        self.assertAllClose(expected, result)

    def test_random_uniform_shape(self):
        point = self.space.random_uniform()
        result = gs.shape(point)
        expected = (self.k_landmarks, self.m_ambient,)

        self.assertAllClose(result, expected)

        point = self.space.random_uniform(self.n_samples)
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
        point, vector = self.matrices.random_uniform(2)
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
        point = self.space.random_uniform()
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
        point = self.space.random_uniform(self.n_samples)
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
        point = self.space.random_uniform()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_projection_vectorized(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(transposed_point, horizontal)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    def test_horizontal_and_is_tangent(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(vector, point)
        horizontal = self.space.horizontal_projection(tan, point)

        horizontal = gs.stack([horizontal, vector])
        result = self.space.is_tangent(horizontal, point)
        expected = gs.array([True, False])

        self.assertAllClose(result, expected)

    def test_align(self):
        point, base_point = self.space.random_uniform(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(result)

    def test_align_vectorization(self):
        base_point = self.space.random_uniform()
        point = self.space.random_uniform(2)
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

        base_point = self.space.random_uniform(2)
        point = self.space.random_uniform()
        aligned = self.space.align(point, base_point)
        alignment = gs.matmul(Matrices.transpose(aligned), base_point)
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

    def test_inner_product_shape(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(vector, point)
        inner = self.space.metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_exp_and_belongs(self):
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(vector, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(result)

        exp = self.space.metric.exp(gs.zeros_like(point), point)
        result = gs.isclose(point, exp)
        self.assertTrue(gs.all(result))

    def test_exp_and_belongs_vectorization(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform(self.n_samples)
        tan = self.space.to_tangent(vector, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

        point = point[0]
        tan = self.space.to_tangent(vector, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

    def test_log_and_exp(self):
        point, base_point = self.space.random_uniform(2)
        log = self.space.metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        exp = self.space.metric.exp(log, base_point)
        self.assertAllClose(exp, point)

    def test_exp_and_log(self):
        base_point = self.space.random_uniform()
        vector = gs.random.rand(self.k_landmarks, self.m_ambient)
        tangent_vec = self.space.to_tangent(vector, base_point)
        point = self.space.metric.exp(tangent_vec, base_point)
        log = self.space.metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        self.assertAllClose(tangent_vec, log)

    def test_log_vectorization(self):
        point = self.space.random_uniform(self.n_samples)
        base_point = self.space.random_uniform()
        log = self.space.metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(gs.all(result))

        exp = self.space.metric.exp(log, base_point)
        self.assertAllClose(exp, point)

        log = self.space.metric.log(base_point, point)
        result = self.space.is_tangent(log, point)
        self.assertTrue(gs.all(result))

        exp = self.space.metric.exp(log, point)
        expected = gs.stack([base_point] * self.n_samples)
        self.assertAllClose(exp, expected)

    def test_kendall_inner_product_shape(self):
        vector = gs.random.rand(
            self.n_samples, self.k_landmarks, self.m_ambient)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(vector, point)
        inner = self.shape_metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_kendall_log_and_exp(self):
        point, base_point = self.space.random_uniform(2)
        expected = self.space.align(point, base_point)
        log = self.shape_metric.log(expected, base_point)
        result = self.space.is_horizontal(log, base_point)
        self.assertTrue(result)

        exp = self.shape_metric.exp(log, base_point)
        self.assertAllClose(exp, expected)

    def test_kendall_exp_and_log(self):
        base_point = self.space.random_uniform()
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
        point, base_point = self.space.random_uniform(2)
        result = self.shape_metric.dist(point, base_point)
        log = self.shape_metric.log(point, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)

    def test_dist_vectorization(self):
        point = self.space.random_uniform(self.n_samples)
        base_point = self.space.random_uniform(self.n_samples)
        aligned = self.space.align(point, base_point)
        result = self.shape_metric.dist(aligned, base_point)
        log = self.shape_metric.log(aligned, base_point)
        expected = self.shape_metric.norm(log, base_point)
        self.assertAllClose(result, expected)
