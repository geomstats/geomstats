"""Unit tests for the preshape space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.pre_shape import PreShapeSpace


class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.k_landmarks = 4
        self.m_ambient = 3
        self.space = PreShapeSpace(self.k_landmarks, self.m_ambient)
        self.matrices = self.space.embedding_manifold
        self.n_samples = 10

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

    def test_random_uniform(self):
        point = self.space.random_uniform()
        result = gs.shape(point)
        expected = (self.m_ambient, self.k_landmarks,)

        self.assertAllClose(result, expected)

        point = self.space.random_uniform(self.n_samples)
        result = gs.shape(point)
        expected = (self.n_samples, self.m_ambient, self.k_landmarks,)
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        point = gs.array(
            [[1., 0., 0., 1.],
             [0., 1., 0., 1.],
             [0., 0., 1., 1.]])
        proj = self.space.projection(point)
        result = self.space.belongs(proj)
        expected = True

        self.assertAllClose(expected, result)

    def test_is_centered(self):
        point = gs.ones((self.m_ambient, self.k_landmarks))
        result = self.space.is_centered(point)
        self.assertTrue(~ result)

        point = gs.zeros((self.m_ambient, self.k_landmarks))
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_center(self):
        point = gs.ones((self.m_ambient, self.k_landmarks))
        point = self.space.center(point)
        result = self.space.is_centered(point)
        self.assertTrue(result)

    def test_to_center_is_centered_vectorization(self):
        point = gs.ones((self.n_samples, self.m_ambient, self.k_landmarks))
        point = self.space.center(point)
        result = gs.all(self.space.is_centered(point))
        self.assertTrue(result)

    def test_is_tangent_to_tangent(self):
        point, vector = self.matrices.random_uniform(2)
        point = self.space.projection(point)

        result = self.space.is_tangent(vector, point)
        self.assertTrue(~ result)

        tangent_vec = self.space.to_tangent(vector, point)
        result = self.space.is_tangent(tangent_vec, point)
        self.assertTrue(result)

        vec = gs.array([tangent_vec, vector])
        result = self.space.is_tangent(vec, point)
        expected = gs.array([True, False])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_vertical_projection(self):
        w = gs.random.rand(self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(w, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(tan, transposed_point)
        expected = tmp_expected - Matrices.transpose(tmp_expected)

        tmp_result = gs.matmul(vertical, transposed_point)
        result = tmp_result - Matrices.transpose(tmp_result)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_vertical_projection_vectorization(self):
        w = gs.random.rand(self.n_samples, self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform(self.n_samples)
        tan = self.space.to_tangent(w, point)
        vertical = self.space.vertical_projection(tan, point)
        transposed_point = Matrices.transpose(point)

        tmp_expected = gs.matmul(tan, transposed_point)
        expected = tmp_expected - Matrices.transpose(tmp_expected)

        tmp_result = gs.matmul(vertical, transposed_point)
        result = tmp_result - Matrices.transpose(tmp_result)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_horizontal_projection(self):
        w = gs.random.rand(self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(w, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(horizontal, transposed_point)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_horizontal_projection_vectorized(self):
        w = gs.random.rand(self.n_samples, self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform(self.n_samples)
        tan = self.space.to_tangent(w, point)
        horizontal = self.space.horizontal_projection(tan, point)
        transposed_point = Matrices.transpose(point)
        result = gs.matmul(horizontal, transposed_point)
        expected = Matrices.transpose(result)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_horizontal_and_is_horizontal(self):
        w = gs.random.rand(self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(w, point)
        horizontal = self.space.horizontal_projection(tan, point)

        horizontal = gs.stack([horizontal, w])
        result = self.space.is_tangent(horizontal, point)
        expected = gs.array([True, False])

        self.assertAllClose(result, expected)

    def test_realign(self):
        point, base_point = self.space.random_uniform(2)
        aligned = self.space.realign(point, base_point)
        alignment = gs.matmul(aligned, Matrices.transpose(base_point))
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(result)

    def test_realign_vectorization(self):
        base_point = self.space.random_uniform()
        point = self.space.random_uniform(2)
        aligned = self.space.realign(point, base_point)
        alignment = gs.matmul(aligned, Matrices.transpose(base_point))
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

        base_point = self.space.random_uniform(2)
        point = self.space.random_uniform()
        aligned = self.space.realign(point, base_point)
        alignment = gs.matmul(aligned, Matrices.transpose(base_point))
        result = Matrices.is_symmetric(alignment)
        self.assertTrue(gs.all(result))

    def test_inner_product(self):
        w = gs.random.rand(self.n_samples, self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(w, point)
        inner = self.space.metric.inner_product(tan, tan, point)
        self.assertAllClose(inner.shape, (self.n_samples,))

    def test_exp_and_belongs(self):
        w = gs.random.rand(self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform()
        tan = self.space.to_tangent(w, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(result)

        exp = self.space.metric.exp(gs.zeros_like(point), point)
        result = gs.isclose(point, exp)
        self.assertTrue(gs.all(result))

    def test_exp_and_belongs_vectorization(self):
        w = gs.random.rand(self.n_samples, self.m_ambient, self.k_landmarks)
        point = self.space.random_uniform(self.n_samples)
        tan = self.space.to_tangent(w, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

        point = point[0]
        tan = self.space.to_tangent(w, point)
        exp = self.space.metric.exp(tan, point)
        result = self.space.belongs(exp)
        self.assertTrue(gs.all(result))

    def test_log(self):
        point, base_point = self.space.random_uniform(2)
        log = self.space.metric.log(point, base_point)
        result = self.space.is_tangent(log, base_point)
        self.assertTrue(result)

        exp = self.space.metric.exp(log, base_point)
        self.assertAllClose(exp, point)

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
