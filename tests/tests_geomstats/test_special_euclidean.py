"""Unit tests for special euclidean group in matrix representation."""

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_euclidean import SpecialEuclidean,\
    SpecialEuclideanMatrixCannonicalLeftMetric,\
    SpecialEuclideanMatrixLieAlgebra


class TestSpecialEuclidean(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(12)
        self.n = 2
        self.group = SpecialEuclidean(n=self.n)
        self.n_samples = 3
        self.point = self.group.random_point(self.n_samples)
        self.tangent_vec = self.group.to_tangent(gs.random.rand(
            self.n_samples, self.group.n + 1, self.group.n + 1), self.point)

    def test_belongs(self):
        theta = gs.pi / 3
        point_1 = gs.array([
            [gs.cos(theta), - gs.sin(theta), 2.],
            [gs.sin(theta), gs.cos(theta), 3.],
            [0., 0., 1.]])
        result = self.group.belongs(point_1)
        self.assertTrue(result)

        point_2 = gs.array([
            [gs.cos(theta), - gs.sin(theta), 2.],
            [gs.sin(theta), gs.cos(theta), 3.],
            [0., 0., 0.]])
        result = self.group.belongs(point_2)
        self.assertFalse(result)

        point = gs.array([point_1, point_2])
        expected = gs.array([True, False])
        result = self.group.belongs(point)
        self.assertAllClose(result, expected)

        point = point_1[0]
        result = self.group.belongs(point)
        self.assertFalse(result)

        point = gs.zeros((2, 3))
        result = self.group.belongs(point)
        self.assertFalse(result)

        point = gs.zeros((2, 2, 3))
        result = self.group.belongs(point)
        self.assertFalse(gs.all(result))
        self.assertAllClose(result.shape, (2, ))

    def test_random_point_and_belongs(self):
        point = self.group.random_point()
        result = self.group.belongs(point)
        self.assertTrue(result)

        point = self.group.random_point(self.n_samples)
        result = self.group.belongs(point)
        expected = gs.array([True] * self.n_samples)
        self.assertAllClose(result, expected)

    def test_identity(self):
        result = self.group.identity
        expected = gs.eye(self.n + 1)
        self.assertAllClose(result, expected)

    def test_is_tangent(self):
        theta = gs.pi / 3
        vec_1 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 0.]])
        point = self.group.random_point()
        tangent_vec = self.group.compose(point, vec_1)
        result = self.group.is_tangent(tangent_vec, point)
        self.assertTrue(result)

        vec_2 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 1.]])
        tangent_vec = self.group.compose(point, vec_2)
        result = self.group.is_tangent(tangent_vec, point)
        self.assertFalse(result)

        vec = gs.array([vec_1, vec_2])
        expected = gs.array([True, False])
        result = self.group.is_tangent(vec)
        self.assertAllClose(result, expected)

    def test_to_tangent_vec_vectorization(self):
        n = self.group.n
        tangent_vecs = gs.arange(self.n_samples * (n + 1) ** 2)
        tangent_vecs = gs.cast(tangent_vecs, gs.float32)
        tangent_vecs = gs.reshape(
            tangent_vecs, (self.n_samples,) + (n + 1,) * 2)
        point = self.group.random_point(self.n_samples)
        tangent_vecs = Matrices.mul(point, tangent_vecs)
        regularized = self.group.to_tangent(tangent_vecs, point)
        result = Matrices.mul(
            Matrices.transpose(point), regularized) + \
            Matrices.mul(Matrices.transpose(regularized), point)
        result = result[:, :n, :n]
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_compose_and_inverse_matrix_form(self):
        point = self.group.random_point()
        inv_point = self.group.inverse(point)
        result = self.group.compose(point, inv_point)
        expected = self.group.identity
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            result = self.group.compose(inv_point, point)
            expected = self.group.identity
            self.assertAllClose(result, expected)

    def test_compose_vectorization(self):
        n_samples = self.n_samples
        n_points_a = self.group.random_point(n_samples=n_samples)
        n_points_b = self.group.random_point(n_samples=n_samples)
        one_point = self.group.random_point(n_samples=1)

        result = self.group.compose(one_point, n_points_a)
        self.assertAllClose(
            gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

        result = self.group.compose(n_points_a, one_point)

        if not geomstats.tests.tf_backend():
            self.assertAllClose(
                gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

            result = self.group.compose(n_points_a, n_points_b)
            self.assertAllClose(
                gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

    def test_inverse_vectorization(self):
        n_samples = self.n_samples
        points = self.group.random_point(n_samples=n_samples)
        result = self.group.inverse(points)
        self.assertAllClose(
            gs.shape(result), (n_samples,) + (self.group.n + 1,) * 2)

    def test_compose_matrix_form(self):
        point = self.group.random_point()
        result = self.group.compose(point, self.group.identity)
        expected = point
        self.assertAllClose(result, expected)

        if not geomstats.tests.tf_backend():
            # Composition by identity, on the left
            # Expect the original transformation
            result = self.group.compose(self.group.identity, point)
            expected = point
            self.assertAllClose(result, expected)

            # Composition of translations (no rotational part)
            # Expect the sum of the translations
            point_a = gs.array([[1., 0., 1.],
                                [0., 1., 1.5],
                                [0., 0., 1.]])
            point_b = gs.array([[1., 0., 2.],
                                [0., 1., 2.5],
                                [0., 0., 1.]])

            result = self.group.compose(point_a, point_b)
            last_line_0 = gs.array_from_sparse(
                [(0, 2), (1, 2)], [1., 1.], (3, 3))
            expected = point_a + point_b * last_line_0
            self.assertAllClose(result, expected)

    def test_left_exp_coincides(self):
        vector_group = SpecialEuclidean(n=2, point_type='vector')
        theta = gs.pi / 3
        initial_vec = gs.array([theta, 2., 2.])
        initial_matrix_vec = self.group.lie_algebra.matrix_representation(
            initial_vec)
        vector_exp = vector_group.left_canonical_metric.exp(initial_vec)
        result = self.group.left_canonical_metric.exp(initial_matrix_vec)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected)

    def test_right_exp_coincides(self):
        vector_group = SpecialEuclidean(n=2, point_type='vector')
        theta = gs.pi / 2
        initial_vec = gs.array([theta, 1., 1.])
        initial_matrix_vec = self.group.lie_algebra.matrix_representation(
            initial_vec)
        vector_exp = vector_group.right_canonical_metric.exp(initial_vec)
        result = self.group.right_canonical_metric.exp(
            initial_matrix_vec, n_steps=25)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected, atol=1e-6)

    def test_basis_belongs(self):
        lie_algebra = self.group.lie_algebra
        result = lie_algebra.belongs(lie_algebra.basis)
        self.assertTrue(gs.all(result))

    def test_basis_has_the_right_dimension(self):
        for n in range(2, 5):
            algebra = SpecialEuclideanMatrixLieAlgebra(n)
            self.assertEqual(int(n * (n + 1) / 2), algebra.dim)

    def test_belongs_lie_algebra(self):
        theta = gs.pi / 3
        vec_1 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 0.]])
        result = self.group.lie_algebra.belongs(vec_1)
        expected = True
        self.assertAllClose(result, expected)

        vec_2 = gs.array([
            [0., - theta, 2.],
            [theta, 0., 3.],
            [0., 0., 1.]])
        result = self.group.lie_algebra.belongs(vec_2)
        expected = False
        self.assertAllClose(result, expected)

        vec = gs.array([vec_1, vec_2])
        expected = gs.array([True, False])
        result = self.group.lie_algebra.belongs(vec)
        self.assertAllClose(result, expected)

    def test_basis_representation_is_correctly_vectorized(self):
        for n in range(2, 5):
            algebra = SpecialEuclideanMatrixLieAlgebra(n)
            shape = gs.shape(algebra.basis_representation(algebra.basis))
            dim = int(n * (n + 1) / 2)
            self.assertAllClose(shape, (dim, dim))

    def test_left_metric_wrong_group(self):
        group = self.group.rotations
        self.assertRaises(
            ValueError,
            lambda: SpecialEuclideanMatrixCannonicalLeftMetric(group))

        group = SpecialEuclidean(3, point_type='vector')
        self.assertRaises(
            ValueError,
            lambda: SpecialEuclideanMatrixCannonicalLeftMetric(group))

    def test_exp_and_belongs(self):
        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec, self.point)
        result = self.group.belongs(exp)
        self.assertTrue(gs.all(result))

        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec[0], self.point[0])
        result = self.group.belongs(exp)
        self.assertTrue(result)

    @geomstats.tests.np_and_tf_only
    def test_log_and_is_tan(self):
        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec, self.point)
        log = self.group.left_canonical_metric.log(exp, self.point)
        result = self.group.is_tangent(log, self.point)
        self.assertTrue(gs.all(result))

        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec[0], self.point[0])
        log = self.group.left_canonical_metric.log(exp, self.point)
        result = self.group.is_tangent(log, self.point)
        self.assertTrue(gs.all(result))

        log = self.group.left_canonical_metric.log(exp, self.point[0])
        result = self.group.is_tangent(log, self.point[0])
        self.assertTrue(result)

    @geomstats.tests.np_and_tf_only
    def test_exp_log(self):
        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec, self.point)
        result = self.group.left_canonical_metric.log(exp, self.point)
        self.assertAllClose(result, self.tangent_vec)

        exp = self.group.left_canonical_metric.exp(
            self.tangent_vec[0], self.point[0])
        result = self.group.left_canonical_metric.log(exp, self.point[0])
        self.assertAllClose(result, self.tangent_vec[0])

    def test_parallel_transport(self):
        metric = self.group.left_canonical_metric
        shape = (self.n_samples, self.group.n + 1, self.group.n + 1)

        results = helper.test_parallel_transport(self.group, metric, shape)
        for res in results:
            self.assertTrue(res)

    def test_lie_algebra_basis_belongs(self):
        basis = self.group.lie_algebra.basis
        result = self.group.lie_algebra.belongs(basis)
        self.assertTrue(gs.all(result))

    def test_lie_algebra_projection_and_belongs(self):
        vec = gs.random.rand(
            self.n_samples, self.group.n + 1, self.group.n + 1)
        tangent_vec = self.group.lie_algebra.projection(vec)
        result = self.group.lie_algebra.belongs(tangent_vec)
        self.assertTrue(gs.all(result))

    def test_basis_representation(self):
        vec = gs.random.rand(self.n_samples, self.group.dim)
        tangent_vec = self.group.lie_algebra.matrix_representation(vec)
        result = self.group.lie_algebra.basis_representation(tangent_vec)
        self.assertAllClose(result, vec)

        result = self.group.lie_algebra.basis_representation(tangent_vec[0])
        self.assertAllClose(result, vec[0])

    def test_metrics_expected_point_type(self):
        left = self.group.left_canonical_metric
        right = self.group.right_canonical_metric
        metric = self.group.metric
        for m in [left, right, metric]:
            self.assertTrue(m.default_point_type == 'matrix')

    def test_metric_left_invariant(self):
        group = self.group
        point = group.random_point()
        expected = group.left_canonical_metric.norm(self.tangent_vec)

        translated = group.tangent_translation_map(point)(self.tangent_vec)
        result = group.left_canonical_metric.norm(translated)
        self.assertAllClose(result, expected)

    def test_projection_and_belongs(self):
        shape = (self.n_samples, self.n + 1, self.n + 1)
        result = helper.test_projection_and_belongs(self.group, shape)
        for res in result:
            self.assertTrue(res)
