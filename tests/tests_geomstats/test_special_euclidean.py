"""Unit tests for special euclidean group in matrix representation."""

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_euclidean import SpecialEuclidean


class TestSpecialEuclidean(geomstats.tests.TestCase):
    def setup_method(self):
        gs.random.seed(12)
        self.n = 2
        self.group = SpecialEuclidean(n=self.n)
        self.n_samples = 3
        self.point = self.group.random_point(self.n_samples)
        self.tangent_vec = self.group.to_tangent(
            gs.random.rand(self.n_samples, self.group.n + 1, self.group.n + 1),
            self.point,
        )

    def test_to_tangent_vec_vectorization(self):
        n = self.group.n
        tangent_vecs = gs.arange(self.n_samples * (n + 1) ** 2)
        tangent_vecs = gs.cast(tangent_vecs, gs.float32)
        tangent_vecs = gs.reshape(tangent_vecs, (self.n_samples,) + (n + 1,) * 2)
        point = self.group.random_point(self.n_samples)
        tangent_vecs = Matrices.mul(point, tangent_vecs)
        regularized = self.group.to_tangent(tangent_vecs, point)
        result = Matrices.mul(Matrices.transpose(point), regularized) + Matrices.mul(
            Matrices.transpose(regularized), point
        )
        result = result[:, :n, :n]
        expected = gs.zeros_like(result)
        self.assertAllClose(result, expected)

    def test_left_exp_coincides(self):
        vector_group = SpecialEuclidean(n=2, point_type="vector")
        theta = gs.pi / 3
        initial_vec = gs.array([theta, 2.0, 2.0])
        initial_matrix_vec = self.group.lie_algebra.matrix_representation(initial_vec)
        vector_exp = vector_group.left_canonical_metric.exp(initial_vec)
        result = self.group.left_canonical_metric.exp(initial_matrix_vec)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected)

    def test_right_exp_coincides(self):
        vector_group = SpecialEuclidean(n=2, point_type="vector")
        theta = gs.pi / 2
        initial_vec = gs.array([theta, 1.0, 1.0])
        initial_matrix_vec = self.group.lie_algebra.matrix_representation(initial_vec)
        vector_exp = vector_group.right_canonical_metric.exp(initial_vec)
        result = self.group.right_canonical_metric.exp(initial_matrix_vec, n_steps=25)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected, atol=1e-6)

    def test_parallel_transport(self):
        metric = self.group.left_canonical_metric
        shape = (self.n_samples, self.group.n + 1, self.group.n + 1)

        results = helper.test_parallel_transport(self.group, metric, shape)
        for res in results:
            self.assertTrue(res)

    def test_metric_left_invariant(self):
        group = self.group
        point = group.random_point()
        expected = group.left_canonical_metric.norm(self.tangent_vec)

        translated = group.tangent_translation_map(point)(self.tangent_vec)
        result = group.left_canonical_metric.norm(translated)
        self.assertAllClose(result, expected)
