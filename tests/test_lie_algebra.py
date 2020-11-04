"""Unit tests for Lie algebra."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class TestLieAlgebra(geomstats.tests.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = int(self.n * (self.n - 1) / 2)
        self.algebra = SkewSymmetricMatrices(n=self.n)

    def test_dimension(self):
        result = self.algebra.dim
        expected = self.dim
        self.assertAllClose(result, expected)

    def test_matrix_representation_and_belongs(self):
        n_samples = 2
        point = gs.random.rand(n_samples * self.dim)
        point = gs.reshape(point, (n_samples, self.dim))
        mat = self.algebra.matrix_representation(point)
        result = gs.all(self.algebra.belongs(mat))
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_basis_and_matrix_representation(self):
        n_samples = 2
        expected = gs.random.rand(n_samples * self.dim)
        expected = gs.reshape(expected, (n_samples, self.dim))
        mat = self.algebra.matrix_representation(expected)
        result = self.algebra.basis_representation(mat)
        self.assertAllClose(result, expected)

    def test_orthonormal_basis(self):
        group = SpecialOrthogonal(3)
        lie_algebra = SkewSymmetricMatrices(3)
        metric = InvariantMetric(group=group)
        basis = lie_algebra.orthonormal_basis(metric.metric_mat_at_identity)
        result = metric.inner_product_at_identity(basis[0], basis[1])
        self.assertAllClose(result, 0.)

        result = metric.inner_product_at_identity(basis[1], basis[1])
        self.assertAllClose(result, 1.)

        metric_mat = from_vector_to_diagonal_matrix(
            gs.array([1., 2., 3.]))
        metric = InvariantMetric(
            group=group,
            metric_mat_at_identity=metric_mat)
        basis = lie_algebra.orthonormal_basis(metric.metric_mat_at_identity)
        result = metric.inner_product_at_identity(basis[0], basis[1])
        self.assertAllClose(result, 0.)

        result = metric.inner_product_at_identity(basis[1], basis[1])
        self.assertAllClose(result, 1.)
