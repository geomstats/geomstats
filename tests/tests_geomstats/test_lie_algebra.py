"""Unit tests for Lie algebra."""

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import TestCase
from tests.data_generation import TestData
from tests.parametrizers import Parametrizer


class TestLieAlgebra(TestCase, metaclass=Parametrizer):
    class TestDataLieAlgebra(TestData):
        def dimension_data(self):
            smoke_data = [dict(algebra=SkewSymmetricMatrices(4), expected=6)]
            return self.generate_tests(smoke_data)

        def matrix_representation_and_belongs_data(self):
            smoke_data = [
                dict(algebra=SkewSymmetricMatrices(4), point=gs.random.rand(2, 6))
            ]
            return self.generate_tests(smoke_data)

        def orthonormal_basis_data(self):
            smoke_data = [
                dict(group=SpecialOrthogonal(3), metric_mat_at_identity=None),
                dict(
                    group=SpecialOrthogonal(3),
                    metric_mat_at_identity=from_vector_to_diagonal_matrix(
                        gs.array([1.0, 2.0, 3.0])
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

        def orthonormal_basis_se3_data(self):
            smoke_data = [
                dict(group=SpecialEuclidean(3), metric_mat_at_identity=None),
                dict(
                    group=SpecialEuclidean(3),
                    metric_mat_at_identity=from_vector_to_diagonal_matrix(
                        gs.cast(gs.arange(1, SpecialEuclidean(3).dim + 1), gs.float32)
                    ),
                ),
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataLieAlgebra()

    def test_dimension(self, algebra, expected):
        self.assertAllClose(algebra.dim, expected)

    def test_matrix_representation_and_belongs(self, algebra, point):
        mat = algebra.matrix_representation(point)
        result = gs.all(algebra.belongs(mat))
        self.assertTrue(result)

    def test_orthonormal_basis(self, group, metric_mat_at_identity):
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group, metric_mat_at_identity)
        basis = metric.normal_basis(lie_algebra.basis)
        result = metric.inner_product_at_identity(basis[0], basis[1])
        self.assertAllClose(result, 0.0)

        result = metric.inner_product_at_identity(basis[1], basis[1])
        self.assertAllClose(result, 1.0)

    def test_orthonormal_basis_se3(self, group, metric_mat_at_identity):
        lie_algebra = group.lie_algebra
        metric = InvariantMetric(group, metric_mat_at_identity)
        basis = metric.normal_basis(lie_algebra.basis)
        for i, x in enumerate(basis):
            for y in basis[i:]:
                result = metric.inner_product_at_identity(x, y)
                expected = 0.0 if gs.any(x != y) else 1.0
                self.assertAllClose(result, expected)
