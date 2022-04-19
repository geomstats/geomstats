"""Unit tests for Lie algebra."""

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from tests.conftest import Parametrizer, TestCase
from tests.data.lie_algebra_data import TestDataLieAlgebra


class TestLieAlgebra(TestCase, metaclass=Parametrizer):

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
