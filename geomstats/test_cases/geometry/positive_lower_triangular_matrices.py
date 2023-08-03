import pytest

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class PositiveLowerTriangularMatricesTestCase(OpenSetTestCase):
    def test_gram(self, point, expected, atol):
        res = self.space.gram(point)
        self.assertAllClose(res, expected, atol=atol)

    def test_differential_gram(self, tangent_vec, base_point, expected, atol):
        res = self.space.differential_gram(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_differential_gram(self, tangent_vec, base_point, expected, atol):
        res = self.space.inverse_differential_gram(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_gram_belongs_to_spd_matrices(self, n_points):
        point = self.data_generator.random_point(n_points)
        gram = self.space.gram(point)

        res = SPDMatrices(self.space.n).belongs(gram)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_differential_gram_belongs_to_symmetric_matrices(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        differential_gram = self.space.differential_gram(tangent_vec, base_point)
        res = SymmetricMatrices(self.space.n).belongs(differential_gram)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_inverse_differential_gram_belongs_to_lower_triangular_matrices(
        self, n_points
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        inverse_differential_gram = self.space.inverse_differential_gram(
            tangent_vec, base_point
        )
        res = self.space.embedding_space.belongs(inverse_differential_gram)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)


class CholeskyMetricTestCase(RiemannianMetricTestCase):
    def test_diag_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.diag_inner_product(
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_strictly_lower_inner_product(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        res = self.space.metric.strictly_lower_inner_product(
            tangent_vec_a,
            tangent_vec_b,
        )
        self.assertAllClose(res, expected, atol=atol)
