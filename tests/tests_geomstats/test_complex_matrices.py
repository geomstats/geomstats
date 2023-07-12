"""Unit tests for the manifold of complex matrices."""

import geomstats.backend as gs
from geomstats.geometry.complex_matrices import ComplexMatrices
from tests.conftest import Parametrizer
from tests.data.complex_matrices_data import (
    ComplexMatricesMetricTestData,
    ComplexMatricesTestData,
)
from tests.geometry_test_cases import (
    ComplexRiemannianMetricTestCase,
    VectorSpaceTestCase,
)


class TestComplexMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    testing_data = ComplexMatricesTestData()
    Space = testing_data.Space

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).belongs(mat),
            expected,
        )

    def test_congruent(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.congruent(
                mat_a,
                mat_b,
            ),
            expected,
        )

    def test_frobenius_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.frobenius_product(
                mat_a,
                mat_b,
            ),
            expected,
        )

    def test_transconjugate(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).transconjugate(mat),
            expected,
        )

    def test_is_hermitian(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_hermitian(mat),
            expected,
        )

    def test_is_hpd(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_hpd(mat),
            expected,
        )

    def test_to_hermitian(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_hermitian(mat),
            expected,
        )

    def test_to_matrix_type_is_matrix_type(self, m, n, matrix_type, mat):
        cls_mn = ComplexMatrices(m, n)
        to_function = getattr(cls_mn, "to_" + matrix_type)
        is_function = getattr(cls_mn, "is_" + matrix_type)
        self.assertAllClose(gs.all(is_function(to_function(mat))), True)

    def test_basis(self, m, n, expected):
        result = ComplexMatrices(m, n).basis
        self.assertAllClose(result, expected)


class TestComplexMatricesMetric(
    ComplexRiemannianMetricTestCase, metaclass=Parametrizer
):
    skip_test_inner_product_is_symmetric = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = ComplexMatricesMetricTestData()

    def test_inner_product(self, space, tangent_vec_a, tangent_vec_b, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(
            space.metric.inner_product(
                tangent_vec_a,
                tangent_vec_b,
            ),
            expected,
        )

    def test_norm(self, space, vector, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(
            space.metric.norm(vector),
            expected,
        )

    def test_inner_product_norm(self, space, n_points):
        space.equip_with_metric(self.Metric)
        mat = space.random_point(n_points)

        self.assertAllClose(
            space.metric.inner_product(mat, mat),
            gs.power(space.metric.norm(mat), 2),
        )
