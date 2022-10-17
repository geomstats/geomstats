"""Unit tests for the manifold of complex matrices."""

import geomstats.backend as gs
import tests.conftest
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

CDTYPE = gs.get_default_cdtype()


class TestComplexMatrices(VectorSpaceTestCase, metaclass=Parametrizer):
    testing_data = ComplexMatricesTestData()
    Space = testing_data.Space

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).belongs(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_equal(self, m, n, mat1, mat2, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).equal(
                gs.array(mat1, dtype=CDTYPE), gs.array(mat2, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_mul(self, mat, expected):
        self.assertAllClose(ComplexMatrices.mul(*mat), gs.array(expected))

    def test_bracket(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.bracket(
                gs.array(mat_a, dtype=CDTYPE),
                gs.array(mat_b, dtype=CDTYPE),
            ),
            gs.array(expected),
        )

    def test_congruent(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.congruent(
                gs.array(mat_a, dtype=CDTYPE),
                gs.array(mat_b, dtype=CDTYPE),
            ),
            gs.array(expected),
        )

    def test_frobenius_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.frobenius_product(
                gs.array(mat_a, dtype=CDTYPE),
                gs.array(mat_b, dtype=CDTYPE),
            ),
            gs.array(expected),
        )

    def test_trace_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            ComplexMatrices.trace_product(
                gs.array(mat_a, dtype=CDTYPE),
                gs.array(mat_b, dtype=CDTYPE),
            ),
            gs.array(expected),
        )

    def test_flatten(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).flatten(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_transpose(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).transpose(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_transconjugate(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).transconjugate(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).diagonal(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_diagonal(gs.array(mat, dtype=CDTYPE)),
            expected,
        )

    def test_is_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_symmetric(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_hermitian(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_hermitian(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_skew_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_skew_symmetric(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_pd(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_pd(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_spd(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_spd(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_hpd(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_hpd(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_upper_triangular(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_lower_triangular(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_is_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_strictly_lower_triangular(
                gs.array(mat, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_is_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).is_strictly_upper_triangular(
                gs.array(mat, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_to_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_diagonal(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    @tests.conftest.np_autograd_and_torch_only
    def test_to_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_symmetric(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    @tests.conftest.np_autograd_and_torch_only
    def test_to_hermitian(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_hermitian(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_to_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_lower_triangular(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_to_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_upper_triangular(gs.array(mat, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_to_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_strictly_lower_triangular(
                gs.array(mat, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_to_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_strictly_upper_triangular(
                gs.array(mat, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_to_lower_triangular_diagonal_scaled(self, m, n, mat, expected):
        self.assertAllClose(
            ComplexMatrices(m, n).to_lower_triangular_diagonal_scaled(
                gs.array(mat, dtype=CDTYPE)
            ),
            gs.array(expected),
        )

    def test_flatten_reshape(self, m, n, mat):
        cls_mn = ComplexMatrices(m, n)
        self.assertAllClose(
            cls_mn.reshape(cls_mn.flatten(gs.array(mat, dtype=CDTYPE))),
            gs.array(mat, dtype=CDTYPE),
        )

    def test_to_matrix_type_is_matrix_type(self, m, n, matrix_type, mat):
        cls_mn = ComplexMatrices(m, n)
        to_function = getattr(cls_mn, "to_" + matrix_type)
        is_function = getattr(cls_mn, "is_" + matrix_type)
        self.assertAllClose(
            gs.all(is_function(to_function(gs.array(mat, dtype=CDTYPE)))), True
        )

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

    def test_inner_product(self, m, n, tangent_vec_a, tangent_vec_b, expected):
        self.assertAllClose(
            self.Metric(m, n).inner_product(
                gs.array(tangent_vec_a, dtype=CDTYPE),
                gs.array(tangent_vec_b, dtype=CDTYPE),
            ),
            gs.array(expected),
        )

    def test_norm(self, m, n, vector, expected):
        self.assertAllClose(
            self.Metric(m, n).norm(gs.array(vector, dtype=CDTYPE)),
            gs.array(expected),
        )

    def test_inner_product_norm(self, m, n, mat):
        self.assertAllClose(
            self.Metric(m, n).inner_product(mat, mat),
            gs.power(self.Metric(m, n).norm(mat), 2),
        )
