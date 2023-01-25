"""Unit tests for the manifold of matrices."""

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer
from tests.data.matrices_data import MatricesMetricTestData, MatricesTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestMatrices(VectorSpaceTestCase, metaclass=Parametrizer):

    testing_data = MatricesTestData()

    def test_belongs(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).belongs(gs.array(mat)), gs.array(expected))

    def test_equal(self, m, n, mat1, mat2, expected):
        self.assertAllClose(
            self.Space(m, n).equal(gs.array(mat1), gs.array(mat2)),
            gs.array(expected),
        )

    def test_mul(self, mat, expected):
        self.assertAllClose(Matrices.mul(*mat), gs.array(expected))

    def test_bracket(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.bracket(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_congruent(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.congruent(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_frobenius_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.frobenius_product(gs.array(mat_a), gs.array(mat_b)),
            gs.array(expected),
        )

    def test_trace_product(self, mat_a, mat_b, expected):
        self.assertAllClose(
            Matrices.trace_product(gs.array(mat_a), gs.array(mat_b)), gs.array(expected)
        )

    def test_flatten(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).flatten(gs.array(mat)), gs.array(expected))

    def test_transpose(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).transpose(gs.array(mat)), gs.array(expected)
        )

    def test_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).diagonal(gs.array(mat)), gs.array(expected)
        )

    def test_is_diagonal(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).is_diagonal(gs.array(mat)), expected)

    def test_is_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_symmetric(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_skew_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_skew_symmetric(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_pd(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).is_pd(gs.array(mat)), gs.array(expected))

    def test_is_spd(self, m, n, mat, expected):
        self.assertAllClose(self.Space(m, n).is_spd(gs.array(mat)), gs.array(expected))

    def test_is_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_strictly_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_is_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).is_strictly_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_diagonal(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_diagonal(gs.array(mat)), gs.array(expected)
        )

    def test_to_symmetric(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_symmetric(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_strictly_lower_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_strictly_lower_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_strictly_upper_triangular(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_strictly_upper_triangular(gs.array(mat)),
            gs.array(expected),
        )

    def test_to_lower_triangular_diagonal_scaled(self, m, n, mat, expected):
        self.assertAllClose(
            self.Space(m, n).to_lower_triangular_diagonal_scaled(gs.array(mat)),
            gs.array(expected),
        )

    def test_flatten_reshape(self, m, n, mat):
        cls_mn = self.Space(m, n)
        self.assertAllClose(
            cls_mn.reshape(cls_mn.flatten(gs.array(mat))), gs.array(mat)
        )

    def test_to_matrix_type_is_matrix_type(self, m, n, matrix_type, mat):
        cls_mn = self.Space(m, n)
        to_function = getattr(cls_mn, "to_" + matrix_type)
        is_function = getattr(cls_mn, "is_" + matrix_type)
        self.assertAllClose(gs.all(is_function(to_function(gs.array(mat)))), True)

    def test_basis(self, m, n, expected):
        result = self.Space(m, n).basis
        self.assertAllClose(result, expected)


class TestMatricesMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
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

    testing_data = MatricesMetricTestData()

    def test_inner_product(self, m, n, tangent_vec_a, tangent_vec_b, expected):
        self.assertAllClose(
            self.Metric(m, n).inner_product(
                gs.array(tangent_vec_a), gs.array(tangent_vec_b)
            ),
            gs.array(expected),
        )

    def test_norm(self, m, n, vector, expected):
        self.assertAllClose(
            self.Metric(m, n).norm(gs.array(vector)), gs.array(expected)
        )

    def test_inner_product_norm(self, m, n, mat):
        self.assertAllClose(
            self.Metric(m, n).inner_product(mat, mat),
            gs.power(self.Metric(m, n).norm(mat), 2),
        )
