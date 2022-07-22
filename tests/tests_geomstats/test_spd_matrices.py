"""Unit tests for the manifold of symmetric positive definite matrices."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.lower_triangular_matrices import LowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricBuresWasserstein,
    SPDMetricEuclidean,
    SPDMetricLogEuclidean,
)
from tests.conftest import Parametrizer
from tests.data.spd_matrices_data import (
    SPDMatricesTestData,
    SPDMetricAffineTestData,
    SPDMetricBuresWassersteinTestData,
    SPDMetricEuclideanTestData,
    SPDMetricLogEuclideanTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestSPDMatrices(OpenSetTestCase, metaclass=Parametrizer):
    """Test of SPDMatrices methods."""

    space = SPDMatrices

    testing_data = SPDMatricesTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(SPDMatrices(n).belongs(gs.array(mat)), gs.array(expected))

    def test_projection(self, n, mat, expected):
        self.assertAllClose(
            SPDMatrices(n).projection(gs.array(mat)), gs.array(expected)
        )

    def test_logm(self, spd_mat, logm):
        self.assertAllClose(SPDMatrices.logm(gs.array(spd_mat)), gs.array(logm))

    def test_cholesky_factor(self, n, spd_mat, cf):
        result = SPDMatrices.cholesky_factor(gs.array(spd_mat))

        self.assertAllClose(result, gs.array(cf))
        self.assertTrue(gs.all(PositiveLowerTriangularMatrices(n).belongs(result)))

    def test_differential_cholesky_factor(self, n, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_cholesky_factor(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))
        self.assertTrue(gs.all(LowerTriangularMatrices(n).belongs(result)))

    def test_differential_power(self, power, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_power(
            power, gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_power(self, power, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_power(
            power, gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_differential_log(self, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_log(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_log(self, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_log(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_differential_exp(self, tangent_vec, base_point, expected):
        result = SPDMatrices.differential_exp(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inverse_differential_exp(self, tangent_vec, base_point, expected):
        result = SPDMatrices.inverse_differential_exp(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_cholesky_factor_belongs(self, n, mat):
        result = SPDMatrices(n).cholesky_factor(gs.array(mat))
        self.assertAllClose(
            gs.all(PositiveLowerTriangularMatrices(n).belongs(result)), True
        )


class TestSPDMetricAffine(RiemannianMetricTestCase, metaclass=Parametrizer):
    connection = metric = SPDMetricAffine
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SPDMetricAffineTestData()

    def test_inner_product(
        self, n, power_affine, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = SPDMetricAffine(n, power_affine)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, expected)

    def test_exp(self, n, power_affine, tangent_vec, base_point, expected):
        metric = SPDMetricAffine(n, power_affine)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, n, power_affine, point, base_point, expected):
        metric = SPDMetricAffine(n, power_affine)
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )


class TestSPDMetricBuresWasserstein(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = SPDMetricBuresWasserstein
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SPDMetricBuresWassersteinTestData()

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        metric = SPDMetricBuresWasserstein(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_autograd_only
    def test_parallel_transport(self, n):
        space = SPDMatrices(*n)
        metric = self.metric(*n)
        shape = (2, *n, *n)

        point = space.random_point(2)
        end_point = space.random_point(2)
        tan_b = gs.random.rand(*shape)
        tan_b = space.to_tangent(tan_b, point)

        # use a vector orthonormal to tan_b
        tan_a = gs.random.rand(*shape)
        tan_a = space.to_tangent(tan_a, point)

        # orthonormalize and move to base_point
        tan_a -= gs.einsum(
            "...,...ij->...ij",
            metric.inner_product(tan_a, tan_b, point)
            / metric.squared_norm(tan_b, point),
            tan_b,
        )
        tan_b = gs.einsum("...ij,...->...ij", tan_b, 1.0 / metric.norm(tan_b, point))
        tan_a = gs.einsum("...ij,...->...ij", tan_a, 1.0 / metric.norm(tan_a, point))

        transported = metric.parallel_transport(
            tan_a, point, end_point=end_point, n_steps=15, step="rk4"
        )
        result = metric.norm(transported, end_point)
        expected = metric.norm(tan_a, point)
        self.assertAllClose(result, expected)

        is_tangent = space.is_tangent(transported, end_point)
        self.assertTrue(gs.all(is_tangent))

        transported = metric.parallel_transport(
            tan_a, point, tan_b, n_steps=15, step="rk4"
        )

        end_point = metric.exp(tan_b, point)
        result = metric.norm(transported, end_point)
        expected = metric.norm(tan_a, point)
        self.assertAllClose(result, expected)

        is_tangent = space.is_tangent(transported, end_point)
        self.assertTrue(gs.all(is_tangent))


class TestSPDMetricEuclidean(RiemannianMetricTestCase, metaclass=Parametrizer):
    connection = metric = SPDMetricEuclidean
    skip_test_exp_geodesic_ivp = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_belongs = True
    skip_test_log_after_exp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SPDMetricEuclideanTestData()

    def test_inner_product(
        self, n, power_euclidean, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = SPDMetricEuclidean(n, power_euclidean)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    @geomstats.tests.np_autograd_and_tf_only
    def test_exp_domain(self, n, power_euclidean, tangent_vec, base_point, expected):
        metric = SPDMetricEuclidean(n, power_euclidean)
        result = metric.exp_domain(
            gs.array(tangent_vec), gs.array(base_point), expected
        )
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, power_euclidean, point, base_point, expected):
        metric = SPDMetricEuclidean(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_exp_after_log(self, n, power_euclidean, point, base_point):
        metric = SPDMetricEuclidean(n, power_euclidean)
        log = metric.log(gs.array(point), base_point=gs.array(base_point))
        result = metric.exp(tangent_vec=log, base_point=gs.array(base_point))
        self.assertAllClose(result, point, atol=gs.atol * 1000)

    def test_squared_dist_is_symmetric(self, n, power_euclidean, point_a, point_b):
        metric = SPDMetricEuclidean(n, power_euclidean)
        sd_a_b = metric.squared_dist(point_a, point_b)
        sd_b_a = metric.squared_dist(point_b, point_a)
        self.assertAllClose(sd_a_b, sd_b_a, atol=gs.atol * 100)

    def test_parallel_transport(
        self, n, power_euclidean, tangent_vec_a, base_point, tangent_vec_b
    ):
        metric = SPDMetricEuclidean(n, power_euclidean)
        result = metric.parallel_transport(tangent_vec_a, base_point, tangent_vec_b)
        self.assertAllClose(result, tangent_vec_a)


class TestSPDMetricLogEuclidean(RiemannianMetricTestCase, metaclass=Parametrizer):
    connection = metric = SPDMetricLogEuclidean
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_exp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SPDMetricLogEuclideanTestData()

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_dist(self, n, point_a, point_b, expected):
        metric = SPDMetricLogEuclidean(n)
        result = metric.dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))
