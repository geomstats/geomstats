"""Unit tests for special euclidean group in matrix representation."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.special_euclidean import SpecialEuclidean
from tests.conftest import Parametrizer, TestCase, np_backend, tf_backend
from tests.data.special_euclidean_data import (
    SpecialEuclidean3VectorsTestData,
    SpecialEuclideanMatrixCanonicalLeftMetricTestData,
    SpecialEuclideanMatrixCanonicalRightMetricTestData,
    SpecialEuclideanMatrixLieAlgebraTestData,
    SpecialEuclideanTestData,
)
from tests.geometry_test_cases import (
    InvariantMetricTestCase,
    LieGroupTestCase,
    MatrixLieAlgebraTestCase,
)


class TestSpecialEuclidean(LieGroupTestCase, metaclass=Parametrizer):
    skip_test_log_after_exp = tf_backend()
    skip_test_exp_after_log = tf_backend()
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SpecialEuclideanTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(
            SpecialEuclidean(n).belongs(gs.array(mat)), gs.array(expected)
        )

    def test_identity(self, n, expected):
        self.assertAllClose(SpecialEuclidean(n).identity, gs.array(expected))

    def test_is_tangent(self, n, tangent_vec, base_point, expected):
        result = SpecialEuclidean(n).is_tangent(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_metrics_default_point_type(self, n, metric_str):
        group = self.Space(n)
        self.assertTrue(getattr(group, metric_str).default_point_type == "matrix")

    def test_inverse_shape(self, n, points, expected):
        group = self.Space(n)
        self.assertAllClose(gs.shape(group.inverse(points)), expected)

    def test_compose_shape(self, n, point_a, point_b, expected):
        group = self.Space(n)
        result = gs.shape(group.compose(gs.array(point_a), gs.array(point_b)))
        self.assertAllClose(result, expected)

    def test_regularize_shape(self, n, point_type, n_samples):
        group = self.Space(n, point_type)
        points = group.random_point(n_samples=n_samples)
        regularized_points = group.regularize(points)

        self.assertAllClose(
            gs.shape(regularized_points),
            (n_samples, *group.get_point_type_shape()),
        )

    def test_compose(self, n, point_type, point_1, point_2, expected):
        group = self.Space(n, point_type)
        result = group.compose(point_1, point_2)
        self.assertAllClose(result, expected)

    def test_group_exp_from_identity(self, n, point_type, tangent_vec, expected):
        group = self.Space(n, point_type)
        result = group.exp(base_point=group.identity, tangent_vec=tangent_vec)
        self.assertAllClose(result, expected)

    def test_group_log_from_identity(self, n, point_type, point, expected):
        group = self.Space(n, point_type)
        result = group.log(base_point=group.identity, point=point)
        self.assertAllClose(result, expected)


class TestSpecialEuclideanMatrixLieAlgebra(
    MatrixLieAlgebraTestCase, metaclass=Parametrizer
):
    testing_data = SpecialEuclideanMatrixLieAlgebraTestData()

    def test_dim(self, n, expected):
        algebra = self.Space(n)
        self.assertAllClose(algebra.dim, expected)

    def test_belongs(self, n, vec, expected):
        algebra = self.Space(n)
        self.assertAllClose(algebra.belongs(gs.array(vec)), gs.array(expected))


class TestSpecialEuclideanMatrixCanonicalLeftMetric(
    InvariantMetricTestCase,
    metaclass=Parametrizer,
):

    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SpecialEuclideanMatrixCanonicalLeftMetricTestData()

    def test_left_metric_wrong_group(self, group, expected):
        with expected:
            self.Metric(group)


class TestSpecialEuclideanMatrixCanonicalRightMetric(
    InvariantMetricTestCase,
    metaclass=Parametrizer,
):
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = np_backend()
    skip_test_log_shape = np_backend()
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_squared_dist_is_symmetric = np_backend()
    skip_test_log_after_exp = True
    skip_test_exp_after_log = True
    skip_test_log_is_tangent = np_backend()
    skip_test_geodesic_bvp_belongs = np_backend()
    skip_test_exp_ladder_parallel_transport = np_backend()
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_belongs = np_backend()
    skip_test_squared_dist_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_dist_is_positive = np_backend()
    skip_test_triangle_inequality_of_dist = np_backend()
    skip_test_dist_is_symmetric = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_squared_dist_is_positive = np_backend()
    skip_test_exp_after_log_at_identity = np_backend()
    skip_test_log_after_exp_at_identity = np_backend()
    skip_test_log_at_identity_belongs_to_lie_algebra = np_backend()
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SpecialEuclideanMatrixCanonicalRightMetricTestData()

    def test_right_exp_coincides(self, n, initial_vec):
        group = SpecialEuclidean(n=n)
        vector_group = SpecialEuclidean(n=n, point_type="vector")
        initial_matrix_vec = group.lie_algebra.matrix_representation(initial_vec)
        vector_exp = vector_group.right_canonical_metric.exp(initial_vec)
        result = group.right_canonical_metric.exp(initial_matrix_vec, n_steps=25)
        expected = vector_group.matrix_from_vector(vector_exp)
        self.assertAllClose(result, expected, atol=1e-6)


class TestSpecialEuclidean3Vectors(TestCase, metaclass=Parametrizer):
    testing_data = SpecialEuclidean3VectorsTestData()

    @tests.conftest.np_and_autograd_only
    def test_exp_after_log(self, metric, point, base_point, atol):
        """
        Test that the Riemannian right exponential and the
        Riemannian right logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.exp(metric.log(point, base_point), base_point)
        expected = group.regularize(point)
        expected = gs.cast(expected, gs.float64)
        norm = gs.linalg.norm(expected)
        if norm != 0:
            atol *= norm
        self.assertAllClose(result, expected, atol=atol)

    @tests.conftest.np_and_autograd_only
    def test_exp_after_log_right_with_angles_close_to_pi(
        self,
        metric,
        point,
        base_point,
        atol,
    ):
        group = SpecialEuclidean(3, "vector")
        result = metric.exp(metric.log(point, base_point), base_point)
        expected = group.regularize(point)

        inv_expected = gs.concatenate([-expected[:3], expected[3:6]])

        norm = gs.linalg.norm(expected)
        if norm != 0:
            atol *= norm

        self.assertTrue(
            gs.allclose(result, expected, atol=atol)
            or gs.allclose(result, inv_expected, atol=atol)
        )

    @tests.conftest.np_and_autograd_only
    def test_log_after_exp_with_angles_close_to_pi(
        self, metric, tangent_vec, base_point, atol
    ):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.log(metric.exp(tangent_vec, base_point), base_point)

        expected = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point, metric=metric
        )

        inv_expected = gs.concatenate([-expected[:3], expected[3:6]])

        norm = gs.linalg.norm(expected)
        if norm != 0:
            atol *= norm

        self.assertTrue(
            gs.allclose(result, expected, atol=atol)
            or gs.allclose(result, inv_expected, atol=atol)
        )

    @tests.conftest.np_and_autograd_only
    def test_log_after_exp(self, metric, tangent_vec, base_point, atol):
        """
        Test that the Riemannian left exponential and the
        Riemannian left logarithm are inverse.
        Expect their composition to give the identity function.
        """
        group = SpecialEuclidean(3, "vector")
        result = metric.log(metric.exp(tangent_vec, base_point), base_point)

        expected = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point, metric=metric
        )

        norm = gs.linalg.norm(expected)
        if norm != 0:
            atol *= norm
        self.assertAllClose(result, expected, atol=atol)

    @tests.conftest.np_and_autograd_only
    def test_exp(self, metric, base_point, tangent_vec, expected):
        result = metric.exp(base_point=base_point, tangent_vec=tangent_vec)
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_log(self, metric, point, base_point, expected):
        result = metric.log(point, base_point)
        self.assertAllClose(result, expected)

    @tests.conftest.np_autograd_and_tf_only
    def test_regularize_extreme_cases(self, point, expected):
        group = SpecialEuclidean(3, "vector")
        result = group.regularize(point)
        self.assertAllClose(result, expected)
