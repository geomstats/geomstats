"""Unit tests for the invariant metrics on Lie groups."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices
from tests.conftest import Parametrizer, autograd_backend, np_backend
from tests.data.invariant_metric_data import InvariantMetricTestData
from tests.geometry_test_cases import RiemannianMetricTestCase


class TestInvariantMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_shape = np_backend()
    skip_test_geodesic_ivp_belongs = True
    skip_test_exp_ladder_parallel_transport = np_backend()
    skip_test_log_is_tangent = np_backend()
    skip_test_log_shape = np_backend()
    skip_test_geodesic_bvp_belongs = np_backend()
    skip_test_exp_after_log = np_backend() or autograd_backend()
    skip_test_geodesic_bvp_belongs = True
    skip_test_log_after_exp = True
    skip_test_dist_point_to_itself_is_zero = np_backend()
    skip_test_triangle_inequality_of_dist = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = InvariantMetricTestData()

    def test_inner_product_mat_at_identity_shape(
        self, group, metric_mat_at_identity, left_or_right
    ):
        metric = self.Metric(group, metric_mat_at_identity, left_or_right)
        dim = metric.group.dim
        result = metric.metric_mat_at_identity
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_shape(
        self, group, metric_mat_at_identity, left_or_right, base_point
    ):
        metric = self.Metric(group, metric_mat_at_identity, left_or_right)
        base_point = None
        dim = metric.group.dim
        result = metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

        base_point = group.identity
        dim = metric.group.dim
        result = metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_and_its_inverse(
        self, group, metric_mat_at_identity, left_or_right
    ):
        metric = self.Metric(group, metric_mat_at_identity, left_or_right)
        inner_prod_mat = metric.metric_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(group.dim)
        self.assertAllClose(result, expected)

    @geomstats.tests.np_autograd_and_torch_only
    def test_inner_product(
        self,
        group,
        metric_mat_at_identity,
        left_or_right,
        tangent_vec_a,
        tangent_vec_b,
        base_point,
        expected,
    ):
        metric = self.Metric(group, metric_mat_at_identity, left_or_right)
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = self.Metric(group=group)
        result = metric.structure_constant(tangent_vec_a, tangent_vec_b, tangent_vec_c)
        self.assertAllClose(result, expected)

    def test_dual_adjoint_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c
    ):
        metric = self.Metric(group=group)
        result = metric.inner_product_at_identity(
            metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(result, expected)

    def test_connection(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = self.Metric(group)
        self.assertAllClose(metric.connection(tangent_vec_a, tangent_vec_b), expected)

    def test_connection_translation_map(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        metric = self.Metric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

    def test_sectional_curvature(self, group, tangent_vec_a, tangent_vec_b, expected):
        metric = self.Metric(group)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(result, expected)

    def test_sectional_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        metric = self.Metric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_curvature(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        metric = self.Metric(group)
        result = metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None
        )
        self.assertAllClose(result, expected)

    def test_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, point, expected
    ):
        metric = self.Metric(group)
        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        result = metric.curvature(tan_a, tan_b, tan_c, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_curvature_derivative_at_identity(
        self,
        group,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        expected,
    ):
        metric = self.Metric(group)
        result = metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d
        )

        self.assertAllClose(result, expected)

    def test_curvature_derivative_tangent_translation_map(
        self,
        group,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        expected,
    ):
        metric = self.Metric(group=group)
        translation_map = group.tangent_translation_map(base_point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        tan_d = translation_map(tangent_vec_d)
        result = metric.curvature_derivative(tan_a, tan_b, tan_c, tan_d, base_point)
        self.assertAllClose(result, expected)

    def test_integrated_exp_at_id(
        self,
        group,
    ):
        metric = self.Metric(group=group)
        basis = metric.normal_basis(group.lie_algebra.basis)

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = group.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-4)

    def test_integrated_se3_exp_at_id(self, group):
        lie_algebra = group.lie_algebra
        metric = self.Metric(group=group)
        canonical_metric = group.left_canonical_metric
        basis = metric.normal_basis(lie_algebra.basis)

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        expected = canonical_metric.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

        result = metric.exp(tangent_vec, identity, n_steps=100, step="rk2")
        self.assertAllClose(expected, result, atol=1e-4)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_integrated_exp_and_log_at_id(self, group):
        metric = self.Metric(group=group)
        basis = group.lie_algebra.basis

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity

        exp = metric.exp(tangent_vec, identity, n_steps=100, step="rk4")
        result = metric.log(exp, identity, n_steps=15, step="rk4", verbose=False)
        self.assertAllClose(tangent_vec, result, atol=1e-5)

    def test_integrated_parallel_transport(self, group, n, n_samples):
        metric = self.Metric(group=group)
        point = group.identity
        tan_b = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_b = group.to_tangent(tan_b)

        # use a vector orthonormal to tan_b
        tan_a = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_a = group.to_tangent(tan_a)
        coef = metric.inner_product(tan_a, tan_b) / metric.squared_norm(tan_b)
        tan_a -= gs.einsum("...,...ij->...ij", coef, tan_b)
        tan_b = gs.einsum(
            "...ij,...->...ij", tan_b, 1.0 / metric.norm(tan_b, base_point=point)
        )
        tan_a = gs.einsum(
            "...ij,...->...ij", tan_a, 1.0 / metric.norm(tan_a, base_point=point)
        )

        expected = group.left_canonical_metric.parallel_transport(tan_a, point, tan_b)
        result, end_point_result = metric.parallel_transport(
            tan_a, point, tan_b, n_steps=20, step="rk4", return_endpoint=True
        )
        expected_end_point = metric.exp(tan_b, point, n_steps=20)

        self.assertAllClose(end_point_result, expected_end_point, atol=gs.atol * 1000)
        self.assertAllClose(expected, result, atol=gs.atol * 1000)

    def test_log_antipodals(self, group, rotation_mat1, rotation_mat2, expected):
        with expected:
            group.bi_invariant_metric.log(rotation_mat1, rotation_mat2)

    @geomstats.tests.np_autograd_and_tf_only
    def test_left_exp_and_exp_from_identity_left_diag_metrics(self, metric_args, point):
        metric = self.Metric(*metric_args)
        left_exp_from_id = metric.left_exp_from_identity(point)
        exp_from_id = metric.exp_from_identity(point)

        self.assertAllClose(left_exp_from_id, exp_from_id)

    @geomstats.tests.np_autograd_and_tf_only
    def test_left_log_and_log_from_identity_left_diag_metrics(self, metric_args, point):
        metric = self.Metric(*metric_args)
        left_log_from_id = metric.left_log_from_identity(point)
        log_from_id = metric.log_from_identity(point)

        self.assertAllClose(left_log_from_id, log_from_id)

    @geomstats.tests.np_autograd_and_tf_only
    def test_exp_log_composition_at_identity(self, metric_args, tangent_vec):
        metric = self.Metric(*metric_args)
        result = metric.left_log_from_identity(
            point=metric.left_exp_from_identity(tangent_vec=tangent_vec)
        )
        self.assertAllClose(result, tangent_vec)

    @geomstats.tests.np_autograd_and_tf_only
    def test_log_exp_composition_at_identity(self, metric_args, point):
        metric = self.Metric(*metric_args)
        result = metric.left_exp_from_identity(
            tangent_vec=metric.left_log_from_identity(point=point)
        )
        self.assertAllClose(result, point)
