"""Unit tests for the invariant metrics on Lie groups."""

import geomstats.backend as gs
import tests.conftest
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
        self, group, metric_mat_at_identity, left
    ):
        group.equip_with_metric(
            self.Metric, metric_mat_at_identity=metric_mat_at_identity, left=left
        )

        dim = group.dim
        result = group.metric.metric_mat_at_identity
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_shape(
        self, group, metric_mat_at_identity, left, base_point
    ):
        group.equip_with_metric(
            self.Metric, metric_mat_at_identity=metric_mat_at_identity, left=left
        )

        base_point = None
        dim = group.dim
        result = group.metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

        base_point = group.identity
        dim = group.dim
        result = group.metric.metric_matrix(base_point=base_point)
        self.assertAllClose(gs.shape(result), (dim, dim))

    def test_inner_product_matrix_and_its_inverse(
        self, group, metric_mat_at_identity, left
    ):
        group.equip_with_metric(
            self.Metric, metric_mat_at_identity=metric_mat_at_identity, left=left
        )

        inner_prod_mat = group.metric.metric_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        result = gs.matmul(inv_inner_prod_mat, inner_prod_mat)
        expected = gs.eye(group.dim)
        self.assertAllClose(result, expected)

    def test_inner_product(
        self,
        group,
        metric_mat_at_identity,
        left,
        tangent_vec_a,
        tangent_vec_b,
        base_point,
        expected,
    ):
        group.equip_with_metric(
            self.Metric, metric_mat_at_identity=metric_mat_at_identity, left=left
        )

        result = group.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        group.equip_with_metric(self.Metric)

        result = group.metric.structure_constant(
            tangent_vec_a, tangent_vec_b, tangent_vec_c
        )
        self.assertAllClose(result, expected)

    def test_dual_adjoint_structure_constant(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c
    ):
        group.equip_with_metric(self.Metric)

        result = group.metric.inner_product_at_identity(
            group.metric.dual_adjoint(tangent_vec_a, tangent_vec_b), tangent_vec_c
        )
        expected = group.metric.structure_constant(
            tangent_vec_a, tangent_vec_c, tangent_vec_b
        )
        self.assertAllClose(result, expected)

    def test_connection(self, group, tangent_vec_a, tangent_vec_b, expected):
        group.equip_with_metric(self.Metric)

        self.assertAllClose(
            group.metric.connection(tangent_vec_a, tangent_vec_b), expected
        )

    def test_connection_translation_map(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        group.equip_with_metric(self.Metric)

        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = group.metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

    def test_sectional_curvature(self, group, tangent_vec_a, tangent_vec_b, expected):
        group.equip_with_metric(self.Metric)

        result = group.metric.sectional_curvature(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(result, expected)

    def test_sectional_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, point, expected
    ):
        group.equip_with_metric(self.Metric)

        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        result = group.metric.connection(tan_a, tan_b, point)
        expected = translation_map(expected)
        self.assertAllClose(result, expected)

    def test_curvature(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, expected
    ):
        group.equip_with_metric(self.Metric)

        result = group.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point=None
        )
        self.assertAllClose(result, expected)

    def test_curvature_translation_point(
        self, group, tangent_vec_a, tangent_vec_b, tangent_vec_c, point, expected
    ):
        group.equip_with_metric(self.Metric)

        translation_map = group.tangent_translation_map(point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        result = group.metric.curvature(tan_a, tan_b, tan_c, point)
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
        group.equip_with_metric(self.Metric)

        result = group.metric.curvature_derivative(
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
        group.equip_with_metric(self.Metric)

        translation_map = group.tangent_translation_map(base_point)
        tan_a = translation_map(tangent_vec_a)
        tan_b = translation_map(tangent_vec_b)
        tan_c = translation_map(tangent_vec_c)
        tan_d = translation_map(tangent_vec_d)
        result = group.metric.curvature_derivative(
            tan_a, tan_b, tan_c, tan_d, base_point
        )
        self.assertAllClose(result, expected)

    def test_integrated_exp_at_id(
        self,
        group,
    ):
        group.equip_with_metric(self.Metric)

        basis = group.metric.normal_basis(group.lie_algebra.basis)

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = group.metric.exp(tangent_vec, identity)
        expected = group.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

    def test_integrated_se3_exp_at_id(self, group):
        group.equip_with_metric(self.Metric)

        lie_algebra = group.lie_algebra
        canonical_metric = group.default_metric()(group)

        basis = group.metric.normal_basis(lie_algebra.basis)

        vector = gs.random.rand(len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity
        result = group.metric.exp(tangent_vec, identity)
        expected = canonical_metric.exp(tangent_vec, identity)
        self.assertAllClose(expected, result, atol=1e-4)

    @tests.conftest.autograd_and_torch_only
    def test_integrated_exp_and_log_at_id(self, group):
        group.equip_with_metric(self.Metric)

        basis = group.lie_algebra.basis

        vector = gs.random.rand(2, len(basis))
        tangent_vec = gs.einsum("...j,jkl->...kl", vector, basis)
        identity = group.identity

        exp = group.metric.exp(tangent_vec, identity)
        result = group.metric.log(exp, identity)
        self.assertAllClose(tangent_vec, result, atol=1e-5)

    def test_integrated_parallel_transport(self, group, n, n_samples):
        group.equip_with_metric(self.Metric)

        point = group.identity
        tan_b = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_b = group.to_tangent(tan_b)

        # use a vector orthonormal to tan_b
        tan_a = Matrices(n + 1, n + 1).random_point(n_samples)
        tan_a = group.to_tangent(tan_a)
        coef = group.metric.inner_product(tan_a, tan_b) / group.metric.squared_norm(
            tan_b
        )
        tan_a -= gs.einsum("...,...ij->...ij", coef, tan_b)
        tan_b = gs.einsum(
            "...ij,...->...ij", tan_b, 1.0 / group.metric.norm(tan_b, base_point=point)
        )
        tan_a = gs.einsum(
            "...ij,...->...ij", tan_a, 1.0 / group.metric.norm(tan_a, base_point=point)
        )

        left_canonical_metric = group.default_metric()(group)
        expected = left_canonical_metric.parallel_transport(tan_a, point, tan_b)
        result, end_point_result = group.metric.parallel_transport(
            tan_a, point, tan_b, n_steps=20, step="rk4", return_endpoint=True
        )
        expected_end_point = group.metric.exp(tan_b, point)

        self.assertAllClose(end_point_result, expected_end_point, atol=gs.atol * 1000)
        self.assertAllClose(expected, result, atol=gs.atol * 1000)

    def test_log_antipodals(self, group, rotation_mat1, rotation_mat2, expected):
        group.equip_with_metric(self.Metric)
        with expected:
            group.metric.log(rotation_mat1, rotation_mat2)

    def test_left_exp_and_exp_from_identity_left_diag_metrics(
        self, group, metric_args, point
    ):
        group.equip_with_metric(self.Metric, **metric_args)

        left_exp_from_id = group.metric.left_exp_from_identity(point)
        exp_from_id = group.metric.exp_from_identity(point)

        self.assertAllClose(left_exp_from_id, exp_from_id)

    def test_left_log_and_log_from_identity_left_diag_metrics(
        self, group, metric_args, point
    ):
        group.equip_with_metric(self.Metric, **metric_args)

        left_log_from_id = group.metric.left_log_from_identity(point)
        log_from_id = group.metric.log_from_identity(point)
        self.assertAllClose(left_log_from_id, log_from_id)

    def test_exp_log_composition_at_identity(self, group, metric_args, tangent_vec):
        group.equip_with_metric(self.Metric, **metric_args)

        result = group.metric.left_log_from_identity(
            point=group.metric.left_exp_from_identity(tangent_vec=tangent_vec)
        )
        self.assertAllClose(result, tangent_vec)

    def test_log_exp_composition_at_identity(self, group, metric_args, point):
        group.equip_with_metric(self.Metric, **metric_args)

        result = group.metric.left_exp_from_identity(
            tangent_vec=group.metric.left_log_from_identity(point=point)
        )
        self.assertAllClose(result, point)
