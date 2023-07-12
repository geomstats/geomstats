import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer, TestCase, pytorch_backend
from tests.data.special_orthogonal_data import (
    BiInvariantMetricTestData,
    InvariantMetricTestData,
    SpecialOrthogonal3TestData,
    SpecialOrthogonalTestData,
)
from tests.geometry_test_cases import InvariantMetricTestCase, LieGroupTestCase

EPSILON = 1e-5


class TestSpecialOrthogonal(LieGroupTestCase, metaclass=Parametrizer):
    skip_test_exp_after_log = pytorch_backend()
    skip_test_projection_belongs = True
    skip_test_random_tangent_vec_is_tangent = True
    skip_test_to_tangent_at_identity_belongs_to_lie_algebra = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SpecialOrthogonalTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.Space(n).belongs(mat), expected)

    def test_dim(self, n, expected):
        self.assertAllClose(self.Space(n).dim, expected)

    def test_identity(self, n, point_type, expected):
        self.assertAllClose(self.Space(n, point_type).identity, expected)

    def test_is_tangent(self, n, vec, base_point, expected):
        group = self.Space(n)
        self.assertAllClose(group.is_tangent(vec, base_point), expected)

    def test_skew_to_vector_and_vector_to_skew(self, n, point_type, vec):
        group = self.Space(n, point_type)
        mat = group.skew_matrix_from_vector(vec)
        result = group.vector_from_skew_matrix(mat)
        self.assertAllClose(result, vec)

    def test_are_antipodals(self, n, mat1, mat2, expected):
        group = self.Space(n)
        self.assertAllClose(group.are_antipodals(mat1, mat2), expected)

    def test_log_at_antipodals_value_error(self, n, point, base_point, expected):
        group = self.Space(n)
        with expected:
            group.log(point, base_point)

    def test_from_vector_from_matrix(self, n, n_samples):
        group = self.Space(n)
        groupvec = self.Space(n, point_type="vector")
        point = groupvec.random_point(n_samples)
        rot_mat = group.matrix_from_rotation_vector(point)
        self.assertAllClose(
            group.rotation_vector_from_matrix(rot_mat), group.regularize(point)
        )

    def test_rotation_vector_from_matrix(self, n, point_type, point, expected):
        group = self.Space(n, point_type)
        self.assertAllClose(group.rotation_vector_from_matrix(point), expected)

    def test_projection(self, n, point_type, mat, expected):
        group = self.Space(n=n, point_type=point_type)
        self.assertAllClose(group.projection(mat), expected)

    def test_projection_shape(self, n, point_type, n_samples, expected):
        group = self.Space(n=n, point_type=point_type)
        self.assertAllClose(
            gs.shape(group.projection(group.random_point(n_samples))), expected
        )

    def test_skew_matrix_from_vector(self, n, vec, expected):
        group = self.Space(n=n, point_type="vector")
        self.assertAllClose(group.skew_matrix_from_vector(vec), expected)

    def test_rotation_vector_rotation_matrix_regularize(self, n, point):
        group = SpecialOrthogonal(n=n)
        rot_mat = group.matrix_from_rotation_vector(point)
        self.assertAllClose(
            group.regularize(point),
            group.rotation_vector_from_matrix(rot_mat),
        )

    def test_matrix_from_rotation_vector(self, n, rot_vec, expected):
        group = SpecialOrthogonal(n)
        result = group.matrix_from_rotation_vector(rot_vec)
        self.assertAllClose(result, expected)

    def test_compose_with_inverse_is_identity(self, space_args):
        group = SpecialOrthogonal(*space_args)
        point = gs.squeeze(group.random_point())
        inv_point = group.inverse(point)
        self.assertAllClose(group.compose(point, inv_point), group.identity)

    def test_compose(self, n, point_type, point_a, point_b, expected):
        group = SpecialOrthogonal(n, point_type)
        result = group.compose(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_regularize(self, n, point_type, angle, expected):
        group = SpecialOrthogonal(n, point_type)
        result = group.regularize(angle)
        self.assertAllClose(result, expected)

    def test_exp(self, n, point_type, tangent_vec, base_point, expected):
        group = self.Space(n, point_type)
        result = group.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, n, point_type, point, base_point, expected):
        group = self.Space(n, point_type)
        result = group.log(point=point, base_point=base_point)
        self.assertAllClose(result, expected)

    def test_compose_shape(self, n, point_type, n_samples):
        group = self.Space(n, point_type=point_type)
        n_points_a = group.random_uniform(n_samples=n_samples)
        n_points_b = group.random_uniform(n_samples=n_samples)
        one_point = group.random_uniform(n_samples=1)

        result = group.compose(one_point, n_points_a)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, one_point)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

        result = group.compose(n_points_a, n_points_b)
        self.assertAllClose(gs.shape(result), (n_samples,) + group.shape)

    def test_rotation_vector_and_rotation_matrix(self, n, point_type, rot_vec):
        group = self.Space(n, point_type=point_type)
        rot_mats = group.matrix_from_rotation_vector(rot_vec)
        result = group.rotation_vector_from_matrix(rot_mats)
        expected = group.regularize(rot_vec)
        self.assertAllClose(result, expected)


class TestSpecialOrthogonal3Vectors(TestCase, metaclass=Parametrizer):
    testing_data = SpecialOrthogonal3TestData()
    Space = testing_data.Space

    def test_tait_bryan_angles_matrix(self, extrinsic, zyx, vec, mat):
        group = self.Space(3, point_type="vector")

        mat_from_vec = group.matrix_from_tait_bryan_angles(
            vec, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(mat_from_vec, mat)
        vec_from_mat = group.tait_bryan_angles_from_matrix(
            mat, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(vec_from_mat, vec)

    def test_tait_bryan_angles_quaternion(self, extrinsic, zyx, vec, quat):
        group = self.Space(3, point_type="vector")

        quat_from_vec = group.quaternion_from_tait_bryan_angles(
            vec, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(quat_from_vec, quat)
        vec_from_quat = group.tait_bryan_angles_from_quaternion(
            quat, extrinsic=extrinsic, zyx=zyx
        )
        self.assertAllClose(vec_from_quat, vec)

    def test_quaternion_from_rotation_vector_tait_bryan_angles(
        self, coord, order, point
    ):
        group = self.Space(3, point_type="vector")

        quat = group.quaternion_from_rotation_vector(point)
        tait_bryan_angle = group.tait_bryan_angles_from_quaternion(quat, coord, order)
        result = group.quaternion_from_tait_bryan_angles(tait_bryan_angle, coord, order)
        self.assertAllClose(result, quat)

    def test_tait_bryan_angles_rotation_vector(self, coord, order, point):
        group = self.Space(3, point_type="vector")

        tait_bryan_angle = group.tait_bryan_angles_from_rotation_vector(
            point, coord, order
        )
        result = group.rotation_vector_from_tait_bryan_angles(
            tait_bryan_angle, coord, order
        )
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_quaternion_and_rotation_vector_with_angles_close_to_pi(self, point):
        group = self.Space(3, point_type="vector")

        quaternion = group.quaternion_from_rotation_vector(point)
        result = group.rotation_vector_from_quaternion(quaternion)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_quaternion_and_matrix_with_angles_close_to_pi(self, point):
        group = self.Space(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        quat = group.quaternion_from_matrix(mat)
        result = group.matrix_from_quaternion(quat)
        expected1 = mat
        expected2 = gs.linalg.inv(mat)
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_rotation_vector_and_rotation_matrix_with_angles_close_to_pi(self, point):
        group = self.Space(3, point_type="vector")
        mat = group.matrix_from_rotation_vector(point)
        result = group.rotation_vector_from_matrix(mat)
        expected1 = group.regularize(point)
        expected2 = -1 * expected1
        expected = gs.allclose(result, expected1) or gs.allclose(result, expected2)
        self.assertTrue(expected)

    def test_lie_bracket(self, tangent_vec_a, tangent_vec_b, base_point, expected):
        group = self.Space(3, point_type="vector")
        result = group.lie_bracket(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    def test_group_exp_after_log_with_angles_close_to_pi(self, point, base_point):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        group = self.Space(3, point_type="vector")
        result = group.exp(group.log(point, base_point), base_point)
        expected = group.regularize(point)
        inv_expected = -expected

        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_group_log_after_exp_with_angles_close_to_pi(self, tangent_vec, base_point):
        """
        This tests that the composition of
        log and exp gives identity.
        """
        group = self.Space(3, point_type="vector")
        result = group.log(group.exp(tangent_vec, base_point), base_point)
        reg_tangent_vec = group.regularize_tangent_vec(
            tangent_vec=tangent_vec, base_point=base_point
        )
        expected = reg_tangent_vec
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected, atol=5e-3)
            or gs.allclose(result, inv_expected, atol=5e-3)
        )

    def test_left_jacobian_vectorization(self, n_samples):
        group = self.Space(3, point_type="vector")
        points = group.random_uniform(n_samples=n_samples)
        jacobians = group.jacobian_translation(point=points, left=True)
        self.assertAllClose(gs.shape(jacobians), (n_samples, group.dim, group.dim))

    def test_inverse(self, n_samples):
        group = self.Space(3, point_type="vector")
        points = group.random_uniform(n_samples=n_samples)
        result = group.inverse(points)

        self.assertAllClose(gs.shape(result), (n_samples, group.dim))

    def test_left_jacobian_through_its_determinant(self, point, expected):
        group = self.Space(3, point_type="vector")
        jacobian = group.jacobian_translation(point=point, left=True)
        result = gs.linalg.det(jacobian)
        self.assertAllClose(result, expected)

    def test_compose_and_inverse(self, point):
        group = self.Space(3, point_type="vector")
        inv_point = group.inverse(point)
        result = group.compose(point, inv_point)
        expected = group.identity
        self.assertAllClose(result, expected)
        result = group.compose(inv_point, point)
        self.assertAllClose(result, expected)

    def test_compose_regularize(self, point):
        group = self.Space(3, point_type="vector")
        result = group.compose(point, group.identity)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

        result = group.compose(group.identity, point)
        expected = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_compose_regularize_angles_close_to_pi(self, point):
        group = self.Space(3, point_type="vector")
        result = group.compose(point, group.identity)
        expected = group.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )

        result = group.compose(group.identity, point)
        expected = group.regularize(point)
        inv_expected = -expected
        self.assertTrue(
            gs.allclose(result, expected) or gs.allclose(result, inv_expected)
        )

    @tests.conftest.np_and_autograd_only
    def test_regularize_extreme_cases(self, point, expected):
        group = SpecialOrthogonal(3, "vector")
        result = group.regularize(point)
        self.assertAllClose(result, expected)

    def test_regularize(self, point, expected):
        group = SpecialOrthogonal(3, "vector")
        result = group.regularize(point)
        self.assertAllClose(result, expected)


class TestBiInvariantMetric(InvariantMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_log_after_exp_at_identity = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = BiInvariantMetricTestData()

    def test_squared_dist_is_less_than_squared_pi(self, group, point_1, point_2):
        """
        This test only concerns the canonical metric.
        For other metrics, the scaling factor can give
        distances above pi.
        """
        group.equip_with_metric(self.Metric)
        point_1 = group.regularize(point_1)
        point_2 = group.regularize(point_2)

        sq_dist = group.metric.squared_dist(point_1, point_2)
        diff = sq_dist - gs.pi**2
        self.assertTrue(diff <= 0 or abs(diff) < EPSILON, f"sq_dist = {sq_dist}")

    def test_exp(self, group, tangent_vec, base_point, expected):
        group.equip_with_metric(self.Metric)
        result = group.metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, group, point, base_point, expected):
        group.equip_with_metric(self.Metric)
        result = group.metric.log(point, base_point)
        self.assertAllClose(result, expected)

    @tests.conftest.np_and_autograd_only
    def test_distance_broadcast(self, group):
        group.equip_with_metric(self.Metric)

        point = group.random_point(5)
        result = group.metric.dist_broadcast(point[:3], point)
        expected = []
        for a in point[:3]:
            expected.append(group.metric.dist(a, point))
        expected = gs.stack(expected)
        self.assertAllClose(result, expected)


class TestInvariantMetricOnSO3(TestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = InvariantMetricTestData()
    Metric = testing_data.Metric

    def test_squared_dist_is_symmetric(
        self, group, metric_mat_at_identity, left, point_1, point_2
    ):
        group.equip_with_metric(
            self.Metric, metric_mat_at_identity=metric_mat_at_identity, left=left
        )
        point_1 = group.regularize(point_1)
        point_2 = group.regularize(point_2)

        sq_dist_1_2 = gs.mod(
            group.metric.squared_dist(point_1, point_2) + 1e-4, gs.pi**2
        )
        sq_dist_2_1 = gs.mod(
            group.metric.squared_dist(point_2, point_1) + 1e-4, gs.pi**2
        )
        self.assertAllClose(sq_dist_1_2, sq_dist_2_1, atol=1e-4)
